// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package rootcoord

import (
	"context"
	"fmt"

	"go.uber.org/zap"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/internal/metastore/model"
	"github.com/milvus-io/milvus/internal/util/proxyutil"
	"github.com/milvus-io/milvus/pkg/v2/log"
)

type addCollectionFunctionFieldTask struct {
	baseTask
	Req            *milvuspb.AddCollectionFunctionFieldRequest
	fieldInfos     []*milvuspb.AddCollectionFunctionFieldRequest_FieldInfo
	functionSchema *schemapb.FunctionSchema
}

func (t *addCollectionFunctionFieldTask) Prepare(ctx context.Context) error {
	if err := CheckMsgType(t.Req.GetBase().GetMsgType(), commonpb.MsgType_AddCollectionFunctionField); err != nil {
		return err
	}

	t.fieldInfos = t.Req.GetFieldInfos()
	t.functionSchema = t.Req.GetFuncSchema()
	log.Info("hc===add function field", zap.Any("functionSchema", t.functionSchema), zap.Any("fieldInfos", t.fieldInfos))
	// if err := checkFieldSchemaForFunction(t.fieldSchema, t.Req.GetFuncSchema()); err != nil {
	// 	return err
	// }
	// hc---skip check for now
	return nil
}

func (t *addCollectionFunctionFieldTask) Execute(ctx context.Context) error {
	oldColl, err := t.core.meta.GetCollectionByName(ctx, t.Req.GetDbName(), t.Req.GetCollectionName(), t.ts)
	if err != nil {
		log.Ctx(ctx).Warn("get collection failed during add function field",
			zap.String("collectionName", t.Req.GetCollectionName()), zap.Uint64("ts", t.ts))
		return err
	}

	fieldIDStart := nextFieldID(oldColl)
	for i, fieldInfo := range t.fieldInfos {
		fieldInfo.FieldSchema.FieldID = fieldIDStart + int64(i)
	}
	t.functionSchema.Id = nextFunctionID(oldColl)
	ts := t.GetTs()
	t.Req.CollectionID = oldColl.CollectionID
	return t.executeAddCollectionFunctionFieldTaskSteps(ctx, oldColl, ts)
}

func (t *addCollectionFunctionFieldTask) GetLockerKey() LockerKey {
	collection := t.core.getCollectionIDStr(t.ctx, t.Req.GetDbName(), t.Req.GetCollectionName(), 0)
	return NewLockerKeyChain(
		NewClusterLockerKey(false),
		NewDatabaseLockerKey(t.Req.GetDbName(), false),
		NewCollectionLockerKey(collection, true),
	)
}

func (t *addCollectionFunctionFieldTask) executeAddCollectionFunctionFieldTaskSteps(ctx context.Context,
	col *model.Collection,
	ts Timestamp,
) error {
	redoTask := newBaseRedoTask(t.core.stepExecutor)
	updatedCollection := col.Clone()

	// 1. convert field infos and function schema to fields and function
	fields := make([]*model.Field, len(t.fieldInfos))
	for i, fieldInfo := range t.fieldInfos {
		fields[i] = model.UnmarshalFieldModel(fieldInfo.FieldSchema)
	}
	updatedCollection.Fields = append(updatedCollection.Fields, fields...)

	name2id := map[string]int64{}
	for _, field := range updatedCollection.Fields {
		name2id[field.Name] = field.FieldID
	}

	t.functionSchema.InputFieldIds = make([]int64, len(t.functionSchema.InputFieldNames))
	for idx, name := range t.functionSchema.InputFieldNames {
		fieldId, ok := name2id[name]
		if !ok {
			return fmt.Errorf("input field %s of function %s not found", name, t.functionSchema.GetName())
		}
		t.functionSchema.InputFieldIds[idx] = fieldId
	}

	t.functionSchema.OutputFieldIds = make([]int64, len(t.functionSchema.OutputFieldNames))
	for idx, name := range t.functionSchema.OutputFieldNames {
		fieldId, ok := name2id[name]
		if !ok {
			return fmt.Errorf("output field %s of function %s not found", name, t.functionSchema.GetName())
		}
		t.functionSchema.OutputFieldIds[idx] = fieldId
	}
	log.Info("hc===add function field",
		zap.Any("functionSchema.inputFieldIds", t.functionSchema.InputFieldIds),
		zap.Any("functionSchema.inputFieldNames", t.functionSchema.InputFieldNames),
		zap.Any("functionSchema.outputFieldNames", t.functionSchema.OutputFieldNames),
		zap.Any("functionSchema.outputFieldIds", t.functionSchema.OutputFieldIds),
		zap.Uint64("ts", ts))

	function := model.UnmarshalFunctionModel(t.functionSchema)
	updatedCollection.Functions = append(updatedCollection.Functions, function)

	// 2. write schema change WAL
	redoTask.AddSyncStep(&WriteSchemaChangeWALStep{
		baseStep:   baseStep{core: t.core},
		collection: updatedCollection,
		ts:         ts,
	})

	// 3. add fields and function to collection meta
	oldColl := col.Clone()
	redoTask.AddSyncStep(&AddCollectionMetaStep{
		baseStep:          baseStep{core: t.core},
		oldColl:           oldColl,
		updatedCollection: updatedCollection,
		newFields:         fields,
		newFunction:       function,
		ts:                ts,
	})

	// 4. broadcast altered collection
	redoTask.AddSyncStep(&BroadcastAlteredCollectionStep{
		baseStep: baseStep{core: t.core},
		req: &milvuspb.AlterCollectionRequest{
			DbName:         t.Req.GetDbName(),
			CollectionName: t.Req.GetCollectionName(),
			CollectionID:   t.Req.GetCollectionID(),
		},
		core: t.core,
	})

	// field needs to be refreshed in the cache
	aliases := t.core.meta.ListAliasesByID(ctx, oldColl.CollectionID)
	redoTask.AddSyncStep(&expireCacheStep{
		baseStep:        baseStep{core: t.core},
		dbName:          t.Req.GetDbName(),
		collectionNames: append(aliases, t.Req.GetCollectionName()),
		collectionID:    oldColl.CollectionID,
		ts:              ts,
		opts:            []proxyutil.ExpireCacheOpt{proxyutil.SetMsgType(commonpb.MsgType_AddCollectionFunctionField)},
	})

	return redoTask.Execute(ctx)
}
