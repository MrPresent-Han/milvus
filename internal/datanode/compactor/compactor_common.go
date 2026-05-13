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

package compactor

import (
	"context"
	"strconv"
	"time"

	"go.opentelemetry.io/otel"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/allocator"
	"github.com/milvus-io/milvus/internal/metastore/kv/binlog"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/pkg/v3/common"
	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
	"github.com/milvus-io/milvus/pkg/v3/proto/indexpb"
	"github.com/milvus-io/milvus/pkg/v3/util/tsoutil"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

const compactionBatchSize = 100

// Storage readers do not share compaction's schema-reconciliation contract, so filter physical fields before opening them.
func newCompactionSegmentRecordReader(ctx context.Context, segment *datapb.CompactionSegmentBinlogs, schema *schemapb.CollectionSchema, storageConfig *indexpb.StorageConfig, opts ...storage.RwOption) (storage.RecordReader, map[int64]struct{}, error) {
	existingFields, err := compactionSegmentStorageFields(segment, storageConfig)
	if err != nil {
		return nil, nil, err
	}
	return newCompactionSegmentRecordReaderWithFields(ctx, segment, schema, storageConfig, existingFields, opts...)
}

func newCompactionSegmentRecordReaderWithFields(ctx context.Context, segment *datapb.CompactionSegmentBinlogs, schema *schemapb.CollectionSchema, storageConfig *indexpb.StorageConfig, existingFields map[int64]struct{}, opts ...storage.RwOption) (storage.RecordReader, map[int64]struct{}, error) {
	readSchema := compactionReadSchema(schema, existingFields)

	if segment.GetManifest() != "" {
		reader, err := storage.NewManifestRecordReader(ctx, segment.GetManifest(), readSchema, opts...)
		return reader, existingFields, err
	}

	readFields := collectionSchemaFields(readSchema)
	fieldBinlogs := filterCompactionFieldBinlogs(segment.GetFieldBinlogs(), readFields)
	rootPath := ""
	if storageConfig != nil {
		rootPath = storageConfig.GetRootPath()
	}
	if err := binlog.DecompressBinLogWithRootPath(rootPath, storage.InsertBinlog,
		segment.GetCollectionID(), segment.GetPartitionID(), segment.GetSegmentID(), fieldBinlogs); err != nil {
		return nil, nil, err
	}

	reader, err := storage.NewBinlogRecordReader(ctx, fieldBinlogs, readSchema, opts...)
	return reader, existingFields, err
}

func compactionReadSchema(schema *schemapb.CollectionSchema, existingFields map[int64]struct{}) *schemapb.CollectionSchema {
	if schema == nil {
		return nil
	}
	readSchema := proto.Clone(schema).(*schemapb.CollectionSchema)

	fields := make([]*schemapb.FieldSchema, 0, len(readSchema.GetFields()))
	for _, field := range readSchema.GetFields() {
		if compactionFieldReadable(field, existingFields) {
			fields = append(fields, field)
		}
	}
	readSchema.Fields = fields

	structFields := make([]*schemapb.StructArrayFieldSchema, 0, len(readSchema.GetStructArrayFields()))
	for _, structField := range readSchema.GetStructArrayFields() {
		childFields := make([]*schemapb.FieldSchema, 0, len(structField.GetFields()))
		for _, field := range structField.GetFields() {
			if compactionFieldReadable(field, existingFields) {
				childFields = append(childFields, field)
			}
		}
		if len(childFields) > 0 {
			structField.Fields = childFields
			structFields = append(structFields, structField)
		}
	}
	readSchema.StructArrayFields = structFields
	return readSchema
}

func compactionFieldReadable(field *schemapb.FieldSchema, existingFields map[int64]struct{}) bool {
	_, ok := existingFields[field.GetFieldID()]
	return ok
}

func filterCompactionFieldBinlogs(fieldBinlogs []*datapb.FieldBinlog, readFields map[int64]struct{}) []*datapb.FieldBinlog {
	filtered := make([]*datapb.FieldBinlog, 0, len(fieldBinlogs))
	for _, fieldBinlog := range fieldBinlogs {
		if compactionFieldBinlogReadable(fieldBinlog, readFields) {
			filtered = append(filtered, fieldBinlog)
		}
	}
	return filtered
}

func compactionFieldBinlogReadable(fieldBinlog *datapb.FieldBinlog, readFields map[int64]struct{}) bool {
	if fieldBinlog == nil {
		return false
	}
	if _, ok := readFields[fieldBinlog.GetFieldID()]; ok {
		return true
	}
	for _, childFieldID := range fieldBinlog.GetChildFields() {
		if _, ok := readFields[childFieldID]; ok {
			return true
		}
	}
	return false
}

type EntityFilter struct {
	deletedPkTs map[interface{}]typeutil.Timestamp // pk2ts
	ttl         int64                              // nanoseconds
	currentTime time.Time

	expiredCount int
	deletedCount int
}

func newEntityFilter(deletedPkTs map[interface{}]typeutil.Timestamp, ttl int64, currTime time.Time) *EntityFilter {
	if deletedPkTs == nil {
		deletedPkTs = make(map[interface{}]typeutil.Timestamp)
	}
	return &EntityFilter{
		deletedPkTs: deletedPkTs,
		ttl:         ttl,
		currentTime: currTime,
	}
}

func (filter *EntityFilter) Filtered(pk any, ts typeutil.Timestamp) bool {
	if filter.isEntityDeleted(pk, ts) {
		filter.deletedCount++
		return true
	}

	// Filtering expired entity
	if filter.isEntityExpired(ts) {
		filter.expiredCount++
		return true
	}
	return false
}

func (filter *EntityFilter) GetExpiredCount() int {
	return filter.expiredCount
}

func (filter *EntityFilter) GetDeletedCount() int {
	return filter.deletedCount
}

func (filter *EntityFilter) GetDeltalogDeleteCount() int {
	return len(filter.deletedPkTs)
}

func (filter *EntityFilter) GetMissingDeleteCount() int {
	diff := filter.GetDeltalogDeleteCount() - filter.GetDeletedCount()
	if diff <= 0 {
		diff = 0
	}
	return diff
}

func (filter *EntityFilter) isEntityDeleted(pk interface{}, pkTs typeutil.Timestamp) bool {
	if deleteTs, ok := filter.deletedPkTs[pk]; ok {
		// insert task and delete task has the same ts when upsert
		// here should be < instead of <=
		// to avoid the upsert data to be deleted after compact
		if pkTs < deleteTs {
			return true
		}
	}
	return false
}

func (filter *EntityFilter) isEntityExpired(entityTs typeutil.Timestamp) bool {
	// entity expire is not enabled if duration <= 0
	if filter.ttl <= 0 {
		return false
	}
	entityTime, _ := tsoutil.ParseTS(entityTs)

	// this dur can represents 292 million years before or after 1970, enough for milvus
	// ttl calculation
	dur := filter.currentTime.UnixMilli() - entityTime.UnixMilli()

	// filter.ttl is nanoseconds
	return filter.ttl/int64(time.Millisecond) <= dur
}

// TODO: remove, used in test only
func serializeWrite(ctx context.Context, allocator allocator.Interface, writer *SegmentWriter) (kvs map[string][]byte, fieldBinlogs map[int64]*datapb.FieldBinlog, err error) {
	_, span := otel.Tracer(typeutil.DataNodeRole).Start(ctx, "serializeWrite")
	defer span.End()

	blobs, tr, err := writer.SerializeYield()
	startID, _, err := allocator.Alloc(uint32(len(blobs)))
	if err != nil {
		return nil, nil, err
	}

	kvs = make(map[string][]byte)
	fieldBinlogs = make(map[int64]*datapb.FieldBinlog)
	for i := range blobs {
		// Blob Key is generated by Serialize from int64 fieldID in collection schema, which won't raise error in ParseInt
		fID, _ := strconv.ParseInt(blobs[i].GetKey(), 10, 64)
		key, _ := binlog.BuildLogPath(storage.InsertBinlog, writer.GetCollectionID(), writer.GetPartitionID(), writer.GetSegmentID(), fID, startID+int64(i))

		kvs[key] = blobs[i].GetValue()
		fieldBinlogs[fID] = &datapb.FieldBinlog{
			FieldID: fID,
			Binlogs: []*datapb.Binlog{
				{
					LogSize:       int64(len(blobs[i].GetValue())),
					MemorySize:    blobs[i].GetMemorySize(),
					LogPath:       key,
					EntriesNum:    blobs[i].RowNum,
					TimestampFrom: tr.GetMinTimestamp(),
					TimestampTo:   tr.GetMaxTimestamp(),
				},
			},
		}
	}

	return
}

func mergeFieldBinlogs(base, paths map[typeutil.UniqueID]*datapb.FieldBinlog) {
	for fID, fpath := range paths {
		if _, ok := base[fID]; !ok {
			base[fID] = &datapb.FieldBinlog{FieldID: fID, Binlogs: make([]*datapb.Binlog, 0)}
		}
		base[fID].Binlogs = append(base[fID].Binlogs, fpath.GetBinlogs()...)
	}
}

func getTTLFieldID(schema *schemapb.CollectionSchema) int64 {
	ttlFieldName := ""
	for _, pair := range schema.GetProperties() {
		if pair.GetKey() == common.CollectionTTLFieldKey {
			ttlFieldName = pair.GetValue()
			break
		}
	}
	if ttlFieldName == "" {
		return -1
	}
	for _, field := range schema.GetFields() {
		if field.GetName() == ttlFieldName && field.GetDataType() == schemapb.DataType_Timestamptz {
			return field.GetFieldID()
		}
	}
	return -1
}
