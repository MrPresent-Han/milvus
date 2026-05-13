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
	"fmt"
	"sort"

	"github.com/apache/arrow/go/v17/arrow"
	"github.com/apache/arrow/go/v17/arrow/array"
	"github.com/apache/arrow/go/v17/arrow/memory"
	"github.com/cockroachdb/errors"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/internal/storagev2/packed"
	"github.com/milvus-io/milvus/internal/util/function"
	"github.com/milvus-io/milvus/pkg/v3/common"
	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
	"github.com/milvus-io/milvus/pkg/v3/proto/indexpb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

type FunctionMaterializer interface {
	Materialize(rec storage.Record) (map[int64]arrow.Array, error)
	Close()
}

type RecordMaterializer struct {
	materializers []FunctionMaterializer
	missingFields []*schemapb.FieldSchema
}

func NewRecordMaterializer(schema *schemapb.CollectionSchema, functions []*schemapb.FunctionSchema, existingFields map[int64]struct{}) (*RecordMaterializer, error) {
	materializer := &RecordMaterializer{}
	materializedFields := make(map[int64]struct{})
	for _, functionSchema := range functions {
		missingOutputIndexes := missingFunctionOutputIndexes(functionSchema, existingFields)
		if len(missingOutputIndexes) == 0 {
			continue
		}
		for _, outputIndex := range missingOutputIndexes {
			materializedFields[functionSchema.GetOutputFieldIds()[outputIndex]] = struct{}{}
		}

		runner, err := function.NewFunctionRunner(schema, functionSchema)
		if err != nil {
			materializer.Close()
			return nil, err
		}
		if runner == nil {
			materializer.Close()
			return nil, errors.Newf("failed to set up function runner for %s", functionSchema.GetName())
		}
		functionMaterializer, err := newFunctionMaterializer(schema, runner, missingOutputIndexes, true)
		if err != nil {
			runner.Close()
			materializer.Close()
			return nil, err
		}
		materializer.materializers = append(materializer.materializers, functionMaterializer)
	}
	materializer.missingFields = missingNonMaterializedSchemaFields(schema, existingFields, materializedFields)
	return materializer, nil
}

func NewRecordMaterializerWithRunners(schema *schemapb.CollectionSchema, runners []function.FunctionRunner, existingFields map[int64]struct{}) (*RecordMaterializer, error) {
	materializer := &RecordMaterializer{}
	materializedFields := make(map[int64]struct{})
	for _, runner := range runners {
		if runner == nil {
			materializer.Close()
			return nil, errors.New("function runner is nil")
		}
		functionSchema := runner.GetSchema()
		missingOutputIndexes := missingFunctionOutputIndexes(functionSchema, existingFields)
		if len(missingOutputIndexes) == 0 {
			continue
		}
		for _, outputIndex := range missingOutputIndexes {
			materializedFields[functionSchema.GetOutputFieldIds()[outputIndex]] = struct{}{}
		}

		functionMaterializer, err := newFunctionMaterializer(schema, runner, missingOutputIndexes, false)
		if err != nil {
			materializer.Close()
			return nil, err
		}
		materializer.materializers = append(materializer.materializers, functionMaterializer)
	}
	materializer.missingFields = missingNonMaterializedSchemaFields(schema, existingFields, materializedFields)
	return materializer, nil
}

func (m *RecordMaterializer) Wrap(rec storage.Record) (storage.Record, error) {
	if !m.hasMaterialization() {
		return rec, nil
	}

	computed := make(map[int64]arrow.Array)
	for _, materializer := range m.materializers {
		arrays, err := materializer.Materialize(rec)
		if err != nil {
			releaseArrowArrays(computed)
			return nil, err
		}
		for fieldID, arr := range arrays {
			computed[fieldID] = arr
		}
	}
	for _, field := range m.missingFields {
		fieldID := field.GetFieldID()
		if _, ok := computed[fieldID]; ok {
			continue
		}
		arr, err := storage.GenerateEmptyArrayFromSchema(field, rec.Len())
		if err != nil {
			releaseArrowArrays(computed)
			return nil, err
		}
		computed[fieldID] = arr
	}
	if len(computed) == 0 {
		return rec, nil
	}
	return &materializedRecord{base: rec, computed: computed}, nil
}

func (m *RecordMaterializer) Close() {
	if m == nil {
		return
	}
	for _, materializer := range m.materializers {
		materializer.Close()
	}
}

func (m *RecordMaterializer) hasMaterialization() bool {
	return m != nil && (len(m.materializers) > 0 || len(m.missingFields) > 0)
}

type materializedRecord struct {
	base     storage.Record
	computed map[int64]arrow.Array
}

var _ storage.Record = (*materializedRecord)(nil)

func (r *materializedRecord) Column(fieldID storage.FieldID) arrow.Array {
	if col, ok := r.computed[int64(fieldID)]; ok {
		return col
	}
	return r.base.Column(fieldID)
}

func (r *materializedRecord) Len() int {
	return r.base.Len()
}

func (r *materializedRecord) Retain() {
	r.base.Retain()
	for _, col := range r.computed {
		col.Retain()
	}
}

func (r *materializedRecord) Release() {
	r.base.Release()
	for _, col := range r.computed {
		col.Release()
	}
}

type materializedRecordReader struct {
	base         storage.RecordReader
	materializer *RecordMaterializer
	current      storage.Record
}

var _ storage.RecordReader = (*materializedRecordReader)(nil)

func newMaterializedRecordReader(base storage.RecordReader, materializer *RecordMaterializer) storage.RecordReader {
	if !materializer.hasMaterialization() {
		return base
	}
	return &materializedRecordReader{base: base, materializer: materializer}
}

func (r *materializedRecordReader) Next() (storage.Record, error) {
	if r.current != nil {
		r.current.Release()
		r.current = nil
	}
	rec, err := r.base.Next()
	if err != nil {
		return nil, err
	}
	wrapped, err := r.materializer.Wrap(rec)
	if err != nil {
		rec.Release()
		return nil, err
	}
	r.current = wrapped
	return wrapped, nil
}

func (r *materializedRecordReader) Close() error {
	if r.current != nil {
		r.current.Release()
		r.current = nil
	}
	r.materializer.Close()
	return r.base.Close()
}

type bm25FunctionMaterializer struct {
	runner               function.FunctionRunner
	inputFieldID         int64
	outputFieldIDs       []int64
	missingOutputIndexes []int
	outputFields         map[int64]*schemapb.FieldSchema
	ownRunner            bool
}

var _ FunctionMaterializer = (*bm25FunctionMaterializer)(nil)

func newFunctionMaterializer(schema *schemapb.CollectionSchema, runner function.FunctionRunner, missingOutputIndexes []int, ownRunner bool) (FunctionMaterializer, error) {
	functionSchema := runner.GetSchema()
	switch functionSchema.GetType() {
	case schemapb.FunctionType_BM25:
		return newBM25FunctionMaterializer(schema, runner, missingOutputIndexes, ownRunner)
	default:
		return nil, errors.Newf("unsupported function type %s", functionSchema.GetType().String())
	}
}

func newBM25FunctionMaterializer(schema *schemapb.CollectionSchema, runner function.FunctionRunner, missingOutputIndexes []int, ownRunner bool) (*bm25FunctionMaterializer, error) {
	functionSchema := runner.GetSchema()
	inputFieldIDs := functionSchema.GetInputFieldIds()
	if len(inputFieldIDs) != 1 {
		return nil, errors.New("bm25 function should have exactly one input field")
	}
	inputField := typeutil.GetField(schema, inputFieldIDs[0])
	if inputField == nil {
		return nil, errors.New("input field not found in schema")
	}
	if inputField.GetDataType() != schemapb.DataType_VarChar && inputField.GetDataType() != schemapb.DataType_Text {
		return nil, errors.New("input field data type must be varchar or text for bm25 function materialization")
	}

	outputFieldIDs := functionSchema.GetOutputFieldIds()
	if len(outputFieldIDs) == 0 {
		return nil, errors.New("bm25 function should have output fields")
	}

	outputFields := make(map[int64]*schemapb.FieldSchema, len(outputFieldIDs))
	for _, outputFieldID := range outputFieldIDs {
		outputField := typeutil.GetField(schema, outputFieldID)
		if outputField == nil {
			return nil, errors.New("output field not found in schema")
		}
		if outputField.GetDataType() != schemapb.DataType_SparseFloatVector {
			return nil, errors.New("output field data type must be sparse float vector for bm25 function materialization")
		}
		outputFields[outputFieldID] = outputField
	}

	return &bm25FunctionMaterializer{
		runner:               runner,
		inputFieldID:         inputFieldIDs[0],
		outputFieldIDs:       outputFieldIDs,
		missingOutputIndexes: missingOutputIndexes,
		outputFields:         outputFields,
		ownRunner:            ownRunner,
	}, nil
}

func (m *bm25FunctionMaterializer) Materialize(rec storage.Record) (map[int64]arrow.Array, error) {
	inputs, err := stringInputsFromRecord(rec, m.inputFieldID)
	if err != nil {
		return nil, err
	}
	outputs, err := m.runner.BatchRun(inputs)
	if err != nil {
		return nil, err
	}
	if len(outputs) != len(m.outputFieldIDs) {
		return nil, errors.Newf("bm25 function materialization expects %d outputs, got %d", len(m.outputFieldIDs), len(outputs))
	}

	result := make(map[int64]arrow.Array, len(m.missingOutputIndexes))
	for _, outputIndex := range m.missingOutputIndexes {
		outputFieldID := m.outputFieldIDs[outputIndex]
		outputSparseArray, ok := outputs[outputIndex].(*schemapb.SparseFloatArray)
		if !ok {
			releaseArrowArrays(result)
			return nil, errors.Newf("unexpected output type from BM25 function runner, expected SparseFloatArray, got %T", outputs[outputIndex])
		}
		arr, err := buildSparseFloatVectorArrowArray(m.outputFields[outputFieldID], outputSparseArray)
		if err != nil {
			releaseArrowArrays(result)
			return nil, err
		}
		result[outputFieldID] = arr
	}
	return result, nil
}

func (m *bm25FunctionMaterializer) Close() {
	if m.ownRunner && m.runner != nil {
		m.runner.Close()
	}
}

func missingFunctionOutputIndexes(functionSchema *schemapb.FunctionSchema, existingFields map[int64]struct{}) []int {
	missing := make([]int, 0, len(functionSchema.GetOutputFieldIds()))
	for idx, outputFieldID := range functionSchema.GetOutputFieldIds() {
		if _, ok := existingFields[outputFieldID]; !ok {
			missing = append(missing, idx)
		}
	}
	return missing
}

func missingNonMaterializedSchemaFields(schema *schemapb.CollectionSchema, existingFields map[int64]struct{}, materializedFields map[int64]struct{}) []*schemapb.FieldSchema {
	missing := make([]*schemapb.FieldSchema, 0)
	for _, field := range typeutil.GetAllFieldSchemas(schema) {
		fieldID := field.GetFieldID()
		if _, ok := existingFields[fieldID]; ok {
			continue
		}
		if _, ok := materializedFields[fieldID]; ok {
			continue
		}
		missing = append(missing, field)
	}
	return missing
}

func stringInputsFromRecord(rec storage.Record, fieldID int64) ([]string, error) {
	col := rec.Column(fieldID)
	if col == nil {
		return nil, merr.WrapErrServiceInternal(fmt.Sprintf("input field %d not found in record", fieldID))
	}
	inputs := make([]string, rec.Len())
	switch values := col.(type) {
	case *array.String:
		for i := 0; i < rec.Len(); i++ {
			inputs[i] = values.Value(i)
		}
	case *array.Binary:
		for i := 0; i < rec.Len(); i++ {
			inputs[i] = string(values.Value(i))
		}
	default:
		return nil, merr.WrapErrServiceInternal(fmt.Sprintf("input field %d data type must be varchar or text for bm25 function materialization, got %T", fieldID, col))
	}
	return inputs, nil
}

func buildSparseFloatVectorArrowArray(field *schemapb.FieldSchema, outputSparseArray *schemapb.SparseFloatArray) (arrow.Array, error) {
	outputSchema := &schemapb.CollectionSchema{Fields: []*schemapb.FieldSchema{field}}
	arrowSchema, err := storage.ConvertToArrowSchema(outputSchema, true)
	if err != nil {
		return nil, err
	}
	builder := array.NewRecordBuilder(memory.DefaultAllocator, arrowSchema)
	defer builder.Release()

	insertData := &storage.InsertData{Data: map[int64]storage.FieldData{
		field.GetFieldID(): &storage.SparseFloatVectorFieldData{
			SparseFloatArray: schemapb.SparseFloatArray{
				Contents: outputSparseArray.GetContents(),
				Dim:      outputSparseArray.GetDim(),
			},
		},
	}}
	if err := storage.BuildRecord(builder, insertData, outputSchema); err != nil {
		return nil, err
	}
	record := builder.NewRecord()
	defer record.Release()

	col := record.Column(0)
	col.Retain()
	return col, nil
}

func releaseArrowArrays(arrays map[int64]arrow.Array) {
	for _, arr := range arrays {
		arr.Release()
	}
}

func releaseWrappedRecord(wrapped storage.Record, base storage.Record) {
	if wrapped != base {
		wrapped.Release()
		return
	}
	base.Release()
}

func compactionSegmentStorageFields(segment *datapb.CompactionSegmentBinlogs, storageConfig *indexpb.StorageConfig) (map[int64]struct{}, error) {
	if segment.GetManifest() != "" {
		return packed.GetManifestFieldIDs(segment.GetManifest(), storageConfig)
	}
	return compactionSegmentBinlogFields(segment), nil
}

func compactionSegmentBinlogFields(segment *datapb.CompactionSegmentBinlogs) map[int64]struct{} {
	fields := make(map[int64]struct{})
	for _, fieldBinlog := range segment.GetFieldBinlogs() {
		if len(fieldBinlog.GetChildFields()) == 0 {
			fields[fieldBinlog.GetFieldID()] = struct{}{}
			continue
		}
		for _, childFieldID := range fieldBinlog.GetChildFields() {
			fields[childFieldID] = struct{}{}
		}
	}
	return fields
}

func collectionSchemaFields(schema *schemapb.CollectionSchema) map[int64]struct{} {
	fields := make(map[int64]struct{}, typeutil.GetTotalFieldsNum(schema))
	for _, field := range typeutil.GetAllFieldSchemas(schema) {
		fields[field.GetFieldID()] = struct{}{}
	}
	return fields
}

func missingSchemaFunctions(schema *schemapb.CollectionSchema, existingFields map[int64]struct{}) []*schemapb.FunctionSchema {
	var missing []*schemapb.FunctionSchema
	for _, functionSchema := range schema.GetFunctions() {
		for _, outputFieldID := range functionSchema.GetOutputFieldIds() {
			if _, ok := existingFields[outputFieldID]; !ok {
				missing = append(missing, functionSchema)
				break
			}
		}
	}
	return missing
}

func droppedSchemaFieldIDs(schema *schemapb.CollectionSchema, existingFields map[int64]struct{}) []int64 {
	targetFields := collectionSchemaFields(schema)
	dropped := make([]int64, 0)
	for fieldID := range existingFields {
		if fieldID < common.StartOfUserFieldID {
			continue
		}
		if _, ok := targetFields[fieldID]; !ok {
			dropped = append(dropped, fieldID)
		}
	}
	sort.Slice(dropped, func(i, j int) bool { return dropped[i] < dropped[j] })
	return dropped
}

func segmentDroppedFieldIDs(schema *schemapb.CollectionSchema, segment *datapb.CompactionSegmentBinlogs, storageConfig *indexpb.StorageConfig) ([]int64, error) {
	existingFields, err := compactionSegmentStorageFields(segment, storageConfig)
	if err != nil {
		return nil, err
	}
	return droppedSchemaFieldIDs(schema, existingFields), nil
}
