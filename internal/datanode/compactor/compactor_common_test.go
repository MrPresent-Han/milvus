package compactor

import (
	"context"
	"testing"

	"github.com/cockroachdb/errors"
	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/pkg/v3/common"
	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
	"github.com/milvus-io/milvus/pkg/v3/proto/internalpb"
)

func TestCompactionSegmentBinlogFieldsUsesChildFields(t *testing.T) {
	fields := compactionSegmentBinlogFields(&datapb.CompactionSegmentBinlogs{
		FieldBinlogs: []*datapb.FieldBinlog{
			{FieldID: 900, ChildFields: []int64{102, 103}},
			{FieldID: 104},
		},
	})

	require.Contains(t, fields, int64(102))
	require.Contains(t, fields, int64(103))
	require.Contains(t, fields, int64(104))
	require.NotContains(t, fields, int64(900))
}

func TestFilterCompactionFieldBinlogsKeepsChildFieldMatch(t *testing.T) {
	fieldBinlogs := []*datapb.FieldBinlog{
		nil,
		{FieldID: 900, ChildFields: []int64{102, 103}},
		{FieldID: 200},
	}

	filtered := filterCompactionFieldBinlogs(fieldBinlogs, map[int64]struct{}{102: {}})
	require.Len(t, filtered, 1)
	require.EqualValues(t, 900, filtered[0].GetFieldID())
	require.Equal(t, []int64{102, 103}, filtered[0].GetChildFields())
}

func TestCompactionReadSchemaFiltersMissingScalarAndStructChildren(t *testing.T) {
	schema := &schemapb.CollectionSchema{
		Fields: []*schemapb.FieldSchema{
			{FieldID: 100, Name: "pk", DataType: schemapb.DataType_Int64},
			{FieldID: 101, Name: "missing", DataType: schemapb.DataType_Int64},
		},
		StructArrayFields: []*schemapb.StructArrayFieldSchema{
			{
				FieldID: 200,
				Name:    "struct_with_child",
				Fields: []*schemapb.FieldSchema{
					{FieldID: 201, Name: "child_present", DataType: schemapb.DataType_Int64},
					{FieldID: 202, Name: "child_missing", DataType: schemapb.DataType_Int64},
				},
			},
			{
				FieldID: 300,
				Name:    "struct_without_child",
				Fields: []*schemapb.FieldSchema{
					{FieldID: 301, Name: "child_missing", DataType: schemapb.DataType_Int64},
				},
			},
		},
	}

	readSchema := compactionReadSchema(schema, map[int64]struct{}{100: {}, 201: {}})
	require.NotNil(t, readSchema)
	require.Len(t, readSchema.GetFields(), 1)
	require.EqualValues(t, 100, readSchema.GetFields()[0].GetFieldID())
	require.Len(t, readSchema.GetStructArrayFields(), 1)
	require.EqualValues(t, 200, readSchema.GetStructArrayFields()[0].GetFieldID())
	require.Len(t, readSchema.GetStructArrayFields()[0].GetFields(), 1)
	require.EqualValues(t, 201, readSchema.GetStructArrayFields()[0].GetFields()[0].GetFieldID())
}

func TestCompactionReadSchemaNilSchema(t *testing.T) {
	require.Nil(t, compactionReadSchema(nil, map[int64]struct{}{}))
}

func TestMissingSchemaFunctionsAndDroppedFields(t *testing.T) {
	functionSchema := &schemapb.FunctionSchema{
		Name:           "bm25",
		Type:           schemapb.FunctionType_BM25,
		InputFieldIds:  []int64{100},
		OutputFieldIds: []int64{101},
	}
	schema := &schemapb.CollectionSchema{
		Fields: []*schemapb.FieldSchema{
			{FieldID: 100, Name: "text", DataType: schemapb.DataType_VarChar},
			{FieldID: 101, Name: "sparse", DataType: schemapb.DataType_SparseFloatVector},
		},
		Functions: []*schemapb.FunctionSchema{functionSchema},
	}
	droppedUserField := int64(common.StartOfUserFieldID + 1000)
	systemField := int64(common.StartOfUserFieldID - 1)
	existingFields := map[int64]struct{}{
		100:              {},
		droppedUserField: {},
		systemField:      {},
	}

	missingFunctions := missingSchemaFunctions(schema, existingFields)
	require.Len(t, missingFunctions, 1)
	require.Equal(t, functionSchema.GetName(), missingFunctions[0].GetName())

	dropped := droppedSchemaFieldIDs(schema, existingFields)
	require.Equal(t, []int64{droppedUserField}, dropped)
}

func bm25AnalyzerTestSchema() *schemapb.CollectionSchema {
	return &schemapb.CollectionSchema{
		Fields: []*schemapb.FieldSchema{
			{FieldID: 100, Name: "text", DataType: schemapb.DataType_VarChar},
			{FieldID: 101, Name: "sparse", DataType: schemapb.DataType_SparseFloatVector},
		},
		Functions: []*schemapb.FunctionSchema{{
			Name:           "bm25",
			Type:           schemapb.FunctionType_BM25,
			InputFieldIds:  []int64{100},
			OutputFieldIds: []int64{101},
		}},
	}
}

func TestMaterializerNeedsAnalyzerResources(t *testing.T) {
	schema := bm25AnalyzerTestSchema()

	// function output (101) missing -> a runner will be built -> resources needed
	require.True(t, materializerNeedsAnalyzerResources(schema, map[int64]struct{}{100: {}}))
	// function output already present -> no runner -> no resources
	require.False(t, materializerNeedsAnalyzerResources(schema, map[int64]struct{}{100: {}, 101: {}}))
	// no functions at all -> no resources (a match field must NOT gate the materializer lease)
	require.False(t, materializerNeedsAnalyzerResources(&schemapb.CollectionSchema{
		Fields: []*schemapb.FieldSchema{{FieldID: 100, Name: "text", DataType: schemapb.DataType_VarChar}},
	}, map[int64]struct{}{100: {}}))
}

func TestPrepareAnalyzerExtraInfoNoDownload(t *testing.T) {
	schema := bm25AnalyzerTestSchema()

	// no FileResources on the plan (sync mode) -> "", non-nil noop release, nil cm never touched
	t.Run("no_resources", func(t *testing.T) {
		plan := &datapb.CompactionPlan{Schema: schema}
		extraInfo, release, err := prepareAnalyzerExtraInfo(context.Background(), nil, plan, "", map[int64]struct{}{100: {}})
		require.NoError(t, err)
		require.Empty(t, extraInfo)
		require.NotNil(t, release)
		release()
		release() // idempotent, must not panic
	})

	// resources present but the function output already exists -> lazy-skip, no download
	t.Run("lazy_skip_when_output_present", func(t *testing.T) {
		plan := &datapb.CompactionPlan{
			Schema:        schema,
			FileResources: []*internalpb.FileResourceInfo{{Id: 7, Name: "dict", Path: "dict.jieba"}},
		}
		extraInfo, release, err := prepareAnalyzerExtraInfo(context.Background(), nil, plan, "", map[int64]struct{}{100: {}, 101: {}})
		require.NoError(t, err)
		require.Empty(t, extraInfo)
		require.NotNil(t, release)
		release()
	})
}

func removeFieldBinlogForTest(kvs map[string][]byte, fieldBinlogs map[int64]*datapb.FieldBinlog, fieldID int64) {
	for _, binlog := range fieldBinlogs[fieldID].GetBinlogs() {
		delete(kvs, binlog.GetLogPath())
	}
	delete(fieldBinlogs, fieldID)
}

func downloadValuesForPathsForTest(kvs map[string][]byte, paths []string) ([][]byte, error) {
	values := make([][]byte, 0, len(paths))
	for _, path := range paths {
		value, ok := kvs[path]
		if !ok {
			return nil, errors.Newf("unexpected download path %s", path)
		}
		values = append(values, value)
	}
	return values, nil
}

func TestFieldBinlogEntriesForTestUsesChildFields(t *testing.T) {
	fieldBinlogs := []*datapb.FieldBinlog{
		{FieldID: 0, ChildFields: []int64{101, 107}, Binlogs: []*datapb.Binlog{{EntriesNum: 3}}},
		{FieldID: 108, Binlogs: []*datapb.Binlog{{EntriesNum: 5}}},
	}

	require.EqualValues(t, 3, fieldBinlogEntriesForTest(fieldBinlogs, 107))
	require.EqualValues(t, 5, fieldBinlogEntriesForTest(fieldBinlogs, 108))
	require.EqualValues(t, 0, fieldBinlogEntriesForTest(fieldBinlogs, 109))
}

func fieldBinlogEntriesForTest(fieldBinlogs []*datapb.FieldBinlog, fieldID int64) int64 {
	var entries int64
	for _, fieldBinlog := range fieldBinlogs {
		matchesField := fieldBinlog.GetFieldID() == fieldID
		for _, childFieldID := range fieldBinlog.GetChildFields() {
			if childFieldID == fieldID {
				matchesField = true
				break
			}
		}
		if !matchesField {
			continue
		}
		for _, binlog := range fieldBinlog.GetBinlogs() {
			entries += binlog.GetEntriesNum()
		}
	}
	return entries
}
