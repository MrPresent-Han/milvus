package util

import (
	"testing"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/stretchr/testify/assert"
)

func TestSchemaDiff_BasicFieldChanges(t *testing.T) {
	// Create old schema with basic fields
	oldSchema := &schemapb.CollectionSchema{
		Name: "test_collection",
		Fields: []*schemapb.FieldSchema{
			{
				FieldID:      1,
				Name:         "id",
				DataType:     schemapb.DataType_Int64,
				IsPrimaryKey: true,
			},
			{
				FieldID:  2,
				Name:     "name",
				DataType: schemapb.DataType_VarChar,
			},
			{
				FieldID:  3,
				Name:     "vector",
				DataType: schemapb.DataType_FloatVector,
			},
		},
		Functions: []*schemapb.FunctionSchema{
			{
				Id:   1,
				Name: "bm25_function",
				Type: schemapb.FunctionType_BM25,
			},
		},
	}

	// Create new schema with some changes
	newSchema := &schemapb.CollectionSchema{
		Name: "test_collection",
		Fields: []*schemapb.FieldSchema{
			{
				FieldID:      1,
				Name:         "id",
				DataType:     schemapb.DataType_Int64,
				IsPrimaryKey: true,
			},
			{
				FieldID:  2,
				Name:     "name",
				DataType: schemapb.DataType_VarChar,
				Nullable: true, // Modified: added nullable
			},
			{
				FieldID:  4, // Added: new field
				Name:     "description",
				DataType: schemapb.DataType_VarChar,
			},
			// Removed: vector field (FieldID 3)
		},
		Functions: []*schemapb.FunctionSchema{
			{
				Id:              1,
				Name:            "bm25_function",
				Type:            schemapb.FunctionType_BM25,
				InputFieldNames: []string{"name"}, // Modified: added input field names
			},
			{
				Id:   2, // Added: new function
				Name: "embedding_function",
				Type: schemapb.FunctionType_Unknown,
			},
		},
	}

	fieldDiff, funcDiff, err := SchemaDiff(oldSchema, newSchema)
	assert.NoError(t, err)
	assert.NotNil(t, fieldDiff)
	assert.NotNil(t, funcDiff)

	// Check field differences
	assert.Len(t, fieldDiff.Added, 1)
	assert.Equal(t, int64(4), fieldDiff.Added[0].GetFieldID())
	assert.Equal(t, "description", fieldDiff.Added[0].GetName())

	assert.Len(t, fieldDiff.Removed, 1)
	assert.Equal(t, int64(3), fieldDiff.Removed[0].GetFieldID())
	assert.Equal(t, "vector", fieldDiff.Removed[0].GetName())

	assert.Len(t, fieldDiff.Modified, 1)
	assert.Equal(t, int64(2), fieldDiff.Modified[0].OldField.GetFieldID())
	assert.Contains(t, fieldDiff.Modified[0].Changes, "Nullable")

	// Check function differences
	assert.Len(t, funcDiff.Added, 1)
	assert.Equal(t, int64(2), funcDiff.Added[0].GetId())
	assert.Equal(t, "embedding_function", funcDiff.Added[0].GetName())

	assert.Len(t, funcDiff.Removed, 0) // No functions removed

	assert.Len(t, funcDiff.Modified, 1)
	assert.Equal(t, int64(1), funcDiff.Modified[0].OldFunction.GetId())
	assert.Contains(t, funcDiff.Modified[0].Changes, "InputFieldNames")
}

func TestSchemaDiff_NilSchemas(t *testing.T) {
	schema := &schemapb.CollectionSchema{
		Name: "test",
		Fields: []*schemapb.FieldSchema{
			{FieldID: 1, Name: "id", DataType: schemapb.DataType_Int64},
		},
	}

	// Test nil old schema
	_, _, err := SchemaDiff(nil, schema)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "old_schema cannot be nil")

	// Test nil new schema
	_, _, err = SchemaDiff(schema, nil)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "new_schema cannot be nil")
}

func TestSchemaDiff_IdenticalSchemas(t *testing.T) {
	schema := &schemapb.CollectionSchema{
		Name: "test_collection",
		Fields: []*schemapb.FieldSchema{
			{
				FieldID:      1,
				Name:         "id",
				DataType:     schemapb.DataType_Int64,
				IsPrimaryKey: true,
			},
		},
		Functions: []*schemapb.FunctionSchema{
			{
				Id:   1,
				Name: "test_function",
				Type: schemapb.FunctionType_BM25,
			},
		},
	}

	fieldDiff, funcDiff, err := SchemaDiff(schema, schema)
	assert.NoError(t, err)
	assert.NotNil(t, fieldDiff)
	assert.NotNil(t, funcDiff)

	// No differences should be found
	assert.Len(t, fieldDiff.Added, 0)
	assert.Len(t, fieldDiff.Removed, 0)
	assert.Len(t, fieldDiff.Modified, 0)

	assert.Len(t, funcDiff.Added, 0)
	assert.Len(t, funcDiff.Removed, 0)
	assert.Len(t, funcDiff.Modified, 0)
}

func TestSchemaDiff_ComplexFieldModifications(t *testing.T) {
	oldSchema := &schemapb.CollectionSchema{
		Fields: []*schemapb.FieldSchema{
			{
				FieldID:     1,
				Name:        "test_field",
				DataType:    schemapb.DataType_VarChar,
				Description: "old description",
				TypeParams: []*commonpb.KeyValuePair{
					{Key: "max_length", Value: "100"},
				},
				IndexParams: []*commonpb.KeyValuePair{
					{Key: "index_type", Value: "TRIE"},
				},
				AutoID:           false,
				State:            schemapb.FieldState_FieldCreated,
				IsDynamic:        false,
				IsPartitionKey:   false,
				IsClusteringKey:  false,
				Nullable:         false,
				IsFunctionOutput: false,
			},
		},
	}

	newSchema := &schemapb.CollectionSchema{
		Fields: []*schemapb.FieldSchema{
			{
				FieldID:     1,
				Name:        "test_field_renamed",
				DataType:    schemapb.DataType_Text, // Changed
				Description: "new description",      // Changed
				TypeParams: []*commonpb.KeyValuePair{
					{Key: "max_length", Value: "200"}, // Changed value
				},
				IndexParams: []*commonpb.KeyValuePair{
					{Key: "index_type", Value: "INVERTED"}, // Changed value
					{Key: "analyzer", Value: "standard"},   // Added new param
				},
				AutoID:           true, // Changed
				State:            schemapb.FieldState_FieldCreated,
				IsDynamic:        true, // Changed
				IsPartitionKey:   true, // Changed
				IsClusteringKey:  true, // Changed
				Nullable:         true, // Changed
				IsFunctionOutput: true, // Changed
			},
		},
	}

	fieldDiff, _, err := SchemaDiff(oldSchema, newSchema)
	assert.NoError(t, err)
	assert.NotNil(t, fieldDiff)

	assert.Len(t, fieldDiff.Added, 0)
	assert.Len(t, fieldDiff.Removed, 0)
	assert.Len(t, fieldDiff.Modified, 1)

	modification := fieldDiff.Modified[0]
	expectedChanges := []string{
		"Name", "DataType", "Description", "TypeParams", "IndexParams",
		"AutoID", "IsDynamic", "IsPartitionKey", "IsClusteringKey",
		"Nullable", "IsFunctionOutput",
	}

	for _, expectedChange := range expectedChanges {
		assert.Contains(t, modification.Changes, expectedChange)
	}
}
