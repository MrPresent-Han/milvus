package util

import (
	"fmt"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
)

// FieldDiff represents the differences in fields between two schemas
type FieldDiff struct {
	Added    []*schemapb.FieldSchema // Fields present in new_schema but not in old_schema
	Removed  []*schemapb.FieldSchema // Fields present in old_schema but not in new_schema
	Modified []FieldModification     // Fields that exist in both but have different properties
}

// FuncDiff represents the differences in functions between two schemas
type FuncDiff struct {
	Added    []*schemapb.FunctionSchema // Functions present in new_schema but not in old_schema
	Removed  []*schemapb.FunctionSchema // Functions present in old_schema but not in new_schema
	Modified []FunctionModification     // Functions that exist in both but have different properties
}

// FieldModification represents a field that has been modified
type FieldModification struct {
	OldField *schemapb.FieldSchema
	NewField *schemapb.FieldSchema
	Changes  []string // List of what changed (e.g., "DataType", "IsPrimaryKey", etc.)
}

// FunctionModification represents a function that has been modified
type FunctionModification struct {
	OldFunction *schemapb.FunctionSchema
	NewFunction *schemapb.FunctionSchema
	Changes     []string // List of what changed (e.g., "Type", "InputFieldIds", etc.)
}

// schema_diff compares two schemas and returns the differences in fields and functions
func SchemaDiff(oldSchema, newSchema *schemapb.CollectionSchema) (*FieldDiff, *FuncDiff, error) {
	if oldSchema == nil {
		return nil, nil, fmt.Errorf("old_schema cannot be nil")
	}
	if newSchema == nil {
		return nil, nil, fmt.Errorf("new_schema cannot be nil")
	}

	fieldDiff, err := compareFields(oldSchema.GetFields(), newSchema.GetFields())
	if err != nil {
		return nil, nil, fmt.Errorf("failed to compare fields: %w", err)
	}

	funcDiff, err := compareFunctions(oldSchema.GetFunctions(), newSchema.GetFunctions())
	if err != nil {
		return nil, nil, fmt.Errorf("failed to compare functions: %w", err)
	}

	return fieldDiff, funcDiff, nil
}

// compareFields compares field arrays and returns the differences
func compareFields(oldFields, newFields []*schemapb.FieldSchema) (*FieldDiff, error) {
	diff := &FieldDiff{
		Added:    make([]*schemapb.FieldSchema, 0),
		Removed:  make([]*schemapb.FieldSchema, 0),
		Modified: make([]FieldModification, 0),
	}

	// Create maps for efficient lookup by FieldID
	oldFieldMap := make(map[int64]*schemapb.FieldSchema)
	newFieldMap := make(map[int64]*schemapb.FieldSchema)

	for _, field := range oldFields {
		if field != nil {
			oldFieldMap[field.GetFieldID()] = field
		}
	}

	for _, field := range newFields {
		if field != nil {
			newFieldMap[field.GetFieldID()] = field
		}
	}

	// Find added and modified fields
	for fieldID, newField := range newFieldMap {
		if oldField, exists := oldFieldMap[fieldID]; exists {
			// Field exists in both, check for modifications
			if changes := compareFieldProperties(oldField, newField); len(changes) > 0 {
				diff.Modified = append(diff.Modified, FieldModification{
					OldField: oldField,
					NewField: newField,
					Changes:  changes,
				})
			}
		} else {
			// Field only exists in new schema
			diff.Added = append(diff.Added, newField)
		}
	}

	// Find removed fields
	for fieldID, oldField := range oldFieldMap {
		if _, exists := newFieldMap[fieldID]; !exists {
			diff.Removed = append(diff.Removed, oldField)
		}
	}

	return diff, nil
}

// compareFunctions compares function arrays and returns the differences
func compareFunctions(oldFunctions, newFunctions []*schemapb.FunctionSchema) (*FuncDiff, error) {
	diff := &FuncDiff{
		Added:    make([]*schemapb.FunctionSchema, 0),
		Removed:  make([]*schemapb.FunctionSchema, 0),
		Modified: make([]FunctionModification, 0),
	}

	// Create maps for efficient lookup by function ID
	oldFuncMap := make(map[int64]*schemapb.FunctionSchema)
	newFuncMap := make(map[int64]*schemapb.FunctionSchema)

	for _, function := range oldFunctions {
		if function != nil {
			oldFuncMap[function.GetId()] = function
		}
	}

	for _, function := range newFunctions {
		if function != nil {
			newFuncMap[function.GetId()] = function
		}
	}

	// Find added and modified functions
	for funcID, newFunc := range newFuncMap {
		if oldFunc, exists := oldFuncMap[funcID]; exists {
			// Function exists in both, check for modifications
			if changes := compareFunctionProperties(oldFunc, newFunc); len(changes) > 0 {
				diff.Modified = append(diff.Modified, FunctionModification{
					OldFunction: oldFunc,
					NewFunction: newFunc,
					Changes:     changes,
				})
			}
		} else {
			// Function only exists in new schema
			diff.Added = append(diff.Added, newFunc)
		}
	}

	// Find removed functions
	for funcID, oldFunc := range oldFuncMap {
		if _, exists := newFuncMap[funcID]; !exists {
			diff.Removed = append(diff.Removed, oldFunc)
		}
	}

	return diff, nil
}

// compareFieldProperties compares individual field properties and returns list of changes
func compareFieldProperties(oldField, newField *schemapb.FieldSchema) []string {
	var changes []string

	if oldField.GetName() != newField.GetName() {
		changes = append(changes, "Name")
	}
	if oldField.GetIsPrimaryKey() != newField.GetIsPrimaryKey() {
		changes = append(changes, "IsPrimaryKey")
	}
	if oldField.GetDescription() != newField.GetDescription() {
		changes = append(changes, "Description")
	}
	if oldField.GetDataType() != newField.GetDataType() {
		changes = append(changes, "DataType")
	}
	if !compareKeyValuePairs(oldField.GetTypeParams(), newField.GetTypeParams()) {
		changes = append(changes, "TypeParams")
	}
	if !compareKeyValuePairs(oldField.GetIndexParams(), newField.GetIndexParams()) {
		changes = append(changes, "IndexParams")
	}
	if oldField.GetAutoID() != newField.GetAutoID() {
		changes = append(changes, "AutoID")
	}
	if oldField.GetState() != newField.GetState() {
		changes = append(changes, "State")
	}
	if oldField.GetElementType() != newField.GetElementType() {
		changes = append(changes, "ElementType")
	}
	if !compareValueFields(oldField.GetDefaultValue(), newField.GetDefaultValue()) {
		changes = append(changes, "DefaultValue")
	}
	if oldField.GetIsDynamic() != newField.GetIsDynamic() {
		changes = append(changes, "IsDynamic")
	}
	if oldField.GetIsPartitionKey() != newField.GetIsPartitionKey() {
		changes = append(changes, "IsPartitionKey")
	}
	if oldField.GetIsClusteringKey() != newField.GetIsClusteringKey() {
		changes = append(changes, "IsClusteringKey")
	}
	if oldField.GetNullable() != newField.GetNullable() {
		changes = append(changes, "Nullable")
	}
	if oldField.GetIsFunctionOutput() != newField.GetIsFunctionOutput() {
		changes = append(changes, "IsFunctionOutput")
	}

	return changes
}

// compareFunctionProperties compares individual function properties and returns list of changes
func compareFunctionProperties(oldFunc, newFunc *schemapb.FunctionSchema) []string {
	var changes []string

	if oldFunc.GetName() != newFunc.GetName() {
		changes = append(changes, "Name")
	}
	if oldFunc.GetDescription() != newFunc.GetDescription() {
		changes = append(changes, "Description")
	}
	if oldFunc.GetType() != newFunc.GetType() {
		changes = append(changes, "Type")
	}
	if !compareStringSlices(oldFunc.GetInputFieldNames(), newFunc.GetInputFieldNames()) {
		changes = append(changes, "InputFieldNames")
	}
	if !compareInt64Slices(oldFunc.GetInputFieldIds(), newFunc.GetInputFieldIds()) {
		changes = append(changes, "InputFieldIds")
	}
	if !compareStringSlices(oldFunc.GetOutputFieldNames(), newFunc.GetOutputFieldNames()) {
		changes = append(changes, "OutputFieldNames")
	}
	if !compareInt64Slices(oldFunc.GetOutputFieldIds(), newFunc.GetOutputFieldIds()) {
		changes = append(changes, "OutputFieldIds")
	}
	if !compareKeyValuePairs(oldFunc.GetParams(), newFunc.GetParams()) {
		changes = append(changes, "Params")
	}

	return changes
}

// Helper functions for comparing different types

func compareKeyValuePairs(old, new []*commonpb.KeyValuePair) bool {
	if len(old) != len(new) {
		return false
	}

	oldMap := make(map[string]string)
	newMap := make(map[string]string)

	for _, kv := range old {
		if kv != nil {
			oldMap[kv.GetKey()] = kv.GetValue()
		}
	}

	for _, kv := range new {
		if kv != nil {
			newMap[kv.GetKey()] = kv.GetValue()
		}
	}

	if len(oldMap) != len(newMap) {
		return false
	}

	for key, oldValue := range oldMap {
		if newValue, exists := newMap[key]; !exists || oldValue != newValue {
			return false
		}
	}

	return true
}

func compareValueFields(old, new *schemapb.ValueField) bool {
	if old == nil && new == nil {
		return true
	}
	if old == nil || new == nil {
		return false
	}

	// Compare the actual values - this is a simplified comparison
	// In a real implementation, you might want to compare the specific value types
	return old.String() == new.String()
}

func compareStringSlices(old, new []string) bool {
	if len(old) != len(new) {
		return false
	}

	for i, v := range old {
		if v != new[i] {
			return false
		}
	}

	return true
}

func compareInt64Slices(old, new []int64) bool {
	if len(old) != len(new) {
		return false
	}

	for i, v := range old {
		if v != new[i] {
			return false
		}
	}

	return true
}
