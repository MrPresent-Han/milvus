package segments

import (
	"context"
	"encoding/binary"
	"fmt"
	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/internal/proto/planpb"
	"github.com/milvus-io/milvus/internal/proto/segcorepb"
	typeutil2 "github.com/milvus-io/milvus/internal/util/typeutil"
	"hash"
	"hash/fnv"
	"math"
)

type Entry struct {
	val interface{}
}

func (r *Row) Equal(other *Row) bool {
	// Check if the number of entries is the same
	if len(r.entries) != len(other.entries) {
		return false
	}
	// Compare each entry for equality
	for i := 0; i < r.keyCount; i++ {
		if r.entries[i].val != other.entries[i].val {
			return false
		}
	}
	return true
}

func newEntry(v interface{}) *Entry {
	return &Entry{val: v}
}

type Row struct {
	entries  []*Entry
	keyCount int
}

func newRow(entries []*Entry, kCount int) *Row {
	return &Row{entries: entries, keyCount: kCount}
}

type Bucket struct {
	rows []*Row
}

func (bucket *Bucket) addRow(row *Row) {
	bucket.rows = append(bucket.rows, row)
}

func (bucket *Bucket) accumulate(row *Row, idx int) error {
	if idx >= len(bucket.rows) || idx < 0 {
		return fmt.Errorf("wrong idx:%d for bucket", idx)
	}
	bucket.rows[idx].Equal()
}

const NONE int = -1

func (bucket *Bucket) find(row *Row) int {
	for idx, existingRow := range bucket.rows {
		if existingRow.Equal(row) {
			return idx
		}
	}
	return NONE
}

func NewBucket() *Bucket {
	return &Bucket{rows: make([]*Row, 0, 1)}
}

type AggReducer struct {
	groupByFieldIds []int64
	aggregate       []*planpb.Aggregate
	schema          *schemapb.CollectionSchema
	hashValsMap     map[uint64]*Bucket
	hashes          []uint64
}

func CalculateRowNumber(results []*segcorepb.RetrieveResults) int64 {
	rowNumber := 0
	for _, result := range results {
		fieldsData := result.GetFieldsData()
		if len(fieldsData) < 1 {
			return 0
		}
		firstFieldData := fieldsData[0]
		switch firstFieldData.GetType() {
		case schemapb.DataType_Bool:
			rowNumber += len(firstFieldData.GetScalars().GetBoolData().Data)
		case schemapb.DataType_Int8:
		case schemapb.DataType_Int16:
		case schemapb.DataType_Int32:
			rowNumber += len(firstFieldData.GetScalars().GetIntData().Data)
		case schemapb.DataType_Int64:
			rowNumber += len(firstFieldData.GetScalars().GetLongData().Data)
		case schemapb.DataType_VarChar:
		case schemapb.DataType_String:
			rowNumber += len(firstFieldData.GetScalars().GetStringData().Data)
		case schemapb.DataType_Float:
			rowNumber += len(firstFieldData.GetScalars().GetFloatData().Data)
		case schemapb.DataType_Double:
			rowNumber += len(firstFieldData.GetScalars().GetDoubleData().Data)
		}
	}
	return int64(rowNumber)
}

func (reducer *AggReducer) Reduce(ctx context.Context, results []*segcorepb.RetrieveResults, segments []Segment, plan *RetrievePlan) (*segcorepb.RetrieveResults, error) {
	if results == nil || len(results) == 0 {
		return nil, fmt.Errorf("no input segment's retrieved results can be reduced")
	}

	// 1. compute hash values for all rows in the result retrieved
	rowNumber := CalculateRowNumber(results)
	reducer.hashes = make([]uint64, rowNumber)
	numGroupingKeys := len(reducer.groupByFieldIds)
	hashers := make([]FieldAccessor, 0, numGroupingKeys)
	accumulators := make([]FieldAccessor, 0)
	firstFieldData := results[0].GetFieldsData()
	for idx, fieldData := range firstFieldData {
		if idx < numGroupingKeys {
			hasher, err := newFieldAccessor(fieldData.GetType())
			if err != nil {
				return nil, err
			}
			hashers = append(hashers, hasher)
		}
		if idx >= numGroupingKeys {
			accumulator, err := newFieldAccessor(fieldData.GetType())
			if err != nil {
				return nil, err
			}
			accumulators = append(accumulators, accumulator)
		}
	}

	outputColumnCount := -1
	rowIdx := 0
	for _, result := range results {
		fieldDatas := result.GetFieldsData()
		if outputColumnCount == -1 {
			outputColumnCount = len(fieldDatas)
		} else if outputColumnCount != len(fieldDatas) {
			return nil, fmt.Errorf("retrieved results from different segments have different size of columns")
		}
		if outputColumnCount == 0 {
			return nil, fmt.Errorf("retrieved results have no column data")
		}
		rowCount := -1
		for i := 0; i < outputColumnCount; i++ {
			fieldData := fieldDatas[i]
			if i < numGroupingKeys {
				hashers[i].setVals(fieldData)
			} else {
				accumulators[i-numGroupingKeys].setVals(fieldData)
			}
			if rowCount == -1 {
				rowCount = hashers[i].rowCount()
			} else if rowCount != hashers[i].rowCount() {
				return nil, fmt.Errorf("field data:%d for different columns have different row count, %d vs %d, wrong state",
					i, rowCount, hashers[i].rowCount())
			}
		}
		for row := 0; row < rowCount; row++ {
			rowEntries := make([]*Entry, outputColumnCount)
			for col := 0; col < outputColumnCount; col++ {
				if col < numGroupingKeys {
					if col > 0 {
						reducer.hashes[rowIdx] = typeutil2.HashMix(reducer.hashes[rowIdx], hashers[col].hash(row))
					} else {
						reducer.hashes[rowIdx] = hashers[col].hash(row)
					}
					rowEntries[col] = newEntry(hashers[col].val(row))
				} else {
					rowEntries[col] = newEntry(accumulators[col-numGroupingKeys].val(row))
				}
			}
			newRow := newRow(rowEntries, numGroupingKeys)
			if bucket := reducer.hashValsMap[reducer.hashes[rowIdx]]; bucket == nil {
				newBucket := NewBucket()
				newBucket.addRow(newRow)
				reducer.hashValsMap[reducer.hashes[rowIdx]] = newBucket
			} else {
				if rowIdx := bucket.find(newRow); rowIdx == NONE {
					bucket.addRow(newRow)
				} else {

				}
			}
			rowIdx++
		}
	}

	return nil, nil
}

func newFieldAccessor(fieldType schemapb.DataType) (FieldAccessor, error) {
	switch fieldType {
	case schemapb.DataType_Bool:
		return newBoolFieldAccessor(), nil
	case schemapb.DataType_Int8:
	case schemapb.DataType_Int16:
	case schemapb.DataType_Int32:
		return newInt32FieldAccessor(), nil
	case schemapb.DataType_Int64:
		return newInt64FieldAccessor(), nil
	case schemapb.DataType_VarChar:
	case schemapb.DataType_String:
		return newStringFieldAccessor(), nil
	case schemapb.DataType_Float:
		return newFloat32FieldAccessor(), nil
	case schemapb.DataType_Double:
		return newFloat64FieldAccessor(), nil
	default:
		return nil, fmt.Errorf("unsupported data type for hasher")
	}
	return nil, nil
}

type FieldAccessor interface {
	hash(idx int) uint64
	val(idx int) interface{}
	setVals(fieldData *schemapb.FieldData)
	rowCount() int
}

type Int32FieldAccessor struct {
	vals   []int32
	hasher hash.Hash64
	buffer []byte
}

func (i32Field *Int32FieldAccessor) hash(idx int) uint64 {
	i32Field.hasher.Reset()
	val := i32Field.vals[idx]
	binary.LittleEndian.PutUint64(i32Field.buffer, uint64(val))
	i32Field.hasher.Write(i32Field.buffer)
	return i32Field.hasher.Sum64()
}

func (i32Field *Int32FieldAccessor) setVals(fieldData *schemapb.FieldData) {
	i32Field.vals = fieldData.GetScalars().GetIntData().GetData()
}

func (i32Field *Int32FieldAccessor) rowCount() int {
	return len(i32Field.vals)
}

func (i32Field *Int32FieldAccessor) val(idx int) interface{} {
	return i32Field.vals[idx]
}

func newInt32FieldAccessor() FieldAccessor {
	return &Int32FieldAccessor{hasher: fnv.New64a(), buffer: make([]byte, 4)}
}

type Int64FieldAccessor struct {
	vals   []int64
	hasher hash.Hash64
	buffer []byte
}

func (i64Field *Int64FieldAccessor) hash(idx int) uint64 {
	i64Field.hasher.Reset()
	val := i64Field.vals[idx]
	binary.LittleEndian.PutUint64(i64Field.buffer, uint64(val))
	i64Field.hasher.Write(i64Field.buffer)
	return i64Field.hasher.Sum64()
}

func (i64Field *Int64FieldAccessor) setVals(fieldData *schemapb.FieldData) {
	i64Field.vals = fieldData.GetScalars().GetLongData().GetData()
}

func (i64Field *Int64FieldAccessor) rowCount() int {
	return len(i64Field.vals)
}

func (i64Field *Int64FieldAccessor) val(idx int) interface{} {
	return i64Field.vals[idx]
}

func newInt64FieldAccessor() FieldAccessor {
	return &Int64FieldAccessor{hasher: fnv.New64a(), buffer: make([]byte, 8)}
}

// BoolFieldAccessor
type BoolFieldAccessor struct {
	vals   []bool
	hasher hash.Hash64
	buffer []byte
}

func (boolField *BoolFieldAccessor) hash(idx int) uint64 {
	boolField.hasher.Reset()
	val := boolField.vals[idx]
	if val {
		boolField.buffer[0] = 1
	} else {
		boolField.buffer[0] = 0
	}
	boolField.hasher.Write(boolField.buffer[:1])
	return boolField.hasher.Sum64()
}

func (boolField *BoolFieldAccessor) setVals(fieldData *schemapb.FieldData) {
	boolField.vals = fieldData.GetScalars().GetBoolData().GetData()
}

func (boolField *BoolFieldAccessor) rowCount() int {
	return len(boolField.vals)
}

func (boolField *BoolFieldAccessor) val(idx int) interface{} {
	return boolField.vals[idx]
}

func newBoolFieldAccessor() FieldAccessor {
	return &BoolFieldAccessor{hasher: fnv.New64a(), buffer: make([]byte, 1)}
}

// Float32FieldAccessor
type Float32FieldAccessor struct {
	vals   []float32
	hasher hash.Hash64
	buffer []byte
}

func (f32FieldAccessor *Float32FieldAccessor) hash(idx int) uint64 {
	f32FieldAccessor.hasher.Reset()
	val := f32FieldAccessor.vals[idx]
	binary.LittleEndian.PutUint32(f32FieldAccessor.buffer, math.Float32bits(val))
	f32FieldAccessor.hasher.Write(f32FieldAccessor.buffer[:4])
	return f32FieldAccessor.hasher.Sum64()
}

func (f32FieldAccessor *Float32FieldAccessor) setVals(fieldData *schemapb.FieldData) {
	f32FieldAccessor.vals = fieldData.GetScalars().GetFloatData().GetData()
}

func (f32FieldAccessor *Float32FieldAccessor) rowCount() int {
	return len(f32FieldAccessor.vals)
}

func (f32FieldAccessor *Float32FieldAccessor) val(idx int) interface{} {
	return f32FieldAccessor.vals[idx]
}

func newFloat32FieldAccessor() FieldAccessor {
	return &Float32FieldAccessor{hasher: fnv.New64a(), buffer: make([]byte, 4)}
}

// Float64FieldAccessor
type Float64FieldAccessor struct {
	vals   []float64
	hasher hash.Hash64
	buffer []byte
}

func (f64Field *Float64FieldAccessor) hash(idx int) uint64 {
	f64Field.hasher.Reset()
	val := f64Field.vals[idx]
	binary.LittleEndian.PutUint64(f64Field.buffer, math.Float64bits(val))
	f64Field.hasher.Write(f64Field.buffer)
	return f64Field.hasher.Sum64()
}

func (f64Field *Float64FieldAccessor) setVals(fieldData *schemapb.FieldData) {
	f64Field.vals = fieldData.GetScalars().GetDoubleData().GetData()
}

func (f64Field *Float64FieldAccessor) rowCount() int {
	return len(f64Field.vals)
}

func (f64Field *Float64FieldAccessor) val(idx int) interface{} {
	return f64Field.vals[idx]
}

func newFloat64FieldAccessor() FieldAccessor {
	return &Float64FieldAccessor{hasher: fnv.New64a(), buffer: make([]byte, 8)}
}

// StringFieldAccessor
type StringFieldAccessor struct {
	vals   []string
	hasher hash.Hash64
	buffer []byte
}

func (stringField *StringFieldAccessor) hash(idx int) uint64 {
	stringField.hasher.Reset()
	val := stringField.vals[idx]
	if len(val) > len(stringField.buffer) {
		newSize := typeutil2.NextPowerOfTwo(len(val))
		stringField.buffer = make([]byte, newSize)
	}
	copy(stringField.buffer, val)
	stringField.hasher.Write(stringField.buffer[0:len(val)])
	return stringField.hasher.Sum64()
}

func (stringField *StringFieldAccessor) setVals(fieldData *schemapb.FieldData) {
	stringField.vals = fieldData.GetScalars().GetStringData().GetData()
}

func (stringField *StringFieldAccessor) rowCount() int {
	return len(stringField.vals)
}
func (stringField *StringFieldAccessor) val(idx int) interface{} {
	return stringField.vals[idx]
}

func newStringFieldAccessor() FieldAccessor {
	return &StringFieldAccessor{hasher: fnv.New64a(), buffer: make([]byte, 1024)}
}

func (reducer *AggReducer) calculateHash(fieldDatas []*schemapb.FieldData) {
	columnCount := len(fieldDatas)
	rowType := fieldDatas[0].GetType()
	var rowCount int = 0
	switch rowType {
	case schemapb.DataType_Int8:
	case schemapb.DataType_Int16:
	case schemapb.DataType_Int32:
	case schemapb.DataType_Int64:
		intArray := fieldDatas[0].GetScalars().GetIntData().GetData()
		strArray := fieldDatas[0].GetScalars().GetStringData().GetData()
		floatArray := fieldDatas[0].GetScalars().GetIntData().GetData()
		//floatArray := fieldDatas[0].GetScalars().GetFlo
		rowCount = len(intArray)
		rowCount = len(strArray)
		rowCount = len(floatArray)
	}

	for i := 0; i < rowCount; i++ {

	}

}
