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

func newEntry(v interface{}) *Entry {
	return &Entry{val: v}
}

type Row struct {
	entries []*Entry
}

type Bucket struct {
	rows []*Row
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
	/*fieldIdMap := make(map[int64]*schemapb.FieldSchema)
	for _, fieldMeta := range reducer.schema.GetFields() {
		fieldIdMap[fieldMeta.GetFieldID()] = fieldMeta
	}
	types := make([]schemapb.DataType, 0, len(reducer.groupByFieldIds))
	for _, groupByField := range reducer.groupByFieldIds {
		if fieldIdMap[groupByField] == nil {
			return nil, fmt.Errorf("group by field must exist in the schema in the reduce stage")
		}
		types = append(types, fieldIdMap[groupByField].GetDataType())
	}*/
	//create hashers based on first result
	rowNumber := CalculateRowNumber(results)
	reducer.hashes = make([]uint64, rowNumber)
	numGroupingKeys := len(reducer.groupByFieldIds)
	hashers := make([]Hasher, 0, numGroupingKeys)
	firstFieldData := results[0].GetFieldsData()
	for idx, fieldData := range firstFieldData {
		if idx < numGroupingKeys {
			hasher, err := newHasher(fieldData.GetType())
			if err != nil {
				return nil, err
			}
			hashers = append(hashers, hasher)
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
		for i := 0; i < numGroupingKeys; i++ {
			fieldData := fieldDatas[i]
			hashers[i].setVals(fieldData)
			if rowCount == -1 {
				rowCount = hashers[i].rowCount()
			} else if rowCount != hashers[i].rowCount() {
				return nil, fmt.Errorf("field data:%d for different columns have different row count, %d vs %d, wrong state",
					i, rowCount, hashers[i].rowCount())
			}
		}
		for i := 0; i < rowCount; i++ {
			for j := 0; j < numGroupingKeys; j++ {
				if i > 0 {
					reducer.hashes[rowIdx] = typeutil2.HashMix(reducer.hashes[rowIdx], hashers[j].hash(i))
				} else {
					reducer.hashes[rowIdx] = hashers[j].hash(i)
				}
			}
			rowIdx++
		}
	}

	return nil, nil
}

func newHasher(fieldType schemapb.DataType) (Hasher, error) {
	switch fieldType {
	case schemapb.DataType_Bool:
		return newBoolHasher(), nil
	case schemapb.DataType_Int8:
	case schemapb.DataType_Int16:
	case schemapb.DataType_Int32:
		return newInt32Hasher(), nil
	case schemapb.DataType_Int64:
		return newInt64Hasher(), nil
	case schemapb.DataType_VarChar:
	case schemapb.DataType_String:
		return newStringHasher(), nil
	case schemapb.DataType_Float:
		return newFloat32Hasher(), nil
	case schemapb.DataType_Double:
		return newFloat64Hasher(), nil
	default:
		return nil, fmt.Errorf("unsupported data type for hasher")
	}
	return nil, nil
}

type Hasher interface {
	hash(idx int) uint64
	setVals(fieldData *schemapb.FieldData)
	rowCount() int
}

type Int32Hasher struct {
	vals   []int32
	hasher hash.Hash64
	buffer []byte
}

func (i32Hasher *Int32Hasher) hash(idx int) uint64 {
	i32Hasher.hasher.Reset()
	val := i32Hasher.vals[idx]
	binary.LittleEndian.PutUint64(i32Hasher.buffer, uint64(val))
	i32Hasher.hasher.Write(i32Hasher.buffer)
	return i32Hasher.hasher.Sum64()
}

func (i32Hasher *Int32Hasher) setVals(fieldData *schemapb.FieldData) {
	i32Hasher.vals = fieldData.GetScalars().GetIntData().GetData()
}

func (i32Hasher *Int32Hasher) rowCount() int {
	return len(i32Hasher.vals)
}

func newInt32Hasher() Hasher {
	return &Int32Hasher{hasher: fnv.New64a(), buffer: make([]byte, 4)}
}

type Int64Hasher struct {
	vals   []int64
	hasher hash.Hash64
	buffer []byte
}

func (i64Hasher *Int64Hasher) hash(idx int) uint64 {
	i64Hasher.hasher.Reset()
	val := i64Hasher.vals[idx]
	binary.LittleEndian.PutUint64(i64Hasher.buffer, uint64(val))
	i64Hasher.hasher.Write(i64Hasher.buffer)
	return i64Hasher.hasher.Sum64()
}

func (i64Hasher *Int64Hasher) setVals(fieldData *schemapb.FieldData) {
	i64Hasher.vals = fieldData.GetScalars().GetLongData().GetData()
}

func (i64Hasher *Int64Hasher) rowCount() int {
	return len(i64Hasher.vals)
}

func newInt64Hasher() Hasher {
	return &Int64Hasher{hasher: fnv.New64a(), buffer: make([]byte, 8)}
}

// BoolHasher
type BoolHasher struct {
	vals   []bool
	hasher hash.Hash64
	buffer []byte
}

func (boolHasher *BoolHasher) hash(idx int) uint64 {
	boolHasher.hasher.Reset()
	val := boolHasher.vals[idx]
	if val {
		boolHasher.buffer[0] = 1
	} else {
		boolHasher.buffer[0] = 0
	}
	boolHasher.hasher.Write(boolHasher.buffer[:1])
	return boolHasher.hasher.Sum64()
}

func (boolHasher *BoolHasher) setVals(fieldData *schemapb.FieldData) {
	boolHasher.vals = fieldData.GetScalars().GetBoolData().GetData()
}

func (boolHasher *BoolHasher) rowCount() int {
	return len(boolHasher.vals)
}

func newBoolHasher() Hasher {
	return &BoolHasher{hasher: fnv.New64a(), buffer: make([]byte, 1)}
}

// Float32Hasher
type Float32Hasher struct {
	vals   []float32
	hasher hash.Hash64
	buffer []byte
}

func (f32Hasher *Float32Hasher) hash(idx int) uint64 {
	f32Hasher.hasher.Reset()
	val := f32Hasher.vals[idx]
	binary.LittleEndian.PutUint32(f32Hasher.buffer, math.Float32bits(val))
	f32Hasher.hasher.Write(f32Hasher.buffer[:4])
	return f32Hasher.hasher.Sum64()
}

func (f32Hasher *Float32Hasher) setVals(fieldData *schemapb.FieldData) {
	f32Hasher.vals = fieldData.GetScalars().GetFloatData().GetData()
}

func (f32Hasher *Float32Hasher) rowCount() int {
	return len(f32Hasher.vals)
}

func newFloat32Hasher() Hasher {
	return &Float32Hasher{hasher: fnv.New64a(), buffer: make([]byte, 4)}
}

// Float64Hasher
type Float64Hasher struct {
	vals   []float64
	hasher hash.Hash64
	buffer []byte
}

func (f64Hasher *Float64Hasher) hash(idx int) uint64 {
	f64Hasher.hasher.Reset()
	val := f64Hasher.vals[idx]
	binary.LittleEndian.PutUint64(f64Hasher.buffer, math.Float64bits(val))
	f64Hasher.hasher.Write(f64Hasher.buffer)
	return f64Hasher.hasher.Sum64()
}

func (f64Hasher *Float64Hasher) setVals(fieldData *schemapb.FieldData) {
	f64Hasher.vals = fieldData.GetScalars().GetDoubleData().GetData()
}

func (f64Hasher *Float64Hasher) rowCount() int {
	return len(f64Hasher.vals)
}

func newFloat64Hasher() Hasher {
	return &Float64Hasher{hasher: fnv.New64a(), buffer: make([]byte, 8)}
}

// StringHasher
type StringHasher struct {
	vals   []string
	hasher hash.Hash64
	buffer []byte
}

func (stringHasher *StringHasher) hash(idx int) uint64 {
	stringHasher.hasher.Reset()
	val := stringHasher.vals[idx]
	if len(val) > len(stringHasher.buffer) {
		newSize := typeutil2.NextPowerOfTwo(len(val))
		stringHasher.buffer = make([]byte, newSize)
	}
	copy(stringHasher.buffer, val)
	stringHasher.hasher.Write(stringHasher.buffer[0:len(val)])
	return stringHasher.hasher.Sum64()
}

func (stringHasher *StringHasher) setVals(fieldData *schemapb.FieldData) {
	stringHasher.vals = fieldData.GetScalars().GetStringData().GetData()
}

func (stringHasher *StringHasher) rowCount() int {
	return len(stringHasher.vals)
}

func newStringHasher() Hasher {
	return &StringHasher{hasher: fnv.New64a(), buffer: make([]byte, 1024)}
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
