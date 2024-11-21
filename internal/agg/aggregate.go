package agg

import (
	"encoding/binary"
	"fmt"
	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/internal/proto/planpb"
	typeutil2 "github.com/milvus-io/milvus/internal/util/typeutil"
	"hash"
	"hash/fnv"
	"math"
	"regexp"
	"strings"
)

const (
	kSum   = "sum"
	kCount = "count"
	kAvg   = "avg"
	kMin   = "min"
	kMax   = "max"
)

var (
	// Define the regular expression pattern once to avoid repeated concatenation.
	aggregationTypes   = kSum + `|` + kCount + `|` + kAvg + `|` + kMin + `|` + kMax
	aggregationPattern = regexp.MustCompile(`(?i)^(` + aggregationTypes + `)\s*\(\s*([\w\*]*)\s*\)$`)
)

// MatchAggregationExpression return isAgg, operator name, operator parameter
func MatchAggregationExpression(expression string) (bool, string, string) {
	// FindStringSubmatch returns the full match and submatches.
	matches := aggregationPattern.FindStringSubmatch(expression)
	if len(matches) > 0 {
		// Return true, the operator, and the captured parameter.
		return true, strings.ToLower(matches[1]), strings.TrimSpace(matches[2])
	}
	return false, "", ""
}

type AggregateBase interface {
	Name() string
	Update()
	ToPB() *planpb.Aggregate
	FieldID() int64
}

func NewAggregate(aggregateName string, aggFieldID int64) (AggregateBase, error) {
	switch aggregateName {
	case kCount:
		return &CountAggregate{fieldID: aggFieldID}, nil
	case kSum:
		return &SumAggregate{fieldID: aggFieldID}, nil
	case kMin:
		return &MinAggregate{fieldID: aggFieldID}, nil
	case kMax:
		return &MaxAggregate{fieldID: aggFieldID}, nil
	default:
		return nil, fmt.Errorf("invalid Aggregation operator %s", aggregateName)
	}
	return nil, nil
}

func FromPB(pb *planpb.Aggregate) (AggregateBase, error) {
	switch pb.Op {
	case planpb.AggregateOp_count:
		return &CountAggregate{fieldID: pb.GetFieldId()}, nil
	case planpb.AggregateOp_sum:
		return &SumAggregate{fieldID: pb.GetFieldId()}, nil
	case planpb.AggregateOp_min:
		return &MinAggregate{fieldID: pb.GetFieldId()}, nil
	case planpb.AggregateOp_max:
		return &MaxAggregate{fieldID: pb.GetFieldId()}, nil
	default:
		return nil, fmt.Errorf("invalid Aggregation operator %d", pb.Op)
	}
	return nil, nil
}

type SumAggregate struct {
	fieldID int64
}

func (sum *SumAggregate) Name() string {
	return kSum
}

func (sum *SumAggregate) Update() {
}

func (sum *SumAggregate) ToPB() *planpb.Aggregate {
	return &planpb.Aggregate{Op: planpb.AggregateOp_sum, FieldId: sum.FieldID()}
}

func (sum *SumAggregate) FieldID() int64 {
	return sum.fieldID
}

type CountAggregate struct {
	fieldID int64
}

func (count *CountAggregate) Name() string {
	return kCount
}
func (count *CountAggregate) Update() {
}

func (count *CountAggregate) ToPB() *planpb.Aggregate {
	return &planpb.Aggregate{Op: planpb.AggregateOp_count, FieldId: count.FieldID()}
}

func (count *CountAggregate) FieldID() int64 {
	return count.fieldID
}

type MinAggregate struct {
	fieldID int64
}

func (min *MinAggregate) Name() string {
	return kMin
}
func (min *MinAggregate) Update() {
}

func (min *MinAggregate) ToPB() *planpb.Aggregate {
	return &planpb.Aggregate{Op: planpb.AggregateOp_min, FieldId: min.FieldID()}
}

func (min *MinAggregate) FieldID() int64 {
	return min.fieldID
}

type MaxAggregate struct {
	fieldID int64
}

func (max *MaxAggregate) Name() string {
	return kMax
}
func (max *MaxAggregate) Update() {
}

func (max *MaxAggregate) ToPB() *planpb.Aggregate {
	return &planpb.Aggregate{Op: planpb.AggregateOp_max, FieldId: max.FieldID()}
}

func (max *MaxAggregate) FieldID() int64 {
	return max.fieldID
}

func AggregatesToPB(aggregates []AggregateBase) []*planpb.Aggregate {
	ret := make([]*planpb.Aggregate, len(aggregates))
	if aggregates != nil {
		for idx, agg := range aggregates {
			ret[idx] = agg.ToPB()
		}
	}
	return ret
}

type Entry struct {
	val interface{}
}

func NewEntry(v interface{}) *Entry {
	return &Entry{val: v}
}

type Row struct {
	entries []*Entry
}

func (r *Row) Equal(other *Row, keyCount int) bool {
	// Check if the number of entries is the same
	if len(r.entries) != len(other.entries) {
		return false
	}
	// Compare each entry for equality
	for i := 0; i < keyCount; i++ {
		if r.entries[i].val != other.entries[i].val {
			return false
		}
	}
	return true
}

func NewRow(entries []*Entry) *Row {
	return &Row{entries: entries}
}

type Bucket struct {
	rows []*Row
}

func (bucket *Bucket) AddRow(row *Row) {
	bucket.rows = append(bucket.rows, row)
}

func (bucket *Bucket) Accumulate(row *Row, idx int, keyCount int, aggs []AggregateBase) error {
	if idx >= len(bucket.rows) || idx < 0 {
		return fmt.Errorf("wrong idx:%d for bucket", idx)
	}
	//bucket.rows[idx].Equal()
	return nil
}

const NONE int = -1

func (bucket *Bucket) Find(row *Row, keyCount int) int {
	for idx, existingRow := range bucket.rows {
		if existingRow.Equal(row, keyCount) {
			return idx
		}
	}
	return NONE
}

func NewBucket() *Bucket {
	return &Bucket{rows: make([]*Row, 0, 1)}
}

func NewFieldAccessor(fieldType schemapb.DataType) (FieldAccessor, error) {
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
	Hash(idx int) uint64
	ValAt(idx int) interface{}
	SetVals(fieldData *schemapb.FieldData)
	RowCount() int
}

type Int32FieldAccessor struct {
	vals   []int32
	hasher hash.Hash64
	buffer []byte
}

func (i32Field *Int32FieldAccessor) Hash(idx int) uint64 {
	i32Field.hasher.Reset()
	val := i32Field.vals[idx]
	binary.LittleEndian.PutUint64(i32Field.buffer, uint64(val))
	i32Field.hasher.Write(i32Field.buffer)
	return i32Field.hasher.Sum64()
}

func (i32Field *Int32FieldAccessor) SetVals(fieldData *schemapb.FieldData) {
	i32Field.vals = fieldData.GetScalars().GetIntData().GetData()
}

func (i32Field *Int32FieldAccessor) RowCount() int {
	return len(i32Field.vals)
}

func (i32Field *Int32FieldAccessor) ValAt(idx int) interface{} {
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

func (i64Field *Int64FieldAccessor) Hash(idx int) uint64 {
	i64Field.hasher.Reset()
	val := i64Field.vals[idx]
	binary.LittleEndian.PutUint64(i64Field.buffer, uint64(val))
	i64Field.hasher.Write(i64Field.buffer)
	return i64Field.hasher.Sum64()
}

func (i64Field *Int64FieldAccessor) SetVals(fieldData *schemapb.FieldData) {
	i64Field.vals = fieldData.GetScalars().GetLongData().GetData()
}

func (i64Field *Int64FieldAccessor) RowCount() int {
	return len(i64Field.vals)
}

func (i64Field *Int64FieldAccessor) ValAt(idx int) interface{} {
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

func (boolField *BoolFieldAccessor) Hash(idx int) uint64 {
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

func (boolField *BoolFieldAccessor) SetVals(fieldData *schemapb.FieldData) {
	boolField.vals = fieldData.GetScalars().GetBoolData().GetData()
}

func (boolField *BoolFieldAccessor) RowCount() int {
	return len(boolField.vals)
}

func (boolField *BoolFieldAccessor) ValAt(idx int) interface{} {
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

func (f32FieldAccessor *Float32FieldAccessor) Hash(idx int) uint64 {
	f32FieldAccessor.hasher.Reset()
	val := f32FieldAccessor.vals[idx]
	binary.LittleEndian.PutUint32(f32FieldAccessor.buffer, math.Float32bits(val))
	f32FieldAccessor.hasher.Write(f32FieldAccessor.buffer[:4])
	return f32FieldAccessor.hasher.Sum64()
}

func (f32FieldAccessor *Float32FieldAccessor) SetVals(fieldData *schemapb.FieldData) {
	f32FieldAccessor.vals = fieldData.GetScalars().GetFloatData().GetData()
}

func (f32FieldAccessor *Float32FieldAccessor) RowCount() int {
	return len(f32FieldAccessor.vals)
}

func (f32FieldAccessor *Float32FieldAccessor) ValAt(idx int) interface{} {
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

func (f64Field *Float64FieldAccessor) Hash(idx int) uint64 {
	f64Field.hasher.Reset()
	val := f64Field.vals[idx]
	binary.LittleEndian.PutUint64(f64Field.buffer, math.Float64bits(val))
	f64Field.hasher.Write(f64Field.buffer)
	return f64Field.hasher.Sum64()
}

func (f64Field *Float64FieldAccessor) SetVals(fieldData *schemapb.FieldData) {
	f64Field.vals = fieldData.GetScalars().GetDoubleData().GetData()
}

func (f64Field *Float64FieldAccessor) RowCount() int {
	return len(f64Field.vals)
}

func (f64Field *Float64FieldAccessor) ValAt(idx int) interface{} {
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

func (stringField *StringFieldAccessor) Hash(idx int) uint64 {
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

func (stringField *StringFieldAccessor) SetVals(fieldData *schemapb.FieldData) {
	stringField.vals = fieldData.GetScalars().GetStringData().GetData()
}

func (stringField *StringFieldAccessor) RowCount() int {
	return len(stringField.vals)
}
func (stringField *StringFieldAccessor) ValAt(idx int) interface{} {
	return stringField.vals[idx]
}

func newStringFieldAccessor() FieldAccessor {
	return &StringFieldAccessor{hasher: fnv.New64a(), buffer: make([]byte, 1024)}
}
