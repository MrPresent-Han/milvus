package reduce

import (
	"fmt"

	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus/pkg/v2/proto/planpb"
	"github.com/milvus-io/milvus/pkg/v2/util/typeutil"
)

type ResultInfo struct {
	nq                  int64
	topK                int64
	metricType          string
	pkType              schemapb.DataType
	offset              int64
	groupByFieldId      int64
	groupByFieldType    schemapb.DataType
	groupByFieldTypeSet bool
	groupSize           int64
	isAdvance           bool
}

func NewReduceSearchResultInfo(
	nq int64,
	topK int64,
) *ResultInfo {
	return &ResultInfo{
		nq:   nq,
		topK: topK,
	}
}

func (r *ResultInfo) WithMetricType(metricType string) *ResultInfo {
	r.metricType = metricType
	return r
}

func (r *ResultInfo) WithPkType(pkType schemapb.DataType) *ResultInfo {
	r.pkType = pkType
	return r
}

func (r *ResultInfo) WithOffset(offset int64) *ResultInfo {
	r.offset = offset
	return r
}

func (r *ResultInfo) WithGroupByField(groupByField int64) *ResultInfo {
	r.groupByFieldId = groupByField
	return r
}

func (r *ResultInfo) WithGroupByFieldType(groupByFieldType schemapb.DataType) *ResultInfo {
	r.groupByFieldType = groupByFieldType
	r.groupByFieldTypeSet = true
	return r
}

func (r *ResultInfo) WithGroupSize(groupSize int64) *ResultInfo {
	r.groupSize = groupSize
	return r
}

func (r *ResultInfo) WithAdvance(advance bool) *ResultInfo {
	r.isAdvance = advance
	return r
}

func (r *ResultInfo) GetNq() int64 {
	return r.nq
}

func (r *ResultInfo) GetTopK() int64 {
	return r.topK
}

func (r *ResultInfo) GetMetricType() string {
	return r.metricType
}

func (r *ResultInfo) GetPkType() schemapb.DataType {
	return r.pkType
}

func (r *ResultInfo) GetOffset() int64 {
	return r.offset
}

func (r *ResultInfo) GetGroupByFieldId() int64 {
	return r.groupByFieldId
}

func (r *ResultInfo) GetGroupByFieldType() schemapb.DataType {
	return r.groupByFieldType
}

func (r *ResultInfo) IsGroupByFieldTypeSet() bool {
	return r.groupByFieldTypeSet
}

func (r *ResultInfo) GetGroupSize() int64 {
	return r.groupSize
}

func (r *ResultInfo) GetIsAdvance() bool {
	return r.isAdvance
}

func (r *ResultInfo) SetMetricType(metricType string) {
	r.metricType = metricType
}

func (r *ResultInfo) WithGroupByFieldTypeFromSearchPlan(schema *schemapb.CollectionSchema, serializedPlan []byte) (*ResultInfo, error) {
	if r.GetGroupByFieldId() <= 0 {
		return r, nil
	}
	if len(serializedPlan) == 0 {
		groupByFieldType, groupByFieldTypeSet, err := ResolveGroupByFieldType(schema, r.GetGroupByFieldId(), schemapb.DataType_None)
		if err != nil {
			return r, err
		}
		if groupByFieldTypeSet {
			r.WithGroupByFieldType(groupByFieldType)
		}
		return r, nil
	}
	var plan planpb.PlanNode
	if err := proto.Unmarshal(serializedPlan, &plan); err != nil {
		return r, err
	}
	queryInfo := plan.GetVectorAnns().GetQueryInfo()
	if queryInfo == nil {
		return r, fmt.Errorf("query info not found in search plan")
	}
	groupByFieldType, groupByFieldTypeSet, err := ResolveGroupByFieldType(schema, queryInfo.GetGroupByFieldId(), queryInfo.GetJsonType())
	if err != nil {
		return r, err
	}
	if groupByFieldTypeSet {
		r.WithGroupByFieldType(groupByFieldType)
	}
	return r, nil
}

// ResolveGroupByFieldType resolves the expected group-by payload type from request metadata and schema.
// Reducers use this instead of deriving metadata from another SearchResultData payload.
func ResolveGroupByFieldType(schema *schemapb.CollectionSchema, groupByFieldID int64, jsonType schemapb.DataType) (schemapb.DataType, bool, error) {
	if groupByFieldID <= 0 {
		return schemapb.DataType_None, false, nil
	}
	if jsonType != schemapb.DataType_None {
		return jsonType, true, nil
	}
	field := typeutil.GetFieldByID(schema, groupByFieldID)
	if field == nil {
		return schemapb.DataType_None, false, fmt.Errorf("group by field id %d not found in schema", groupByFieldID)
	}
	if field.GetDataType() == schemapb.DataType_JSON {
		return schemapb.DataType_None, false, nil
	}
	return field.GetDataType(), true, nil
}

// ValidateGroupByFieldValue checks the SearchResultData invariant not covered by generic field helpers:
// a non-empty group-by search result must carry group values aligned with its ids.
func ValidateGroupByFieldValue(data *schemapb.SearchResultData, expectedType schemapb.DataType, expectedTypeSet bool) error {
	idsLen := typeutil.GetSizeOfIDs(data.GetIds())
	if idsLen == 0 {
		return nil
	}
	field := data.GetGroupByFieldValue()
	if field == nil {
		return fmt.Errorf("group by field value is nil for non-empty search result, ids length=%d", idsLen)
	}
	if expectedTypeSet && field.GetType() != expectedType {
		return fmt.Errorf("group by field type mismatch, expected=%s, actual=%s", expectedType.String(), field.GetType().String())
	}

	validLen := len(field.GetValidData())
	if validLen > 0 && validLen != idsLen {
		return fmt.Errorf("group by valid data length mismatch, valid data length=%d, ids length=%d", validLen, idsLen)
	}
	dataLen, err := getGroupByFieldDataLen(field)
	if err != nil {
		return err
	}
	if validLen == 0 {
		if dataLen != idsLen {
			return fmt.Errorf("group by field value length mismatch, value length=%d, ids length=%d", dataLen, idsLen)
		}
		return nil
	}

	validCount := 0
	for _, valid := range field.GetValidData() {
		if valid {
			validCount++
		}
	}
	if dataLen != idsLen && dataLen != validCount {
		return fmt.Errorf("group by field value length mismatch, value length=%d, ids length=%d, valid count=%d", dataLen, idsLen, validCount)
	}
	return nil
}

func getGroupByFieldDataLen(field *schemapb.FieldData) (int, error) {
	scalars := field.GetScalars()
	if scalars == nil {
		return 0, fmt.Errorf("group by field value should be scalar, actual type=%s", field.GetType().String())
	}
	switch field.GetType() {
	case schemapb.DataType_Bool:
		return len(scalars.GetBoolData().GetData()), nil
	case schemapb.DataType_Int8, schemapb.DataType_Int16, schemapb.DataType_Int32:
		return len(scalars.GetIntData().GetData()), nil
	case schemapb.DataType_Int64:
		return len(scalars.GetLongData().GetData()), nil
	case schemapb.DataType_Timestamptz:
		return len(scalars.GetTimestamptzData().GetData()), nil
	case schemapb.DataType_Float:
		return len(scalars.GetFloatData().GetData()), nil
	case schemapb.DataType_Double:
		return len(scalars.GetDoubleData().GetData()), nil
	case schemapb.DataType_String, schemapb.DataType_VarChar, schemapb.DataType_Text:
		return len(scalars.GetStringData().GetData()), nil
	default:
		return 0, fmt.Errorf("unsupported group by field type: %s", field.GetType().String())
	}
}

type IReduceType int32

const (
	IReduceNoOrder IReduceType = iota
	IReduceInOrder
	IReduceInOrderForBest
)

func ShouldStopWhenDrained(reduceType IReduceType) bool {
	return reduceType == IReduceInOrder || reduceType == IReduceInOrderForBest
}

func ToReduceType(val int32) IReduceType {
	switch val {
	case 1:
		return IReduceInOrder
	case 2:
		return IReduceInOrderForBest
	default:
		return IReduceNoOrder
	}
}

func ShouldUseInputLimit(reduceType IReduceType) bool {
	return reduceType == IReduceNoOrder || reduceType == IReduceInOrder
}
