package reduce

import (
	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
)

type ReduceLevel int

const (
	Delegator    ReduceLevel = iota
	SubDelegator ReduceLevel = 2
	Proxy        ReduceLevel = 3
)

type ResultInfo struct {
	nq             int64
	topK           int64
	metricType     string
	pkType         schemapb.DataType
	offset         int64
	groupByFieldId int64
	groupSize      int64
	isAdvance      bool
	reduceLevel    ReduceLevel
}

func NewResultInfoNqTopK(
	nq int64,
	topK int64,
	groupByField int64,
	groupSize int64,
) *ResultInfo {
	return &ResultInfo{
		nq:             nq,
		topK:           topK,
		groupByFieldId: groupByField,
		groupSize:      groupSize,
	}
}

func NewReduceSearchResultInfo(
	nq int64,
	topK int64,
	metricType string,
	pkType schemapb.DataType,
	offset int64,
	groupByFieldId int64,
	groupSize int64,
	isAdvance bool,
	reduceLevel ReduceLevel,
) *ResultInfo {
	return &ResultInfo{
		nq:             nq,
		topK:           topK,
		metricType:     metricType,
		pkType:         pkType,
		offset:         offset,
		groupByFieldId: groupByFieldId,
		groupSize:      groupSize,
		isAdvance:      isAdvance,
		reduceLevel:    reduceLevel,
	}
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

func (r *ResultInfo) GetGroupSize() int64 {
	return r.groupSize
}

func (r *ResultInfo) GetIsAdvance() bool {
	return r.isAdvance
}

func (r *ResultInfo) GetReduceLevel() ReduceLevel {
	return r.reduceLevel
}

func (r *ResultInfo) SetMetricType(metricType string) {
	r.metricType = metricType
}
