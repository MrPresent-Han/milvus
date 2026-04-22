package reduce

import (
	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
)

// ResultInfo carries reduce-stage parameters across proxy and querynode.
// Group-by is modelled as a single unified slice (groupByFieldIds); a 1-element
// slice is the degenerate single-field case, a ≥2-element slice is multi-field.
type ResultInfo struct {
	nq                  int64
	topK                int64
	metricType          string
	pkType              schemapb.DataType
	offset              int64
	groupSize           int64
	isAdvance           bool
	groupByFieldIds     []int64
	isSearchAggregation bool
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

func (r *ResultInfo) WithGroupSize(groupSize int64) *ResultInfo {
	r.groupSize = groupSize
	return r
}

func (r *ResultInfo) WithAdvance(advance bool) *ResultInfo {
	r.isAdvance = advance
	return r
}

// WithGroupByFieldIds sets the unified group-by field-id slice. A size-1 slice
// represents the degenerate single-field case; size>=2 is multi-field group-by.
func (r *ResultInfo) WithGroupByFieldIds(groupByFieldIds []int64) *ResultInfo {
	r.groupByFieldIds = groupByFieldIds
	return r
}

// WithGroupByFieldIdsFromProto consolidates a proto-level (legacyID, plural)
// pair into the unified slice: plural wins if non-empty; a positive legacy id
// wraps as a 1-element slice; non-positive legacy id with empty plural is
// treated as "no group-by". Used at the proxy↔QN proto-to-ResultInfo boundary.
func (r *ResultInfo) WithGroupByFieldIdsFromProto(legacyID int64, plural []int64) *ResultInfo {
	if len(plural) > 0 {
		r.groupByFieldIds = plural
	} else if legacyID > 0 {
		r.groupByFieldIds = []int64{legacyID}
	} else {
		r.groupByFieldIds = nil
	}
	return r
}

// WithMultiGroupByFieldIds is the legacy alias for WithGroupByFieldIds; both
// write to the same underlying slice. Retained until external Go callers drop.
func (r *ResultInfo) WithMultiGroupByFieldIds(groupByFieldIds []int64) *ResultInfo {
	return r.WithGroupByFieldIds(groupByFieldIds)
}

// WithSearchAggregation marks the reduce as serving a SearchAggregation request.
// The unified reducer uses this flag to skip both the offset application and the
// post-reduce per-group reorganisation step (the aggOp downstream re-organises).
func (r *ResultInfo) WithSearchAggregation(v bool) *ResultInfo {
	r.isSearchAggregation = v
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

func (r *ResultInfo) GetGroupSize() int64 {
	return r.groupSize
}

func (r *ResultInfo) GetIsAdvance() bool {
	return r.isAdvance
}

// HasGroupBy reports whether any group-by is active. Equivalent to
// len(GetGroupByFieldIds()) > 0.
func (r *ResultInfo) HasGroupBy() bool {
	return len(r.groupByFieldIds) > 0
}

// HasMultiGroupBy is a legacy alias for HasGroupBy; retained for QN-side
// readers that still branch on this name.
func (r *ResultInfo) HasMultiGroupBy() bool {
	return len(r.groupByFieldIds) > 0
}

// GetGroupByFieldIds returns the unified group-by field-id slice.
func (r *ResultInfo) GetGroupByFieldIds() []int64 {
	return r.groupByFieldIds
}

// GetMultiGroupByFieldIds is the legacy alias for GetGroupByFieldIds.
func (r *ResultInfo) GetMultiGroupByFieldIds() []int64 {
	return r.groupByFieldIds
}

// GetIsSearchAggregation reports whether the reduce is serving a
// SearchAggregation request (affects offset and per-group reorganisation).
func (r *ResultInfo) GetIsSearchAggregation() bool {
	return r.isSearchAggregation
}

// EffectiveOffset returns the offset the reducer should actually apply.
// SearchAggregation delegates pagination to the downstream aggOp, so its
// effective offset at the reduce stage is always zero.
func (r *ResultInfo) EffectiveOffset() int64 {
	if r.isSearchAggregation {
		return 0
	}
	return r.offset
}

func (r *ResultInfo) SetMetricType(metricType string) {
	r.metricType = metricType
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
