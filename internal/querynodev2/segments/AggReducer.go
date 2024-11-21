package segments

import (
	"context"
	"github.com/milvus-io/milvus/internal/agg"
	"github.com/milvus-io/milvus/internal/proto/internalpb"
	"github.com/milvus-io/milvus/internal/proto/planpb"

	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/internal/proto/segcorepb"
)

type InternalAggReducer struct {
	groupAggReducer *agg.GroupAggReducer
}

func NewInternalAggReducer(groupByFieldIds []int64, aggregates []*planpb.Aggregate, schema *schemapb.CollectionSchema) *InternalAggReducer {
	return &InternalAggReducer{
		agg.NewGroupAggReducer(groupByFieldIds, aggregates, schema),
	}
}

func (reducer *InternalAggReducer) Reduce(ctx context.Context, results []*internalpb.RetrieveResults) (*internalpb.RetrieveResults, error) {
	reducedAggRes, err := reducer.groupAggReducer.Reduce(ctx, agg.InternalResult2AggResult(results))
	return agg.AggResult2internalResult(reducedAggRes), err
}

type SegcoreAggReducer struct {
	groupAggReducer *agg.GroupAggReducer
}

func NewSegcoreAggReducer(groupByFieldIds []int64, aggregates []*planpb.Aggregate, schema *schemapb.CollectionSchema) *SegcoreAggReducer {
	return &SegcoreAggReducer{
		agg.NewGroupAggReducer(groupByFieldIds, aggregates, schema),
	}
}

func (reducer *SegcoreAggReducer) Reduce(ctx context.Context, results []*segcorepb.RetrieveResults, segments []Segment, plan *RetrievePlan) (*segcorepb.RetrieveResults, error) {
	reducedAggRes, err := reducer.groupAggReducer.Reduce(ctx, agg.SegcoreResults2AggResult(results))
	return agg.AggResult2segcoreResult(reducedAggRes), err
}
