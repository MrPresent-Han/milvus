package segments

import (
	"context"
	"github.com/milvus-io/milvus/internal/agg"
	"github.com/milvus-io/milvus/internal/proto/planpb"

	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/internal/proto/segcorepb"
)

type AggReducer struct {
	groupAggReducer *agg.GroupAggReducer
}

func NewAggReducer(groupByFieldIds []int64, aggregates []*planpb.Aggregate, schema *schemapb.CollectionSchema) *AggReducer {
	return &AggReducer{
		agg.NewGroupAggReducer(groupByFieldIds, aggregates, schema),
	}
}

func (reducer *AggReducer) Reduce(ctx context.Context, results []*segcorepb.RetrieveResults, segments []Segment, plan *RetrievePlan) (*segcorepb.RetrieveResults, error) {
	return reducer.groupAggReducer.Reduce(ctx, results)
}
