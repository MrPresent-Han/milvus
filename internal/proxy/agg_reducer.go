package proxy

import (
	"context"
	"github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/internal/agg"
	"github.com/milvus-io/milvus/internal/proto/internalpb"
	"github.com/milvus-io/milvus/internal/proto/planpb"
)

type MilvusAggReducer struct {
	groupAggReducer *agg.GroupAggReducer
}

func NewMilvusAggReducer(groupByFieldIds []int64, aggregates []*planpb.Aggregate, schema *schemapb.CollectionSchema) *MilvusAggReducer {
	return &MilvusAggReducer{
		agg.NewGroupAggReducer(groupByFieldIds, aggregates, schema),
	}
}

func (reducer *MilvusAggReducer) Reduce(results []*internalpb.RetrieveResults) (*milvuspb.QueryResults, error) {
	reducedAggRes, err := reducer.groupAggReducer.Reduce(context.Background(), agg.InternalResult2AggResult(results))
	return agg.AggResult2MilvusResult(reducedAggRes), err
}
