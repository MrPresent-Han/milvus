package proxy

import (
	"context"
	"github.com/milvus-io/milvus/pkg/util/typeutil"

	"github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/internal/proto/internalpb"
	"github.com/milvus-io/milvus/internal/proto/planpb"
)

type milvusReducer interface {
	Reduce([]*internalpb.RetrieveResults) (*milvuspb.QueryResults, error)
}

func createMilvusReducer(ctx context.Context, params *queryParams, req *internalpb.RetrieveRequest, schema *schemapb.CollectionSchema, plan *planpb.PlanNode, collectionName string) milvusReducer {
	if plan.GetQuery().GetIsCount() {
		return &cntReducer{}
	} else if params.tryBestReduce {
		tryBestReduceParams := NewQueryParams(typeutil.Unlimited, params.offset, params.tryBestReduce)
		return newDefaultLimitReducer(ctx, tryBestReduceParams, req, schema, collectionName)
	}
	return newDefaultLimitReducer(ctx, params, req, schema, collectionName)
}
