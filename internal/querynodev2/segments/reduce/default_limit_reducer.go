package reduce

import (
	"context"
	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/internal/proto/internalpb"
	"github.com/milvus-io/milvus/internal/proto/querypb"
	"github.com/milvus-io/milvus/internal/proto/segcorepb"
	"github.com/milvus-io/milvus/internal/querynodev2/segments"
	"github.com/milvus-io/milvus/internal/querynodev2/segments/segbase"
	"github.com/milvus-io/milvus/internal/util/reduce"
)

type defaultLimitReducer struct {
	req    *querypb.QueryRequest
	schema *schemapb.CollectionSchema
}

type mergeParam struct {
	limit          int64
	outputFieldsId []int64
	schema         *schemapb.CollectionSchema
	reduceType     reduce.IReduceType
}

func NewMergeParam(limit int64, outputFieldsId []int64, schema *schemapb.CollectionSchema, reduceType reduce.IReduceType) *mergeParam {
	return &mergeParam{
		limit:          limit,
		outputFieldsId: outputFieldsId,
		schema:         schema,
		reduceType:     reduceType,
	}
}

func (r *defaultLimitReducer) Reduce(ctx context.Context, results []*internalpb.RetrieveResults) (*internalpb.RetrieveResults, error) {
	reduceParam := NewMergeParam(r.req.GetReq().GetLimit(), r.req.GetReq().GetOutputFieldsId(),
		r.schema, reduce.ToReduceType(r.req.GetReq().GetReduceType()))
	return mergeInternalRetrieveResultsAndFillIfEmpty(ctx, results, reduceParam)
}

func newDefaultLimitReducer(req *querypb.QueryRequest, schema *schemapb.CollectionSchema) *defaultLimitReducer {
	return &defaultLimitReducer{
		req:    req,
		schema: schema,
	}
}

type defaultLimitReducerSegcore struct {
	req     *querypb.QueryRequest
	schema  *schemapb.CollectionSchema
	manager *segments.Manager
}

func (r *defaultLimitReducerSegcore) Reduce(ctx context.Context, results []*segcorepb.RetrieveResults, segments []segbase.Segment, plan *segbase.RetrievePlan) (*segcorepb.RetrieveResults, error) {
	mergeParam := NewMergeParam(r.req.GetReq().GetLimit(), r.req.GetReq().GetOutputFieldsId(), r.schema, reduce.ToReduceType(r.req.GetReq().GetReduceType()))
	return mergeSegcoreRetrieveResultsAndFillIfEmpty(ctx, results, mergeParam, segments, plan, r.manager)
}

func newDefaultLimitReducerSegcore(req *querypb.QueryRequest, schema *schemapb.CollectionSchema, manager *segments.Manager) *defaultLimitReducerSegcore {
	return &defaultLimitReducerSegcore{
		req:     req,
		schema:  schema,
		manager: manager,
	}
}
