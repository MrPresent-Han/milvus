package search_agg

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
)

func TestBuildSearchAggregationContext(t *testing.T) {
	schema := testCollectionSchema()
	spec := &commonpb.SearchAggregationSpec{
		Fields: []string{"brand"},
		Size:   5,
		Metrics: map[string]*commonpb.MetricAggSpec{
			"avg_price": {Op: "avg", FieldName: "price"},
		},
		Order: []*commonpb.OrderSpec{{Key: "_count", Direction: "desc"}},
		TopHits: &commonpb.TopHitsSpec{
			Size: 2,
			Sort: []*commonpb.SortSpec{{FieldName: "_score", Direction: "desc"}},
		},
		SubAggregation: &commonpb.SearchAggregationSpec{
			Fields: []string{"category"},
			Size:   3,
			Metrics: map[string]*commonpb.MetricAggSpec{
				"sum_stock": {Op: "sum", FieldName: "stock"},
			},
			Order: []*commonpb.OrderSpec{{Key: "_key", Direction: "asc"}},
		},
	}

	ctx, err := BuildSearchAggregationContext(spec, schema, 2)
	require.NoError(t, err)
	require.Equal(t, int64(2), ctx.NQ)
	require.Len(t, ctx.Levels, 2)
	require.Equal(t, []int64{101}, ctx.Levels[0].OwnFieldIDs)
	require.Equal(t, []int64{102}, ctx.Levels[1].OwnFieldIDs)

	require.True(t, ctx.IsGroupByField(101))
	require.True(t, ctx.IsGroupByField(102))
	// Group-by fields must NOT leak into ExtraOutputFieldIDs — they come from field 17.
	require.False(t, ctx.IsGroupByField(103))
	require.False(t, ctx.IsGroupByField(104))

	require.Equal(t, []int64{103, 104}, ctx.ExtraOutputFieldIDs())
	require.Equal(t, ScoreFieldID, ctx.Levels[0].TopHits.Sort[0].FieldID)
}

func TestBuildMultiFieldGroupByInfo(t *testing.T) {
	schema := testCollectionSchema()
	spec := &commonpb.SearchAggregationSpec{
		Fields: []string{"brand"},
		Metrics: map[string]*commonpb.MetricAggSpec{
			"sum_stock": {Op: "sum", FieldName: "stock"},
			"score_avg": {Op: "avg", FieldName: "_score"},
			"doc_count": {Op: "count", FieldName: "*"},
		},
		TopHits: &commonpb.TopHitsSpec{
			Size: 1,
			Sort: []*commonpb.SortSpec{
				{FieldName: "price", Direction: "desc"},
				{FieldName: "_score", Direction: "desc"},
			},
		},
		SubAggregation: &commonpb.SearchAggregationSpec{
			Fields: []string{"category"},
		},
	}

	info, err := BuildMultiFieldGroupByInfo(spec, schema)
	require.NoError(t, err)
	// Only GroupByFieldIds travels downstream; metric / top_hits fields flow
	// through SearchRequest.OutputFieldsId via segcore's standard fields_data path.
	require.Equal(t, []int64{101, 102}, info.GetGroupByFieldIds())
}

func TestBuildSearchAggregationContextDuplicateFieldAcrossLevels(t *testing.T) {
	schema := testCollectionSchema()
	spec := &commonpb.SearchAggregationSpec{
		Fields: []string{"brand"},
		SubAggregation: &commonpb.SearchAggregationSpec{
			Fields: []string{"brand"},
		},
	}

	_, err := BuildSearchAggregationContext(spec, schema, 1)
	require.Error(t, err)
}

func testCollectionSchema() *schemapb.CollectionSchema {
	return &schemapb.CollectionSchema{
		Name: "agg_test",
		Fields: []*schemapb.FieldSchema{
			{FieldID: 100, Name: "id", DataType: schemapb.DataType_Int64, IsPrimaryKey: true},
			{FieldID: 101, Name: "brand", DataType: schemapb.DataType_VarChar},
			{FieldID: 102, Name: "category", DataType: schemapb.DataType_VarChar},
			{FieldID: 103, Name: "price", DataType: schemapb.DataType_Double},
			{FieldID: 104, Name: "stock", DataType: schemapb.DataType_Int64},
		},
	}
}
