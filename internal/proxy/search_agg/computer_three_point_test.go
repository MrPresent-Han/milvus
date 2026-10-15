package search_agg

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/pkg/v3/util/metric"
)

var threePointValueSink any

// A cached accessor must not allocate a new iterator for every scalar read.
func TestSearchAggregationCachesFieldIterator(t *testing.T) {
	ctx := newTestAggregationContext(t, 1, []LevelContext{{OwnFieldIDs: []int64{101}, Size: 1}}, nil, nil)
	data := &schemapb.SearchResultData{
		Topks: []int64{1},
		FieldsData: []*schemapb.FieldData{
			testLongFieldData(102, []int64{42}),
		},
		GroupByFieldValues: []*schemapb.FieldData{
			testLongFieldData(101, []int64{7}),
		},
	}
	computer := NewSearchAggregationComputer(data, ctx, metric.IP)
	ref := int64(0)
	value, isNull, err := computer.readValueByFieldID(ref, 102)
	require.NoError(t, err)
	require.False(t, isNull)
	require.Equal(t, int64(42), value)

	allocs := testing.AllocsPerRun(100, func() {
		value, _, err := computer.readValueByFieldID(ref, 102)
		if err != nil {
			panic(err)
		}
		threePointValueSink = value
	})
	require.Less(t, allocs, 0.5, "repeated field reads must reuse one iterator")
}

// Two metrics on price must share the value read and its aggregate input.
func TestSearchAggregationSharesSameFieldMetricInput(t *testing.T) {
	ctx := newTestAggregationContext(t, 1, []LevelContext{{
		OwnFieldIDs: []int64{101},
		Size:        1,
		Metrics: map[string]MetricSpec{
			"count_all": {Op: "count", FieldID: CountAllFieldID, FieldType: schemapb.DataType_None},
			"avg_price": {Op: "avg", FieldID: 102, FieldType: schemapb.DataType_Int64},
			"min_price": {Op: "min", FieldID: 102, FieldType: schemapb.DataType_Int64},
		},
	}}, nil, []int64{102})
	data := &schemapb.SearchResultData{
		NumQueries: 1,
		Topks:      []int64{3},
		Ids:        &schemapb.IDs{IdField: &schemapb.IDs_IntId{IntId: &schemapb.LongArray{Data: []int64{1, 2, 3}}}},
		Scores:     []float32{0.9, 0.8, 0.7},
		FieldsData: []*schemapb.FieldData{
			testLongFieldData(102, []int64{10, 20, 30}),
		},
		GroupByFieldValues: []*schemapb.FieldData{
			testLongFieldData(101, []int64{7, 7, 7}),
		},
	}
	computer := NewSearchAggregationComputer(data, ctx, metric.IP)
	result, err := computer.Compute(context.Background())
	require.NoError(t, err)
	require.Len(t, result[0], 1)
	require.Equal(t, int64(3), result[0][0].Metrics["count_all"])
	require.Equal(t, float64(20), result[0][0].Metrics["avg_price"])
	require.Equal(t, int64(10), result[0][0].Metrics["min_price"])

	allocs := testing.AllocsPerRun(100, func() {
		buckets, err := computer.Compute(context.Background())
		if err != nil {
			panic(err)
		}
		threePointValueSink = buckets
	})
	require.Less(t, allocs, 49.0, "three rows must avoid one duplicate metric input allocation each")
}
