package search_agg

import (
	"math"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
)

func TestSearchAggregationComputerComputeSingleLevel(t *testing.T) {
	ctx := NewContext(1,
		[]LevelContext{{
			OwnFieldIDs: []int64{101},
			Size:        100,
			Metrics:     map[string]MetricSpec{"sum_value": {Op: "sum", FieldID: 102, FieldType: schemapb.DataType_Int64}},
			TopHits:     &TopHitsConfig{Size: 100},
			Order:       []OrderCriterion{{Key: "_count", Dir: "desc"}, {Key: "_key", Dir: "asc"}},
		}},
		nil,
		[]int64{102},
	)

	data := &schemapb.SearchResultData{
		NumQueries: 1,
		Topks:      []int64{4},
		Ids:        &schemapb.IDs{IdField: &schemapb.IDs_IntId{IntId: &schemapb.LongArray{Data: []int64{1, 2, 3, 4}}}},
		Scores:     []float32{0.9, 0.8, 0.7, 0.6},
		FieldsData: []*schemapb.FieldData{
			testLongFieldData(102, []int64{10, 20, 30, 40}),
		},
		GroupByFieldValues: []*schemapb.FieldData{
			testStringFieldData(101, []string{"A", "A", "B", "B"}),
		},
	}

	computer := NewSearchAggregationComputer(data, ctx)
	result, err := computer.Compute()
	require.NoError(t, err)
	require.Len(t, result, 1)
	require.Len(t, result[0], 2)

	require.Equal(t, "A", result[0][0].Key[101])
	// Metric result type follows source type (int64) post-Phase2, not a forced float64.
	require.Equal(t, int64(30), result[0][0].Metrics["sum_value"])
	require.Equal(t, int64(2), result[0][0].Count)

	require.Equal(t, "B", result[0][1].Key[101])
	require.Equal(t, int64(70), result[0][1].Metrics["sum_value"])
	require.Equal(t, int64(2), result[0][1].Count)
}

func TestSearchAggregationComputerComputeWithTopHits(t *testing.T) {
	ctx := NewContext(1,
		[]LevelContext{{
			OwnFieldIDs: []int64{101},
			Size:        100,
			TopHits:     &TopHitsConfig{Size: 2, Sort: []SortCriterion{{FieldID: 102, Dir: "desc"}}},
			Order:       []OrderCriterion{{Key: "_key", Dir: "asc"}},
		}},
		[]int64{102},
		[]int64{102},
	)

	data := &schemapb.SearchResultData{
		NumQueries: 1,
		Topks:      []int64{4},
		Ids:        &schemapb.IDs{IdField: &schemapb.IDs_IntId{IntId: &schemapb.LongArray{Data: []int64{1, 2, 3, 4}}}},
		Scores:     []float32{0.9, 0.8, 0.7, 0.6},
		FieldsData: []*schemapb.FieldData{
			testLongFieldData(102, []int64{10, 20, 30, 40}),
		},
		GroupByFieldValues: []*schemapb.FieldData{
			testStringFieldData(101, []string{"A", "A", "B", "B"}),
		},
	}

	computer := NewSearchAggregationComputer(data, ctx)
	result, err := computer.Compute()
	require.NoError(t, err)
	require.Len(t, result[0], 2)
	require.Len(t, result[0][0].Hits, 2)
	require.Len(t, result[0][1].Hits, 2)
	require.Equal(t, int64(20), result[0][0].Hits[0].Fields[102])
	require.Equal(t, int64(40), result[0][1].Hits[0].Fields[102])
}

func TestSearchAggregationComputerReadsGroupByFromField17(t *testing.T) {
	// group-by (brand, 101) comes from group_by_field_values; metric (price, 103),
	// top_hits sort (stock, 104), and user output (title, 105) from fields_data.
	ctx := NewContext(1,
		[]LevelContext{{
			OwnFieldIDs: []int64{101},
			Size:        100,
			Metrics:     map[string]MetricSpec{"sum_price": {Op: "sum", FieldID: 103, FieldType: schemapb.DataType_Int64}},
			TopHits:     &TopHitsConfig{Size: 2, Sort: []SortCriterion{{FieldID: 104, Dir: "desc"}}},
			Order:       []OrderCriterion{{Key: "_key", Dir: "asc"}},
		}},
		[]int64{105},
		[]int64{103, 104},
	)

	data := &schemapb.SearchResultData{
		NumQueries: 1,
		Topks:      []int64{4},
		Ids:        &schemapb.IDs{IdField: &schemapb.IDs_IntId{IntId: &schemapb.LongArray{Data: []int64{1, 2, 3, 4}}}},
		Scores:     []float32{0.9, 0.8, 0.7, 0.6},
		FieldsData: []*schemapb.FieldData{
			testLongFieldData(103, []int64{10, 30, 20, 40}),
			testLongFieldData(104, []int64{100, 200, 300, 400}),
			testStringFieldData(105, []string{"p1", "p2", "p3", "p4"}),
		},
		GroupByFieldValues: []*schemapb.FieldData{
			testStringFieldData(101, []string{"A", "A", "B", "B"}),
		},
	}

	computer := NewSearchAggregationComputer(data, ctx)
	result, err := computer.Compute()
	require.NoError(t, err)
	require.Len(t, result, 1)
	require.Len(t, result[0], 2)

	require.Equal(t, "A", result[0][0].Key[101])
	require.Equal(t, int64(40), result[0][0].Metrics["sum_price"])
	require.Len(t, result[0][0].Hits, 2)
	require.Equal(t, "p2", result[0][0].Hits[0].Fields[105])
	_, hasPriceInHit := result[0][0].Hits[0].Fields[103]
	require.False(t, hasPriceInHit)
	_, hasStockInHit := result[0][0].Hits[0].Fields[104]
	require.False(t, hasStockInHit)

	require.Equal(t, "B", result[0][1].Key[101])
	require.Equal(t, int64(60), result[0][1].Metrics["sum_price"])
	require.Equal(t, "p4", result[0][1].Hits[0].Fields[105])
}

func TestSearchAggregationComputerNormalizesInt32GroupKey(t *testing.T) {
	// A shard returns an int32 group-by column. NormalizeScalar collapses to
	// int64 before hashing so grouping stays consistent with other int widths.
	ctx := NewContext(1,
		[]LevelContext{{
			OwnFieldIDs: []int64{101},
			Size:        100,
			Metrics:     map[string]MetricSpec{"sum_value": {Op: "sum", FieldID: 102, FieldType: schemapb.DataType_Int64}},
			TopHits:     &TopHitsConfig{Size: 100},
		}},
		nil,
		[]int64{102},
	)

	data := &schemapb.SearchResultData{
		NumQueries: 1,
		Topks:      []int64{2},
		Ids:        &schemapb.IDs{IdField: &schemapb.IDs_IntId{IntId: &schemapb.LongArray{Data: []int64{1, 2}}}},
		Scores:     []float32{0.9, 0.8},
		FieldsData: []*schemapb.FieldData{testLongFieldData(102, []int64{10, 20})},
		GroupByFieldValues: []*schemapb.FieldData{
			{
				FieldId: 101,
				Type:    schemapb.DataType_Int32,
				Field: &schemapb.FieldData_Scalars{Scalars: &schemapb.ScalarField{
					Data: &schemapb.ScalarField_IntData{IntData: &schemapb.IntArray{Data: []int32{42, 42}}},
				}},
			},
		},
	}

	computer := NewSearchAggregationComputer(data, ctx)
	result, err := computer.Compute()
	require.NoError(t, err)
	require.Len(t, result[0], 1)
	require.Equal(t, int64(2), result[0][0].Count)
	require.Equal(t, int64(30), result[0][0].Metrics["sum_value"])
}

func TestSearchAggregationComputerNaNDistinctBuckets(t *testing.T) {
	// Two NaN group-by values must NOT merge (NaN != NaN).
	ctx := NewContext(1,
		[]LevelContext{{OwnFieldIDs: []int64{101}, Size: 100, TopHits: &TopHitsConfig{Size: 100}}},
		nil,
		nil,
	)

	data := &schemapb.SearchResultData{
		NumQueries: 1,
		Topks:      []int64{2},
		Ids:        &schemapb.IDs{IdField: &schemapb.IDs_IntId{IntId: &schemapb.LongArray{Data: []int64{1, 2}}}},
		Scores:     []float32{0.9, 0.8},
		FieldsData: nil,
		GroupByFieldValues: []*schemapb.FieldData{
			{
				FieldId: 101,
				Type:    schemapb.DataType_Double,
				Field: &schemapb.FieldData_Scalars{Scalars: &schemapb.ScalarField{
					Data: &schemapb.ScalarField_DoubleData{DoubleData: &schemapb.DoubleArray{
						Data: []float64{math.NaN(), math.NaN()},
					}},
				}},
			},
		},
	}

	computer := NewSearchAggregationComputer(data, ctx)
	result, err := computer.Compute()
	require.NoError(t, err)
	require.Len(t, result[0], 2, "two NaN rows must stay in distinct buckets")
}

func TestSearchAggregationComputerNullGrouping(t *testing.T) {
	// Two null group-by values must merge (null == null for grouping).
	ctx := NewContext(1,
		[]LevelContext{{OwnFieldIDs: []int64{101}, Size: 100, TopHits: &TopHitsConfig{Size: 100}}},
		nil,
		nil,
	)

	data := &schemapb.SearchResultData{
		NumQueries: 1,
		Topks:      []int64{2},
		Ids:        &schemapb.IDs{IdField: &schemapb.IDs_IntId{IntId: &schemapb.LongArray{Data: []int64{1, 2}}}},
		Scores:     []float32{0.9, 0.8},
		FieldsData: nil,
		GroupByFieldValues: []*schemapb.FieldData{
			{
				FieldId:   101,
				Type:      schemapb.DataType_Int64,
				ValidData: []bool{false, false},
				Field: &schemapb.FieldData_Scalars{Scalars: &schemapb.ScalarField{
					Data: &schemapb.ScalarField_LongData{LongData: &schemapb.LongArray{Data: []int64{0, 0}}},
				}},
			},
		},
	}

	computer := NewSearchAggregationComputer(data, ctx)
	result, err := computer.Compute()
	require.NoError(t, err)
	require.Len(t, result[0], 1, "two null rows must merge into a single bucket")
	require.Equal(t, int64(2), result[0][0].Count)
}

func TestSearchAggregationComputerStringMinMax(t *testing.T) {
	// min/max on a VarChar column returns a string through the MetricValue
	// oneof rather than being forced into float64 like the pre-Phase2 code.
	ctx := NewContext(1,
		[]LevelContext{{
			OwnFieldIDs: []int64{101},
			Size:        100,
			Metrics: map[string]MetricSpec{
				"min_title": {Op: "min", FieldID: 102, FieldType: schemapb.DataType_VarChar},
				"max_title": {Op: "max", FieldID: 102, FieldType: schemapb.DataType_VarChar},
			},
			TopHits: &TopHitsConfig{Size: 100},
		}},
		nil,
		[]int64{102},
	)

	data := &schemapb.SearchResultData{
		NumQueries: 1,
		Topks:      []int64{3},
		Ids:        &schemapb.IDs{IdField: &schemapb.IDs_IntId{IntId: &schemapb.LongArray{Data: []int64{1, 2, 3}}}},
		Scores:     []float32{0.9, 0.8, 0.7},
		FieldsData: []*schemapb.FieldData{
			testStringFieldData(102, []string{"apple", "banana", "cherry"}),
		},
		GroupByFieldValues: []*schemapb.FieldData{
			testStringFieldData(101, []string{"A", "A", "A"}),
		},
	}

	computer := NewSearchAggregationComputer(data, ctx)
	result, err := computer.Compute()
	require.NoError(t, err)
	require.Len(t, result[0], 1)
	require.Equal(t, "apple", result[0][0].Metrics["min_title"])
	require.Equal(t, "cherry", result[0][0].Metrics["max_title"])
}

func TestSearchAggregationComputerAvgMetric(t *testing.T) {
	// avg expands into (sum, count) under the hood; finalizeMetrics turns
	// that pair back into a float64 ratio.
	ctx := NewContext(1,
		[]LevelContext{{
			OwnFieldIDs: []int64{101},
			Size:        100,
			Metrics:     map[string]MetricSpec{"avg_value": {Op: "avg", FieldID: 102, FieldType: schemapb.DataType_Int64}},
			TopHits:     &TopHitsConfig{Size: 100},
		}},
		nil,
		[]int64{102},
	)

	data := &schemapb.SearchResultData{
		NumQueries: 1,
		Topks:      []int64{3},
		Ids:        &schemapb.IDs{IdField: &schemapb.IDs_IntId{IntId: &schemapb.LongArray{Data: []int64{1, 2, 3}}}},
		Scores:     []float32{0.9, 0.8, 0.7},
		FieldsData: []*schemapb.FieldData{
			testLongFieldData(102, []int64{10, 20, 30}),
		},
		GroupByFieldValues: []*schemapb.FieldData{
			testStringFieldData(101, []string{"A", "A", "A"}),
		},
	}

	computer := NewSearchAggregationComputer(data, ctx)
	result, err := computer.Compute()
	require.NoError(t, err)
	require.Len(t, result[0], 1)
	require.Equal(t, float64(20), result[0][0].Metrics["avg_value"])
}

func TestSearchAggregationComputerErrorsWhenGroupByMissing(t *testing.T) {
	// Upstream reducer is expected to populate group_by_field_values; if it
	// didn't, Compute() must surface a clear error rather than silently fall
	// back to the fields_data channel.
	ctx := NewContext(1,
		[]LevelContext{{OwnFieldIDs: []int64{101}, Size: 100, TopHits: &TopHitsConfig{Size: 100}}},
		nil,
		nil,
	)
	data := &schemapb.SearchResultData{
		NumQueries:         1,
		Topks:              []int64{1},
		Ids:                &schemapb.IDs{IdField: &schemapb.IDs_IntId{IntId: &schemapb.LongArray{Data: []int64{1}}}},
		Scores:             []float32{0.9},
		FieldsData:         nil,
		GroupByFieldValues: nil,
	}

	computer := NewSearchAggregationComputer(data, ctx)
	_, err := computer.Compute()
	require.Error(t, err)
	require.Contains(t, err.Error(), "group-by field 101 missing from group_by_field_values")
}

func testStringFieldData(fieldID int64, values []string) *schemapb.FieldData {
	return &schemapb.FieldData{
		FieldId: fieldID,
		Type:    schemapb.DataType_VarChar,
		Field: &schemapb.FieldData_Scalars{Scalars: &schemapb.ScalarField{
			Data: &schemapb.ScalarField_StringData{StringData: &schemapb.StringArray{Data: values}},
		}},
	}
}

func testLongFieldData(fieldID int64, values []int64) *schemapb.FieldData {
	return &schemapb.FieldData{
		FieldId: fieldID,
		Type:    schemapb.DataType_Int64,
		Field: &schemapb.FieldData_Scalars{Scalars: &schemapb.ScalarField{
			Data: &schemapb.ScalarField_LongData{LongData: &schemapb.LongArray{Data: values}},
		}},
	}
}
