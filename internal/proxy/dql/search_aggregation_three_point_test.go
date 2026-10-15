package dql

import (
	"context"
	"strconv"
	"testing"

	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/otel/trace"
	"google.golang.org/protobuf/encoding/protojson"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/proxy/search_agg"
	"github.com/milvus-io/milvus/pkg/v3/mlog"
	"github.com/milvus-io/milvus/pkg/v3/proto/internalpb"
	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/metric"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

var threePointReduceSink any

// The aggregation handoff must not copy all retained source field values.
func TestSearchAggregationReducerAvoidsFieldCopies(t *testing.T) {
	const groupCount = 200
	const rowCount = groupCount * 3
	ids := make([]int64, 0, rowCount)
	groups := make([]int64, 0, rowCount)
	prices := make([]int64, 0, rowCount)
	scores := make([]float32, 0, rowCount)
	for member := int64(0); member < 3; member++ {
		for group := int64(0); group < groupCount; group++ {
			ids = append(ids, member*1000+group)
			groups = append(groups, group)
			prices = append(prices, group*10+member)
			scores = append(scores, -float32(member*1000+group))
		}
	}
	source := []*internalpb.SearchResults{{MetricType: metric.L2, ResultData: &schemapb.SearchResultData{
		NumQueries: 1, TopK: groupCount, Topks: []int64{rowCount},
		Ids:                &schemapb.IDs{IdField: &schemapb.IDs_IntId{IntId: &schemapb.LongArray{Data: ids}}},
		Scores:             scores,
		FieldsData:         []*schemapb.FieldData{multiGroupByTestLongField(102, prices)},
		GroupByFieldValues: []*schemapb.FieldData{multiGroupByTestLongField(101, groups)},
	}}}
	op := &searchReduceOperator{
		traceCtx:           context.Background(),
		primaryFieldSchema: &schemapb.FieldSchema{FieldID: 100, Name: "id", DataType: schemapb.DataType_Int64, IsPrimaryKey: true},
		nq:                 1, topK: groupCount,
		queryInfos:          []*planpb.QueryInfo{{Topk: groupCount, GroupSize: 3, GroupByFieldIds: []int64{101}}},
		isSearchAggregation: true,
	}
	ctx := context.Background()
	_, err := op.run(ctx, trace.SpanFromContext(ctx), source)
	require.NoError(t, err)
	previousLevel := mlog.GetLevel()
	mlog.SetLevel(mlog.InfoLevel)
	defer mlog.SetLevel(previousLevel)
	result := testing.Benchmark(func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			output, err := op.run(ctx, trace.SpanFromContext(ctx), source)
			if err != nil {
				b.Fatal(err)
			}
			threePointReduceSink = output
		}
	})
	require.Less(t, result.AllocedBytesPerOp(), int64(100000), "retained fields should remain in source buffers")
}

func TestSearchAggregationRowsPreservesReduceErrorsAndEmptyQueries(t *testing.T) {
	for _, scenario := range []string{"nq", "topk", "scores", "group", "decode", "no_sources", "empty_shard", "pk_type"} {
		t.Run(scenario, func(t *testing.T) {
			data := &schemapb.SearchResultData{NumQueries: 2, TopK: 2, Topks: []int64{1, 0},
				Ids: testSearchResultIDs(1), Scores: []float32{1}, AllSearchCount: 7,
				GroupByFieldValues: []*schemapb.FieldData{multiGroupByTestLongField(101, []int64{3})}}
			wire := []*internalpb.SearchResults{{ResultData: data}}
			pkType := schemapb.DataType_Int64
			switch scenario {
			case "nq":
				data.NumQueries = 1
			case "topk":
				data.TopK = 1
			case "scores":
				data.Scores = nil
			case "group":
				data.GroupByFieldValues = nil
			case "decode":
				wire = []*internalpb.SearchResults{{SlicedBlob: []byte{0xff}}}
			case "no_sources":
				wire = nil
			case "empty_shard":
				data.Ids, data.Scores, data.Topks = testSearchResultIDs(), nil, []int64{0, 0}
				data.GroupByFieldValues = []*schemapb.FieldData{multiGroupByTestLongField(101, nil)}
			case "pk_type":
				pkType = schemapb.DataType_Float
			}
			info := &planpb.QueryInfo{GroupByFieldIds: []int64{101}, GroupSize: 1}
			want, oldErr := reduceResults(context.Background(), wire, 2, 2, 0, metric.IP, pkType, info, false, true, 0, nil)
			got, newErr := reduceSearchAggregationRows(context.Background(), wire, 2, 2, 0, 1, []int64{101}, metric.IP, pkType, 0, nil)
			if oldErr != nil {
				require.EqualError(t, newErr, oldErr.Error())
				require.Equal(t, merr.Code(oldErr), merr.Code(newErr))
				return
			}
			require.NoError(t, newErr)
			require.Empty(t, got.Rows)
			require.Equal(t, want.Results.Topks, got.Topks)
			require.Equal(t, want.Results.AllSearchCount, got.AllSearchCount)
		})
	}
}

// The row-reference path must enforce the historical output-size limit even
// though it no longer calls AppendFieldData to materialize each selected row.
func TestAggregationRowSizerPreservesAppendFieldDataCharge(t *testing.T) {
	fixtures := []string{
		`{"type":"Bool","scalars":{"boolData":{"data":[true,false,true]}}}`,
		`{"type":"Int32","scalars":{"intData":{"data":[1,0,3]}}}`,
		`{"type":"Int64","scalars":{"longData":{"data":[1,0,3]}}}`,
		`{"type":"Float","scalars":{"floatData":{"data":[1,0,3]}}}`,
		`{"type":"Double","scalars":{"doubleData":{"data":[1,0,3]}}}`,
		`{"type":"VarChar","scalars":{"stringData":{"data":["short","","a longer string"]}}}`,
		`{"type":"Array","scalars":{"arrayData":{"elementType":"Int64","data":[{"longData":{"data":[1,2]}},{"longData":{}},{"longData":{"data":[3]}}]}}}`,
		`{"type":"JSON","scalars":{"jsonData":{"data":["e30=","","WzEsMl0="]}}}`,
		`{"type":"Timestamptz","scalars":{"timestamptzData":{"data":[1,0,3]}}}`,
		`{"type":"Geometry","scalars":{"geometryData":{"data":["AQID","","BA=="]}}}`,
		`{"type":"Geometry","scalars":{"geometryWktData":{"data":["POINT (1 2)","","POINT (3 4)"]}}}`,
		`{"type":"FloatVector","vectors":{"dim":2,"floatVector":{"data":[1,2,3,4,5,6]}}}`,
		`{"type":"BinaryVector","vectors":{"dim":8,"binaryVector":"AQID"}}`,
		`{"type":"Float16Vector","vectors":{"dim":1,"float16Vector":"AQIDBAUG"}}`,
		`{"type":"BFloat16Vector","vectors":{"dim":1,"bfloat16Vector":"AQIDBAUG"}}`,
		`{"type":"Int8Vector","vectors":{"dim":1,"int8Vector":"AQID"}}`,
		`{"type":"SparseFloatVector","vectors":{"dim":8,"sparseFloatVector":{"dim":8,"contents":["","AAAAAAAAAAA=","AQAAAAAAAAACAAAAAAAAAA=="]}}}`,
	}
	for _, fixture := range fixtures {
		original := &schemapb.FieldData{}
		require.NoError(t, protojson.Unmarshal([]byte(fixture), original))
		original.FieldId = 100
		for _, nullable := range []bool{false, true} {
			field := proto.Clone(original).(*schemapb.FieldData)
			if nullable {
				typeutil.SetFieldDataValidData(field, []bool{true, false, true})
			}
			fields := []*schemapb.FieldData{field}
			sizer := newAggregationRowSizer(fields)
			indices := typeutil.NewFieldDataIdxComputer(fields)
			dst := typeutil.PrepareResultFieldData(fields, 5)
			for _, row := range []int64{2, 0, 1, 2, 0} {
				want := typeutil.AppendFieldData(dst, fields, row, indices.Compute(row)...)
				require.Equal(t, want, sizer.size(int(row)), "fixture=%s nullable=%t row=%d", fixture, nullable, row)
			}
		}
	}
}

// Row references must preserve the old materialized result for multi-query,
// cross-shard, nullable, and nested SearchAggregation requests.
func TestSearchAggregationRowsMatchMaterializedResults(t *testing.T) {
	for _, pkType := range []schemapb.DataType{schemapb.DataType_Int64, schemapb.DataType_VarChar} {
		for _, metricType := range []string{metric.IP, metric.L2} {
			for _, depth := range []int{1, 3} {
				metrics := map[string]search_agg.MetricSpec{
					"sum": {Op: "sum", FieldID: 104, FieldType: schemapb.DataType_Int64},
					"avg": {Op: "avg", FieldID: 104, FieldType: schemapb.DataType_Int64},
				}
				levels := []search_agg.LevelContext{
					{OwnFieldIDs: []int64{101}, Size: 3, Metrics: metrics,
						TopHits: &search_agg.TopHitsConfig{Size: 3, Sort: []search_agg.SortCriterion{{FieldID: 104, Dir: "desc"}}}},
					{OwnFieldIDs: []int64{102}, Size: 2, Metrics: metrics, TopHits: &search_agg.TopHitsConfig{Size: 2}},
					{OwnFieldIDs: []int64{103}, Size: 2, Metrics: metrics},
				}
				aggCtx, err := search_agg.NewContext(3, levels[:depth], []int64{101, 104, 105}, nil)
				require.NoError(t, err)
				sources := make([]*schemapb.SearchResultData, 4)
				wire := make([]*internalpb.SearchResults, len(sources))
				before := make([]proto.Message, len(sources))
				for shard := range sources {
					const rowsPerQuery = 20
					data := &schemapb.SearchResultData{NumQueries: 3, TopK: aggCtx.DerivedTopK, Topks: []int64{rowsPerQuery, 0, rowsPerQuery}, AllSearchCount: int64(shard + 1)}
					ids, prices := make([]int64, 0, rowsPerQuery*2), make([]int64, 0, rowsPerQuery*2)
					keys := [3][]int64{}
					var priceValid, vectorValid []bool
					var vectors []float32
					for qi := 0; qi < 3; qi += 2 {
						for row := 0; row < rowsPerQuery; row++ {
							ids = append(ids, int64(qi*1000+row*4+shard))
							prices = append(prices, int64(qi*100+row*7+shard))
							priceValid = append(priceValid, row%5 != 0)
							vectorValid = append(vectorValid, row%3 != 0)
							if row%3 != 0 {
								vectors = append(vectors, float32(qi), float32(row), float32(shard), 1)
							}
							keys[0] = append(keys[0], int64(row%3))
							keys[1] = append(keys[1], int64(row/3%2))
							keys[2] = append(keys[2], int64(row/6%2))
							score := float32(1000 - row*4 - shard)
							if metricType == metric.L2 {
								score = -float32(1 + row*4 + shard)
							}
							data.Scores = append(data.Scores, score)
						}
					}
					data.Ids = testSearchResultIDs(ids...)
					if pkType == schemapb.DataType_VarChar {
						values := make([]string, len(ids))
						for i, id := range ids {
							values[i] = strconv.FormatInt(id, 10)
						}
						data.Ids = testSearchResultStringIDs(values...)
					}
					price := multiGroupByTestLongField(104, prices)
					price.GetScalars().ValidData = priceValid
					data.FieldsData = []*schemapb.FieldData{price, {
						FieldId: 105, Type: schemapb.DataType_FloatVector,
						Field: &schemapb.FieldData_Vectors{Vectors: &schemapb.VectorField{Dim: 4, ValidData: vectorValid,
							Data: &schemapb.VectorField_FloatVector{FloatVector: &schemapb.FloatArray{Data: vectors}},
						}},
					}}
					for i := range keys {
						field := multiGroupByTestLongField(int64(101+i), keys[i])
						if depth == 3 && i > 0 {
							valid := make([]bool, len(keys[i]))
							for row := range valid {
								valid[row] = row%(7+i) != 0
							}
							field.GetScalars().ValidData = valid
						}
						data.GroupByFieldValues = append(data.GroupByFieldValues, field)
					}
					if depth == 1 && shard == 2 {
						data.GroupByFieldValue = data.GroupByFieldValues[0]
						data.GroupByFieldValue.FieldId = 0
						data.GroupByFieldValues = nil
					}
					sources[shard] = data
					wire[shard] = &internalpb.SearchResults{ResultData: data, MetricType: metricType}
					before[shard] = proto.Clone(data)
				}
				materialized, err := reduceSearchResultDataWithGroupBy(context.Background(), sources, 3, aggCtx.DerivedTopK, metricType,
					pkType, 0, aggCtx.DerivedGroupSize, aggCtx.AllGroupByFieldIDs(), true)
				require.NoError(t, err)
				want, err := search_agg.NewSearchAggregationComputer(materialized.Results, aggCtx, metricType).Compute(context.Background())
				require.NoError(t, err)
				wire[0].SlicedBlob, err = proto.Marshal(wire[0].ResultData)
				require.NoError(t, err)
				wire[0].ResultData = nil
				selected, err := reduceSearchAggregationRows(context.Background(), wire, 3, aggCtx.DerivedTopK, 0,
					aggCtx.DerivedGroupSize, aggCtx.AllGroupByFieldIDs(), metricType, pkType, 0, nil)
				require.NoError(t, err)
				require.Equal(t, int64(10), selected.AllSearchCount)
				require.Equal(t, materialized.Results.Topks, selected.Topks)
				require.Same(t, sources[1], selected.Sources[1], "in-process source columns must not be copied")
				computer := search_agg.NewSearchAggregationComputerFromRows(selected, aggCtx, metricType)
				for run := 0; run < 2; run++ {
					got, err := computer.Compute(context.Background())
					require.NoError(t, err)
					require.Equal(t, want, got)
				}
				for i, source := range sources {
					require.True(t, proto.Equal(before[i], source), "source %d was mutated", i)
				}
			}
		}
	}
}
