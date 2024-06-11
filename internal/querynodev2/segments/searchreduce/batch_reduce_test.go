// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package segments

import (
	"context"
	"fmt"
	"github.com/milvus-io/milvus/internal/querynodev2/segments"
	"log"
	"math"
	"testing"

	"github.com/golang/protobuf/proto"
	"github.com/stretchr/testify/suite"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/internal/proto/datapb"
	"github.com/milvus-io/milvus/internal/proto/planpb"
	"github.com/milvus-io/milvus/internal/proto/querypb"
	storage "github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/internal/util/initcore"
	"github.com/milvus-io/milvus/pkg/common"
	"github.com/milvus-io/milvus/pkg/util/funcutil"
	"github.com/milvus-io/milvus/pkg/util/paramtable"
	"github.com/milvus-io/milvus/pkg/util/testutils"
	"github.com/milvus-io/milvus/pkg/util/typeutil"
)

type ReduceSuite struct {
	suite.Suite
	chunkManager storage.ChunkManager
	rootPath     string

	collectionID int64
	partitionID  int64
	segmentID    int64
	collection   *segments.Collection
	segment      segments.Segment
}

func (suite *ReduceSuite) SetupSuite() {
	paramtable.Init()
}

func (suite *ReduceSuite) SetupTest() {
	var err error
	ctx := context.Background()
	msgLength := 100

	suite.rootPath = suite.T().Name()
	chunkManagerFactory := storage.NewTestChunkManagerFactory(paramtable.Get(), suite.rootPath)
	suite.chunkManager, _ = chunkManagerFactory.NewPersistentStorageChunkManager(ctx)
	initcore.InitRemoteChunkManager(paramtable.Get())

	suite.collectionID = 100
	suite.partitionID = 10
	suite.segmentID = 1
	schema := segments.GenTestCollectionSchema("test-reduce", schemapb.DataType_Int64, true)
	suite.collection = segments.NewCollection(suite.collectionID,
		schema,
		segments.GenTestIndexMeta(suite.collectionID, schema),
		&querypb.LoadMetaInfo{
			LoadType: querypb.LoadType_LoadCollection,
		})
	suite.segment, err = segments.NewSegment(ctx,
		suite.collection,
		segments.SegmentTypeSealed,
		0,
		&querypb.SegmentLoadInfo{
			SegmentID:     suite.segmentID,
			CollectionID:  suite.collectionID,
			PartitionID:   suite.partitionID,
			NumOfRows:     int64(msgLength),
			InsertChannel: fmt.Sprintf("by-dev-rootcoord-dml_0_%dv0", suite.collectionID),
			Level:         datapb.SegmentLevel_Legacy,
		},
	)
	suite.Require().NoError(err)

	binlogs, _, err := segments.SaveBinLog(ctx,
		suite.collectionID,
		suite.partitionID,
		suite.segmentID,
		msgLength,
		schema,
		suite.chunkManager,
	)
	suite.Require().NoError(err)
	for _, binlog := range binlogs {
		err = suite.segment.(*segments.LocalSegment).LoadFieldData(ctx, binlog.FieldID, int64(msgLength), binlog, false)
		suite.Require().NoError(err)
	}
}

func (suite *ReduceSuite) TearDownTest() {
	suite.segment.Release(context.Background())
	segments.DeleteCollection(suite.collection)
	ctx := context.Background()
	suite.chunkManager.RemoveWithPrefix(ctx, suite.rootPath)
}

func (suite *ReduceSuite) TestReduceParseSliceInfo() {
	originNQs := []int64{2, 3, 2}
	originTopKs := []int64{10, 5, 20}
	nqPerSlice := int64(2)
	sInfo := ParseSliceInfo(originNQs, originTopKs, nqPerSlice)

	expectedSliceNQs := []int64{2, 2, 1, 2}
	expectedSliceTopKs := []int64{10, 5, 5, 20}
	suite.True(funcutil.SliceSetEqual(sInfo.SliceNQs, expectedSliceNQs))
	suite.True(funcutil.SliceSetEqual(sInfo.SliceTopKs, expectedSliceTopKs))
}

func (suite *ReduceSuite) TestReduceAllFunc() {
	nq := int64(10)

	// TODO: replace below by genPlaceholderGroup(nq)
	vec := testutils.GenerateFloatVectors(1, segments.DefaultDim)
	var searchRawData []byte
	for i, ele := range vec {
		buf := make([]byte, 4)
		common.Endian.PutUint32(buf, math.Float32bits(ele+float32(i*2)))
		searchRawData = append(searchRawData, buf...)
	}

	placeholderValue := commonpb.PlaceholderValue{
		Tag:    "$0",
		Type:   commonpb.PlaceholderType_FloatVector,
		Values: [][]byte{},
	}

	for i := 0; i < int(nq); i++ {
		placeholderValue.Values = append(placeholderValue.Values, searchRawData)
	}

	placeholderGroup := commonpb.PlaceholderGroup{
		Placeholders: []*commonpb.PlaceholderValue{&placeholderValue},
	}

	placeGroupByte, err := proto.Marshal(&placeholderGroup)
	if err != nil {
		log.Print("marshal placeholderGroup failed")
	}

	planStr := `vector_anns: <
                 field_id: 107
                 query_info: <
                   topk: 10
                   round_decimal: 6
                   metric_type: "L2"
                   search_params: "{\"nprobe\": 10}"
                 >
                 placeholder_tag: "$0"
               >`
	var planpb planpb.PlanNode
	proto.UnmarshalText(planStr, &planpb)
	serializedPlan, err := proto.Marshal(&planpb)
	suite.NoError(err)
	plan, err := segments.CreateSearchPlanByExpr(context.Background(), suite.collection, serializedPlan)
	suite.NoError(err)
	searchReq, err := segments.ParseSearchRequest(context.Background(), plan, placeGroupByte)
	searchReq.SetMVCCTs(typeutil.MaxTimestamp)
	suite.NoError(err)
	defer searchReq.Delete()

	searchResult, err := suite.segment.Search(context.Background(), searchReq)
	suite.NoError(err)

	err = checkSearchResult(context.Background(), nq, plan, searchResult)
	suite.NoError(err)
}

func (suite *ReduceSuite) TestReduceInvalid() {
	plan := &segments.SearchPlan{}
	_, err := ReduceSearchResultsAndFillData(context.Background(), plan, nil, 1, nil, nil)
	suite.Error(err)

	searchReq, err := segments.GenSearchPlanAndRequests(suite.collection, []int64{suite.segmentID}, segments.IndexHNSW, 10)
	suite.NoError(err)
	searchResults := make([]*segments.SearchResult, 0)
	searchResults = append(searchResults, nil)
	_, err = ReduceSearchResultsAndFillData(context.Background(), searchReq.Plan(), searchResults, 1, []int64{10}, []int64{10})
	suite.Error(err)
}

func (suite *ReduceSuite) TestResult_ReduceSearchResultData() {
	const (
		nq         = 1
		topk       = 4
		metricType = "L2"
	)
	suite.Run("case1", func() {
		ids := []int64{1, 2, 3, 4}
		scores := []float32{-1.0, -2.0, -3.0, -4.0}
		topks := []int64{int64(len(ids))}
		data1 := segments.GenSearchResultData(nq, topk, ids, scores, topks)
		data2 := segments.GenSearchResultData(nq, topk, ids, scores, topks)
		dataArray := make([]*schemapb.SearchResultData, 0)
		dataArray = append(dataArray, data1)
		dataArray = append(dataArray, data2)
		res, err := ReduceSearchResultData(context.TODO(), dataArray, nq, topk)
		suite.Nil(err)
		suite.Equal(ids, res.Ids.GetIntId().Data)
		suite.Equal(scores, res.Scores)
	})
	suite.Run("case2", func() {
		ids1 := []int64{1, 2, 3, 4}
		scores1 := []float32{-1.0, -2.0, -3.0, -4.0}
		topks1 := []int64{int64(len(ids1))}
		ids2 := []int64{5, 1, 3, 4}
		scores2 := []float32{-1.0, -1.0, -3.0, -4.0}
		topks2 := []int64{int64(len(ids2))}
		data1 := segments.GenSearchResultData(nq, topk, ids1, scores1, topks1)
		data2 := segments.GenSearchResultData(nq, topk, ids2, scores2, topks2)
		dataArray := make([]*schemapb.SearchResultData, 0)
		dataArray = append(dataArray, data1)
		dataArray = append(dataArray, data2)
		res, err := ReduceSearchResultData(context.TODO(), dataArray, nq, topk)
		suite.Nil(err)
		suite.ElementsMatch([]int64{1, 5, 2, 3}, res.Ids.GetIntId().Data)
	})
}

func (suite *ReduceSuite) TestResult_SearchGroupByResult() {
	const (
		nq   = 1
		topk = 4
	)
	suite.Run("reduce_group_by_int", func() {
		ids1 := []int64{1, 2, 3, 4}
		scores1 := []float32{-1.0, -2.0, -3.0, -4.0}
		topks1 := []int64{int64(len(ids1))}
		ids2 := []int64{5, 1, 3, 4}
		scores2 := []float32{-1.0, -1.0, -3.0, -4.0}
		topks2 := []int64{int64(len(ids2))}
		data1 := segments.GenSearchResultData(nq, topk, ids1, scores1, topks1)
		data2 := segments.GenSearchResultData(nq, topk, ids2, scores2, topks2)
		data1.GroupByFieldValue = &schemapb.FieldData{
			Type: schemapb.DataType_Int8,
			Field: &schemapb.FieldData_Scalars{
				Scalars: &schemapb.ScalarField{
					Data: &schemapb.ScalarField_IntData{
						IntData: &schemapb.IntArray{
							Data: []int32{2, 3, 4, 5},
						},
					},
				},
			},
		}
		data2.GroupByFieldValue = &schemapb.FieldData{
			Type: schemapb.DataType_Int8,
			Field: &schemapb.FieldData_Scalars{
				Scalars: &schemapb.ScalarField{
					Data: &schemapb.ScalarField_IntData{
						IntData: &schemapb.IntArray{
							Data: []int32{2, 3, 4, 5},
						},
					},
				},
			},
		}
		dataArray := make([]*schemapb.SearchResultData, 0)
		dataArray = append(dataArray, data1)
		dataArray = append(dataArray, data2)
		res, err := ReduceSearchResultData(context.TODO(), dataArray, nq, topk)
		suite.Nil(err)
		suite.ElementsMatch([]int64{1, 2, 3, 4}, res.Ids.GetIntId().Data)
		suite.ElementsMatch([]float32{-1.0, -2.0, -3.0, -4.0}, res.Scores)
		suite.ElementsMatch([]int32{2, 3, 4, 5}, res.GroupByFieldValue.GetScalars().GetIntData().Data)
	})
	suite.Run("reduce_group_by_bool", func() {
		ids1 := []int64{1, 2}
		scores1 := []float32{-1.0, -2.0}
		topks1 := []int64{int64(len(ids1))}
		ids2 := []int64{3, 4}
		scores2 := []float32{-1.0, -1.0}
		topks2 := []int64{int64(len(ids2))}
		data1 := segments.GenSearchResultData(nq, topk, ids1, scores1, topks1)
		data2 := segments.GenSearchResultData(nq, topk, ids2, scores2, topks2)
		data1.GroupByFieldValue = &schemapb.FieldData{
			Type: schemapb.DataType_Bool,
			Field: &schemapb.FieldData_Scalars{
				Scalars: &schemapb.ScalarField{
					Data: &schemapb.ScalarField_BoolData{
						BoolData: &schemapb.BoolArray{
							Data: []bool{true, false},
						},
					},
				},
			},
		}
		data2.GroupByFieldValue = &schemapb.FieldData{
			Type: schemapb.DataType_Bool,
			Field: &schemapb.FieldData_Scalars{
				Scalars: &schemapb.ScalarField{
					Data: &schemapb.ScalarField_BoolData{
						BoolData: &schemapb.BoolArray{
							Data: []bool{true, false},
						},
					},
				},
			},
		}
		dataArray := make([]*schemapb.SearchResultData, 0)
		dataArray = append(dataArray, data1)
		dataArray = append(dataArray, data2)
		res, err := ReduceSearchResultData(context.TODO(), dataArray, nq, topk)
		suite.Nil(err)
		suite.ElementsMatch([]int64{1, 4}, res.Ids.GetIntId().Data)
		suite.ElementsMatch([]float32{-1.0, -1.0}, res.Scores)
		suite.ElementsMatch([]bool{true, false}, res.GroupByFieldValue.GetScalars().GetBoolData().Data)
	})
	suite.Run("reduce_group_by_string", func() {
		ids1 := []int64{1, 2, 3, 4}
		scores1 := []float32{-1.0, -2.0, -3.0, -4.0}
		topks1 := []int64{int64(len(ids1))}
		ids2 := []int64{5, 1, 3, 4}
		scores2 := []float32{-1.0, -1.0, -3.0, -4.0}
		topks2 := []int64{int64(len(ids2))}
		data1 := segments.GenSearchResultData(nq, topk, ids1, scores1, topks1)
		data2 := segments.GenSearchResultData(nq, topk, ids2, scores2, topks2)
		data1.GroupByFieldValue = &schemapb.FieldData{
			Type: schemapb.DataType_VarChar,
			Field: &schemapb.FieldData_Scalars{
				Scalars: &schemapb.ScalarField{
					Data: &schemapb.ScalarField_StringData{
						StringData: &schemapb.StringArray{
							Data: []string{"1", "2", "3", "4"},
						},
					},
				},
			},
		}
		data2.GroupByFieldValue = &schemapb.FieldData{
			Type: schemapb.DataType_VarChar,
			Field: &schemapb.FieldData_Scalars{
				Scalars: &schemapb.ScalarField{
					Data: &schemapb.ScalarField_StringData{
						StringData: &schemapb.StringArray{
							Data: []string{"1", "2", "3", "4"},
						},
					},
				},
			},
		}
		dataArray := make([]*schemapb.SearchResultData, 0)
		dataArray = append(dataArray, data1)
		dataArray = append(dataArray, data2)
		res, err := ReduceSearchResultData(context.TODO(), dataArray, nq, topk)
		suite.Nil(err)
		suite.ElementsMatch([]int64{1, 2, 3, 4}, res.Ids.GetIntId().Data)
		suite.ElementsMatch([]float32{-1.0, -2.0, -3.0, -4.0}, res.Scores)
		suite.ElementsMatch([]string{"1", "2", "3", "4"}, res.GroupByFieldValue.GetScalars().GetStringData().Data)
	})
}

func (suite *ReduceSuite) TestResult_SelectSearchResultData_int() {
	type args struct {
		dataArray     []*schemapb.SearchResultData
		resultOffsets [][]int64
		offsets       []int64
		topk          int64
		nq            int64
		qi            int64
	}
	suite.Run("Integer ID", func() {
		tests := []struct {
			name string
			args args
			want int
		}{
			{
				args: args{
					dataArray: []*schemapb.SearchResultData{
						{
							Ids: &schemapb.IDs{
								IdField: &schemapb.IDs_IntId{
									IntId: &schemapb.LongArray{
										Data: []int64{11, 9, 7, 5, 3, 1},
									},
								},
							},
							Scores: []float32{1.1, 0.9, 0.7, 0.5, 0.3, 0.1},
							Topks:  []int64{2, 2, 2},
						},
						{
							Ids: &schemapb.IDs{
								IdField: &schemapb.IDs_IntId{
									IntId: &schemapb.LongArray{
										Data: []int64{12, 10, 8, 6, 4, 2},
									},
								},
							},
							Scores: []float32{1.2, 1.0, 0.8, 0.6, 0.4, 0.2},
							Topks:  []int64{2, 2, 2},
						},
					},
					resultOffsets: [][]int64{{0, 2, 4}, {0, 2, 4}},
					offsets:       []int64{0, 1},
					topk:          2,
					nq:            3,
					qi:            0,
				},
				want: 0,
			},
		}
		for _, tt := range tests {
			suite.Run(tt.name, func() {
				if got := SelectSearchResultData(tt.args.dataArray, tt.args.resultOffsets, tt.args.offsets, tt.args.qi); got != tt.want {
					suite.T().Errorf("SelectSearchResultData() = %v, want %v", got, tt.want)
				}
			})
		}
	})

	suite.Run("Integer ID with bad score", func() {
		tests := []struct {
			name string
			args args
			want int
		}{
			{
				args: args{
					dataArray: []*schemapb.SearchResultData{
						{
							Ids: &schemapb.IDs{
								IdField: &schemapb.IDs_IntId{
									IntId: &schemapb.LongArray{
										Data: []int64{11, 9, 7, 5, 3, 1},
									},
								},
							},
							Scores: []float32{-math.MaxFloat32, -math.MaxFloat32, -math.MaxFloat32, -math.MaxFloat32, -math.MaxFloat32, -math.MaxFloat32},
							Topks:  []int64{2, 2, 2},
						},
						{
							Ids: &schemapb.IDs{
								IdField: &schemapb.IDs_IntId{
									IntId: &schemapb.LongArray{
										Data: []int64{12, 10, 8, 6, 4, 2},
									},
								},
							},
							Scores: []float32{-math.MaxFloat32, -math.MaxFloat32, -math.MaxFloat32, -math.MaxFloat32, -math.MaxFloat32, -math.MaxFloat32},
							Topks:  []int64{2, 2, 2},
						},
					},
					resultOffsets: [][]int64{{0, 2, 4}, {0, 2, 4}},
					offsets:       []int64{0, 1},
					topk:          2,
					nq:            3,
					qi:            0,
				},
				want: -1,
			},
		}
		for _, tt := range tests {
			suite.Run(tt.name, func() {
				if got := SelectSearchResultData(tt.args.dataArray, tt.args.resultOffsets, tt.args.offsets, tt.args.qi); got != tt.want {
					suite.T().Errorf("SelectSearchResultData() = %v, want %v", got, tt.want)
				}
			})
		}
	})
}

func TestReduce(t *testing.T) {
	suite.Run(t, new(ReduceSuite))
}
