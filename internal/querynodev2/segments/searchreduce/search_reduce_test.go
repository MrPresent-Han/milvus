package segments

import (
	"context"
	"fmt"
	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/internal/proto/datapb"
	"github.com/milvus-io/milvus/internal/proto/querypb"
	"github.com/milvus-io/milvus/internal/querynodev2/segments"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/internal/util/initcore"
	"github.com/milvus-io/milvus/pkg/util/paramtable"
	"github.com/stretchr/testify/suite"
	"testing"
)

type SearchReduceSuite struct {
	suite.Suite
	chunkManager storage.ChunkManager
	rootPath     string

	collectionID int64
	partitionID  int64
	segmentID    int64
	collection   *segments.Collection
	segment      segments.Segment
}

func (suite *SearchReduceSuite) SetupSuite() {
	paramtable.Init()
}

func (suite *SearchReduceSuite) SetupTest() {
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

func (suite *SearchReduceSuite) TearDownTest() {
	suite.segment.Release(context.Background())
	segments.DeleteCollection(suite.collection)
	ctx := context.Background()
	suite.chunkManager.RemoveWithPrefix(ctx, suite.rootPath)
}

func (suite *SearchReduceSuite) TestResult_ReduceSearchResultData() {
	const (
		nq         = 1
		topk       = 4
		metricType = "L2"
	)
	reduceInfo := NewReduceInfo(nq, topk, 0, metricType)
	suite.Run("case1", func() {
		ids := []int64{1, 2, 3, 4}
		scores := []float32{-1.0, -2.0, -3.0, -4.0}
		topks := []int64{int64(len(ids))}
		data1 := segments.GenSearchResultData(nq, topk, ids, scores, topks)
		data2 := segments.GenSearchResultData(nq, topk, ids, scores, topks)
		dataArray := make([]*schemapb.SearchResultData, 0)
		dataArray = append(dataArray, data1)
		dataArray = append(dataArray, data2)
		searchReduce := InitSearchReducer(reduceInfo)
		res, err := searchReduce.ReduceSearchResultData(context.TODO(), dataArray, reduceInfo)
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
		searchReduce := InitSearchReducer(reduceInfo)
		res, err := searchReduce.ReduceSearchResultData(context.TODO(), dataArray, reduceInfo)
		suite.Nil(err)
		suite.ElementsMatch([]int64{1, 5, 2, 3}, res.Ids.GetIntId().Data)
	})
}

func (suite *SearchReduceSuite) TestResult_SearchGroupByResult() {
	const (
		nq   = 1
		topk = 4
	)
	reduceInfo := NewReduceInfo(nq, topk, 0, "")
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
		searchReduce := InitSearchReducer(reduceInfo)
		res, err := searchReduce.ReduceSearchResultData(context.TODO(), dataArray, reduceInfo)
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
		searchReduce := InitSearchReducer(reduceInfo)
		res, err := searchReduce.ReduceSearchResultData(context.TODO(), dataArray, reduceInfo)
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
		searchReduce := InitSearchReducer(reduceInfo)
		res, err := searchReduce.ReduceSearchResultData(context.TODO(), dataArray, reduceInfo)
		suite.Nil(err)
		suite.ElementsMatch([]int64{1, 2, 3, 4}, res.Ids.GetIntId().Data)
		suite.ElementsMatch([]float32{-1.0, -2.0, -3.0, -4.0}, res.Scores)
		suite.ElementsMatch([]string{"1", "2", "3", "4"}, res.GroupByFieldValue.GetScalars().GetStringData().Data)
	})
}

func TestSearchReduce(t *testing.T) {
	suite.Run(t, new(SearchReduceSuite))
}
