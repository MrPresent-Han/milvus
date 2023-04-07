package integration

import (
	"context"
	"errors"
	"github.com/milvus-io/milvus-proto/go-api/msgpb"
	"github.com/milvus-io/milvus/internal/proto/querypb"
	"github.com/milvus-io/milvus/pkg/common"
	"github.com/milvus-io/milvus/pkg/util/commonpbutil"
	"github.com/milvus-io/milvus/pkg/util/distance"
	"go.uber.org/zap"
	"strconv"
	"testing"
	"time"

	"github.com/golang/protobuf/proto"
	"github.com/milvus-io/milvus-proto/go-api/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/milvuspb"
	"github.com/milvus-io/milvus-proto/go-api/schemapb"
	"github.com/milvus-io/milvus/pkg/log"

	"github.com/stretchr/testify/assert"
)

func constructBasicITSchema(int64Field string, floatVecField string, dim int, collectionName string) *schemapb.CollectionSchema {
	pk := &schemapb.FieldSchema{
		FieldID:      0,
		Name:         int64Field,
		IsPrimaryKey: true,
		Description:  "",
		DataType:     schemapb.DataType_Int64,
		TypeParams:   nil,
		IndexParams:  nil,
		AutoID:       true,
	}
	fVec := &schemapb.FieldSchema{
		FieldID:      0,
		Name:         floatVecField,
		IsPrimaryKey: false,
		Description:  "",
		DataType:     schemapb.DataType_FloatVector,
		TypeParams: []*commonpb.KeyValuePair{
			{
				Key:   "dim",
				Value: strconv.Itoa(dim),
			},
		},
		IndexParams: nil,
		AutoID:      false,
	}
	return &schemapb.CollectionSchema{
		Name:        collectionName,
		Description: "",
		AutoID:      false,
		Fields: []*schemapb.FieldSchema{
			pk,
			fVec,
		},
	}
}

func createCollection(t *testing.T, ctx context.Context, cluster *MiniCluster,
	dbName string, collectionName string, schema *schemapb.CollectionSchema, shardNum int32) {
	marshaledSchema, err := proto.Marshal(schema)
	assert.NoError(t, err)

	createCollectionStatus, err := cluster.proxy.CreateCollection(ctx, &milvuspb.CreateCollectionRequest{
		DbName:         dbName,
		CollectionName: collectionName,
		Schema:         marshaledSchema,
		ShardsNum:      shardNum,
	})
	assert.NoError(t, err)
	if createCollectionStatus.GetErrorCode() != commonpb.ErrorCode_Success {
		log.Warn("createCollectionStatus fail reason", zap.String("reason", createCollectionStatus.GetReason()))
	}
	assert.Equal(t, createCollectionStatus.GetErrorCode(), commonpb.ErrorCode_Success)

	log.Info("CreateCollection result", zap.Any("createCollectionStatus", createCollectionStatus))
}

func showCollections(t *testing.T, cluster *MiniCluster, ctx context.Context) {
	showCollectionsResp, err := cluster.proxy.ShowCollections(ctx, &milvuspb.ShowCollectionsRequest{})
	assert.NoError(t, err)
	assert.Equal(t, showCollectionsResp.GetStatus().GetErrorCode(), commonpb.ErrorCode_Success)
	log.Info("ShowCollections result", zap.Any("showCollectionsResp", showCollectionsResp))
}

func insertData(t *testing.T, cluster *MiniCluster, ctx context.Context, dbName string,
	collectionName string, rowNum uint32,
	data ...*schemapb.FieldData) {
	insertResult, err := cluster.proxy.Insert(ctx, &milvuspb.InsertRequest{
		DbName:         dbName,
		CollectionName: collectionName,
		FieldsData:     data,
		HashKeys:       []uint32{},
		NumRows:        rowNum,
	})
	assert.NoError(t, err)
	assert.Equal(t, insertResult.GetStatus().GetErrorCode(), commonpb.ErrorCode_Success)
}

func flushAndWaitSegmentFlushed(t *testing.T, cluster *MiniCluster, ctx context.Context, dbName string, collectionName string) {
	flushResp, err := cluster.proxy.Flush(ctx, &milvuspb.FlushRequest{
		DbName:          dbName,
		CollectionNames: []string{collectionName},
	})
	assert.NoError(t, err)
	segmentIDs, has := flushResp.GetCollSegIDs()[collectionName]
	ids := segmentIDs.GetData()
	assert.NotEmpty(t, segmentIDs)

	segments, err := cluster.metaWatcher.ShowSegments()
	assert.NoError(t, err)
	assert.NotEmpty(t, segments)
	for _, segment := range segments {
		log.Info("ShowSegments result", zap.String("segment", segment.String()))
	}

	if has && len(ids) > 0 {
		flushed := func() bool {
			resp, err := cluster.proxy.GetFlushState(ctx, &milvuspb.GetFlushStateRequest{
				SegmentIDs: ids,
			})
			if err != nil {
				//panic(errors.New("GetFlushState failed"))
				return false
			}
			return resp.GetFlushed()
		}
		for !flushed() {
			// respect context deadline/cancel
			select {
			case <-ctx.Done():
				panic(errors.New("deadline exceeded"))
			default:
			}
			time.Sleep(500 * time.Millisecond)
		}
	}
}

func createIndex(t *testing.T, cluster *MiniCluster, ctx context.Context, collectionName string,
	floatVecField string, dim int, indexType string) {
	// create index
	createIndexStatus, err := cluster.proxy.CreateIndex(ctx, &milvuspb.CreateIndexRequest{
		CollectionName: collectionName,
		FieldName:      floatVecField,
		IndexName:      "_default",
		ExtraParams: []*commonpb.KeyValuePair{
			{
				Key:   "dim",
				Value: strconv.Itoa(dim),
			},
			{
				Key:   common.MetricTypeKey,
				Value: distance.L2,
			},
			{
				Key:   "index_type",
				Value: indexType,
			},
			{
				Key:   "nlist",
				Value: strconv.Itoa(10),
			},
		},
	})
	if createIndexStatus.GetErrorCode() != commonpb.ErrorCode_Success {
		log.Warn("createIndexStatus fail reason", zap.String("reason", createIndexStatus.GetReason()))
	}
	assert.NoError(t, err)
	assert.Equal(t, commonpb.ErrorCode_Success, createIndexStatus.GetErrorCode())
}

func loadIndex(t *testing.T, cluster *MiniCluster, ctx context.Context, dbName string,
	collectionName string, sync bool) {
	// load
	loadStatus, err := cluster.proxy.LoadCollection(ctx, &milvuspb.LoadCollectionRequest{
		DbName:         dbName,
		CollectionName: collectionName,
	})
	assert.NoError(t, err)
	if loadStatus.GetErrorCode() != commonpb.ErrorCode_Success {
		log.Warn("loadStatus fail reason", zap.String("reason", loadStatus.GetReason()))
	}
	assert.Equal(t, commonpb.ErrorCode_Success, loadStatus.GetErrorCode())
	if sync {
		for {
			loadProgress, err := cluster.proxy.GetLoadingProgress(ctx, &milvuspb.GetLoadingProgressRequest{
				CollectionName: collectionName,
			})
			if err != nil {
				panic("GetLoadingProgress fail")
			}
			if loadProgress.GetProgress() == 100 {
				break
			}
			time.Sleep(500 * time.Millisecond)
		}
	}
}

type BalanceSegmentInfo struct {
	collectionID int64
	segmentID    int64
}

func getSegmentsDistOnQueryNodes(t *testing.T, cluster *MiniCluster, ctx context.Context) map[int64][]*BalanceSegmentInfo {
	dist := make(map[int64][]*BalanceSegmentInfo, 0)
	for _, qnode := range cluster.queryNodes {
		resp, err := qnode.GetDataDistribution(ctx, &querypb.GetDataDistributionRequest{
			Base: commonpbutil.NewMsgBase(
				commonpbutil.WithMsgType(commonpb.MsgType_GetDistribution),
			),
			Checkpoints: make(map[string]*msgpb.MsgPosition),
		})
		assert.NoError(t, err)
		dist[resp.GetNodeID()] = make([]*BalanceSegmentInfo, 0, len(resp.GetSegments()))
		for _, segInfo := range resp.GetSegments() {
			dist[resp.GetNodeID()] = append(dist[resp.GetNodeID()], &BalanceSegmentInfo{
				collectionID: segInfo.GetCollection(),
				segmentID:    segInfo.GetID(),
			})
		}
	}
	return dist
}
