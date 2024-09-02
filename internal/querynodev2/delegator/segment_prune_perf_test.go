package delegator

import (
	"context"
	"fmt"
	"path"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/bits-and-blooms/bitset"
	"github.com/golang/protobuf/proto"
	"github.com/stretchr/testify/suite"
	"go.uber.org/zap"

	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/internal/parser/planparserv2"
	"github.com/milvus-io/milvus/internal/proto/internalpb"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/internal/util/testutil"
	"github.com/milvus-io/milvus/pkg/common"
	"github.com/milvus-io/milvus/pkg/log"
	"github.com/milvus-io/milvus/pkg/util/metautil"
	"github.com/milvus-io/milvus/pkg/util/paramtable"
	"github.com/milvus-io/milvus/pkg/util/typeutil"
)

type SegmentPrunerPerfSuite struct {
	suite.Suite
	partitionStats      map[UniqueID]*storage.PartitionStatsSnapshot
	schema              *schemapb.CollectionSchema
	collectionName      string
	primaryFieldName    string
	clusterKeyFieldName string
	autoID              bool
	targetPartition     int64
	dim                 int
	sealedSegments      []SnapshotItem
	chunkManager        storage.ChunkManager

	// for part stats
	collectionID     int64
	partID           int64
	vChannelName     string
	partStatsVersion int64

	// for distribute test
	nodeNum int
}

func (sps *SegmentPrunerPerfSuite) SetupTest() {
	paramtable.Init()

	sps.collectionName = "test_segment_prune"
	sps.primaryFieldName = "pk"
	sps.clusterKeyFieldName = "c_key"
	sps.autoID = false
	sps.dim = 128
	sps.collectionID = 450745738725425527
	sps.partID = 450745738725425528
	sps.vChannelName = "by-dev-rootcoord-dml_0_450745738725425527v0"
	sps.partStatsVersion = 450745738736232912
	sps.nodeNum = 4

	// set up schema
	clusterKeyFieldType := schemapb.DataType_Int64
	fieldName2DataType := make(map[string]schemapb.DataType)
	fieldName2DataType[sps.primaryFieldName] = schemapb.DataType_Int64
	fieldName2DataType[sps.clusterKeyFieldName] = clusterKeyFieldType
	fieldName2DataType["float"] = schemapb.DataType_Float
	fieldName2DataType["float_vector"] = schemapb.DataType_FloatVector

	sps.schema = testutil.ConstructCollectionSchemaWithKeys(sps.collectionName,
		fieldName2DataType,
		sps.primaryFieldName,
		"",
		sps.clusterKeyFieldName,
		sps.autoID,
		sps.dim)

	for _, fieldSchema := range sps.schema.GetFields() {
		if fieldSchema.GetName() == sps.clusterKeyFieldName {
			fieldSchema.FieldID = 101
			break
		}
	}

	// init chunk manager
	rootPath := "files"
	chunkManagerFactory := storage.NewTestChunkManagerFactory(paramtable.Get(), rootPath)
	sps.chunkManager, _ = chunkManagerFactory.NewPersistentStorageChunkManager(context.Background())

	// init partitionStats
	sps.partitionStats = make(map[UniqueID]*storage.PartitionStatsSnapshot)
	idPath := metautil.JoinIDPath(sps.collectionID, sps.partID)
	idPath = path.Join(idPath, sps.vChannelName)
	statsFilePath := path.Join(rootPath, common.PartitionStatsPath, idPath, strconv.FormatInt(sps.partStatsVersion, 10))
	statsBytes, _ := sps.chunkManager.Read(context.Background(), statsFilePath)
	partStats, _ := storage.DeserializePartitionsStatsSnapshot(statsBytes)
	sps.partitionStats[sps.partID] = partStats

	// init sealed segments
	segmentPathPrefix := "files/insert_log/450745738725425527/450745738725425528/"
	segmentIDLists := make([]UniqueID, 0)
	sps.chunkManager.WalkWithPrefix(context.Background(), segmentPathPrefix, false, func(chunkObjectInfo *storage.ChunkObjectInfo) bool {
		fileParts := strings.Split(chunkObjectInfo.FilePath, "/")
		segmentID, _ := strconv.ParseInt(fileParts[len(fileParts)-2], 10, 64)
		_, exist := partStats.SegmentStats[segmentID]
		if exist {
			segmentIDLists = append(segmentIDLists, segmentID)
		}
		return true
	})

	sealedSegments := make([]SnapshotItem, sps.nodeNum)
	for i, segmentID := range segmentIDLists {
		nodeIdx := i % sps.nodeNum
		if sealedSegments[nodeIdx].Segments == nil {
			sealedSegments[nodeIdx].NodeID = int64(nodeIdx)
			sealedSegments[nodeIdx].Segments = make([]SegmentEntry, 0)
		}
		sealedSegments[nodeIdx].Segments = append(sealedSegments[nodeIdx].Segments,
			SegmentEntry{NodeID: int64(nodeIdx), SegmentID: segmentID, PartitionID: sps.partID})
	}
	sps.sealedSegments = sealedSegments
}

func (sps *SegmentPrunerPerfSuite) TestPruneSegmentsByScalarIntField() {
	paramtable.Init()
	targetPartitions := make([]UniqueID, 0)
	targetPartitions = append(targetPartitions, sps.targetPartition)
	schemaHelper, _ := typeutil.CreateSchemaHelper(sps.schema)
	exprs := []string{
		"c_key>200 and c_key<300",
		/*"c_key>300",
		"c_key<600",
		"c_key>=400",
		"c_key<=700",
		"c_key<600 and c_key<=1200",
		"c_key>=600 and c_key<1200",*/
	}
	exprCount := len(exprs)
	reqs := make([]*internalpb.RetrieveRequest, 0)
	for i := 0; i < exprCount; i++ {
		exprStr := exprs[i%exprCount]
		planNode, _ := planparserv2.CreateRetrievePlan(schemaHelper, exprStr)
		serializedPlan, _ := proto.Marshal(planNode)
		queryReq := &internalpb.RetrieveRequest{
			SerializedExprPlan: serializedPlan,
			PartitionIDs:       targetPartitions,
		}
		reqs = append(reqs, queryReq)
	}
	testCount := 20000000
	testSegments := make([]SnapshotItem, len(sps.sealedSegments))
	var pruneCost time.Duration
	for i := 0; i < testCount; i++ {
		copy(testSegments, sps.sealedSegments)
		start := time.Now()
		PruneSegments(context.TODO(), sps.partitionStats, nil, reqs[i%exprCount], sps.schema, testSegments, PruneInfo{paramtable.Get().QueryNodeCfg.DefaultSegmentFilterRatio.GetAsFloat()})
		duration := time.Since(start)
		pruneCost += duration
		log.Info("Pruned done", zap.Duration("single duration", duration))
	}
	log.Info("Pruned Finished", zap.Duration("total duration", pruneCost))
}

func (sps *SegmentPrunerPerfSuite) TestBitsetEffect() {
	bs1 := bitset.New(10)
	bs2 := bitset.New(10)

	bs1.Set(5)
	bs1.Set(6)

	bs2.Set(6)
	bs2.Set(8)
	bs1.InPlaceUnion(bs2)
	fmt.Println(fmt.Sprintf("count:%d", bs1.Count()))
}

func (sps *SegmentPrunerPerfSuite) TestPrintPartStats() {
	// prepare stats
	rootPath := "test"
	chunkManagerFactory := storage.NewTestChunkManagerFactory(paramtable.Get(), rootPath)
	sps.chunkManager, _ = chunkManagerFactory.NewPersistentStorageChunkManager(context.Background())

	statsFilePath := "test/451294387301046833"
	statsBytes, _ := sps.chunkManager.Read(context.Background(), statsFilePath)
	partStats, _ := storage.DeserializePartitionsStatsSnapshot(statsBytes)
	log.Info("statsCount", zap.Int("count", len(partStats.SegmentStats)))
	for segID, stats := range partStats.SegmentStats {
		fieldStat := stats.FieldStats[0]
		log.Info("statsInfo",
			zap.Int64("segID", segID),
			zap.Any("rows", stats.NumRows),
			zap.Any("min", fieldStat.Min),
			zap.Any("max", fieldStat.Max))
	}
	partStatsMap := make(map[UniqueID]*storage.PartitionStatsSnapshot, 0)
	partStatsMap[451294387290071625] = partStats

	// prepare schema
	fieldName2DataType := make(map[string]schemapb.DataType)
	primaryFieldName := "pk"
	clusterKeyFieldName := "key"
	fieldName2DataType[primaryFieldName] = schemapb.DataType_Int64
	fieldName2DataType["key"] = schemapb.DataType_VarChar
	fieldName2DataType["embedding"] = schemapb.DataType_FloatVector
	collectionName := "major_compaction_collection_enable_scalar_clustering_key_wiki_10M_fb"

	schema := testutil.ConstructCollectionSchemaWithKeys(collectionName,
		fieldName2DataType,
		primaryFieldName,
		"",
		clusterKeyFieldName,
		sps.autoID,
		sps.dim)

	schemaHelper, _ := typeutil.CreateSchemaHelper(schema)
	exprStr := "key=='f8d5a851-0fc1-4f84-808c-450721a-5'"
	planNode, _ := planparserv2.CreateRetrievePlan(schemaHelper, exprStr)
	serializedPlan, _ := proto.Marshal(planNode)
	searchReq := &internalpb.SearchRequest{
		SerializedExprPlan: serializedPlan,
	}
	PruneSegments(context.TODO(), partStatsMap, searchReq, nil, schema, nil, PruneInfo{})
}

func TestSegmentPrunerPerfSuite(t *testing.T) {
	suite.Run(t, new(SegmentPrunerPerfSuite))
}
