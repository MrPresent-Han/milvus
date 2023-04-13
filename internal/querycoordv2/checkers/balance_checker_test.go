package checkers

import (
	"testing"

	etcdkv "github.com/milvus-io/milvus/internal/kv/etcd"
	"github.com/milvus-io/milvus/internal/proto/querypb"
	"github.com/milvus-io/milvus/internal/querycoordv2/balance"
	"github.com/milvus-io/milvus/internal/querycoordv2/meta"
	. "github.com/milvus-io/milvus/internal/querycoordv2/params"
	"github.com/milvus-io/milvus/internal/querycoordv2/session"
	"github.com/milvus-io/milvus/internal/querycoordv2/task"
	"github.com/milvus-io/milvus/internal/querycoordv2/utils"
	"github.com/milvus-io/milvus/internal/util/etcd"

	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/suite"
)

type BalanceCheckerTestSuite struct {
	suite.Suite
	kv        *etcdkv.EtcdKV
	checker   *BalanceChecker
	meta      *meta.Meta
	broker    *meta.MockBroker
	nodeMgr   *session.NodeManager
	scheduler *task.MockScheduler
}

func (suite *BalanceCheckerTestSuite) SetupSuite() {
	Params.InitOnce()
}

func (suite *BalanceCheckerTestSuite) SetupTest() {
	var err error
	config := GenerateEtcdConfig()
	cli, err := etcd.GetEtcdClient(
		config.UseEmbedEtcd,
		config.EtcdUseSSL,
		config.Endpoints,
		config.EtcdTLSCert,
		config.EtcdTLSKey,
		config.EtcdTLSCACert,
		config.EtcdTLSMinVersion)
	suite.Require().NoError(err)
	suite.kv = etcdkv.NewEtcdKV(cli, config.MetaRootPath)

	// meta
	store := meta.NewMetaStore(suite.kv)
	idAllocator := RandomIncrementIDAllocator()
	suite.nodeMgr = session.NewNodeManager()
	suite.meta = meta.NewMeta(idAllocator, store, suite.nodeMgr)
	suite.broker = meta.NewMockBroker(suite.T())
	suite.scheduler = task.NewMockScheduler(suite.T())

	balancer := suite.createMockBalancer()
	suite.checker = NewBalanceChecker(suite.meta, balancer, suite.nodeMgr, suite.scheduler)
}

func (suite *BalanceCheckerTestSuite) createMockBalancer() balance.Balance {
	balancer := balance.NewMockBalancer(suite.T())
	balancer.EXPECT().BalanceReplica(mock.Anything).Maybe().Return(
		func(replica *meta.Replica) ([]balance.SegmentAssignPlan, []balance.ChannelAssignPlan) {
			segPlans, chanPlans := make([]balance.SegmentAssignPlan, 0), make([]balance.ChannelAssignPlan, 0)
			segment := utils.CreateTestSegment(1, 1, 1, 1, 1, "chan")
			mockSegPlan := balance.SegmentAssignPlan{
				Segment:   segment,
				From:      -1,
				To:        -1,
				ReplicaID: -1,
			}
			segPlans = append(segPlans, mockSegPlan)
			return segPlans, chanPlans
		})
	return balancer
}

func (suite *BalanceCheckerTestSuite) TearDownTest() {
	suite.kv.Close()
}

func (suite *BalanceCheckerTestSuite) TestAutoBalanceConf() {
	//set up nodes info
	nodeID1, nodeID2 := 1, 2
	suite.nodeMgr.Add(session.NewNodeInfo(int64(nodeID1), "localhost"))
	suite.nodeMgr.Add(session.NewNodeInfo(int64(nodeID2), "localhost"))
	suite.checker.meta.ResourceManager.AssignNode(meta.DefaultResourceGroupName, int64(nodeID1))
	suite.checker.meta.ResourceManager.AssignNode(meta.DefaultResourceGroupName, int64(nodeID2))

	// set collections meta
	cid1, replicaId1 := 1, 1
	collection1 := utils.CreateTestCollection(int64(cid1), int32(replicaId1))
	collection1.Status = querypb.LoadStatus_Loaded
	replica1 := utils.CreateTestReplica(int64(replicaId1), int64(cid1), []int64{int64(nodeID1), int64(nodeID2)})
	suite.checker.meta.CollectionManager.PutCollection(collection1)
	suite.checker.meta.ReplicaManager.Put(replica1)

	cid2, replicaId2 := 2, 2
	collection2 := utils.CreateTestCollection(int64(cid2), int32(replicaId2))
	collection2.Status = querypb.LoadStatus_Loaded
	replica2 := utils.CreateTestReplica(int64(replicaId2), int64(cid2), []int64{int64(nodeID1), int64(nodeID2)})
	suite.checker.meta.CollectionManager.PutCollection(collection2)
	suite.checker.meta.ReplicaManager.Put(replica2)

	//test disable auto balance
	Params.QueryCoordCfg.AutoBalance = false
	suite.scheduler.EXPECT().GetSegmentTaskNum().Maybe().Return(func() int {
		return 0
	})
	collectionToBalance := suite.checker.collectionsToBalance()
	suite.Empty(collectionToBalance)
	segPlans, _ := suite.checker.balanceCollections(collectionToBalance)
	suite.Empty(segPlans)

	//test enable auto balance
	Params.QueryCoordCfg.AutoBalance = true
	idsToBalance := []int64{int64(nodeID1)}
	collectionToBalance = suite.checker.collectionsToBalance()
	suite.ElementsMatch(idsToBalance, collectionToBalance)
	//next round
	idsToBalance = []int64{int64(nodeID2)}
	collectionToBalance = suite.checker.collectionsToBalance()
	suite.ElementsMatch(idsToBalance, collectionToBalance)
	//final round
	collectionToBalance = suite.checker.collectionsToBalance()
	suite.Empty(collectionToBalance)
}

func (suite *BalanceCheckerTestSuite) TestBusyScheduler() {
	//set up nodes info
	nodeID1, nodeID2 := 1, 2
	suite.nodeMgr.Add(session.NewNodeInfo(int64(nodeID1), "localhost"))
	suite.nodeMgr.Add(session.NewNodeInfo(int64(nodeID2), "localhost"))
	suite.checker.meta.ResourceManager.AssignNode(meta.DefaultResourceGroupName, int64(nodeID1))
	suite.checker.meta.ResourceManager.AssignNode(meta.DefaultResourceGroupName, int64(nodeID2))

	// set collections meta
	cid1, replicaId1 := 1, 1
	collection1 := utils.CreateTestCollection(int64(cid1), int32(replicaId1))
	collection1.Status = querypb.LoadStatus_Loaded
	replica1 := utils.CreateTestReplica(int64(replicaId1), int64(cid1), []int64{int64(nodeID1), int64(nodeID2)})
	suite.checker.meta.CollectionManager.PutCollection(collection1)
	suite.checker.meta.ReplicaManager.Put(replica1)

	cid2, replicaId2 := 2, 2
	collection2 := utils.CreateTestCollection(int64(cid2), int32(replicaId2))
	collection2.Status = querypb.LoadStatus_Loaded
	replica2 := utils.CreateTestReplica(int64(replicaId2), int64(cid2), []int64{int64(nodeID1), int64(nodeID2)})
	suite.checker.meta.CollectionManager.PutCollection(collection2)
	suite.checker.meta.ReplicaManager.Put(replica2)

	//test scheduler busy
	Params.QueryCoordCfg.AutoBalance = true
	suite.scheduler.EXPECT().GetSegmentTaskNum().Maybe().Return(func() int {
		return 1
	})
	collectionToBalance := suite.checker.collectionsToBalance()
	suite.Empty(collectionToBalance)
	segPlans, _ := suite.checker.balanceCollections(collectionToBalance)
	suite.Empty(segPlans)
}

func (suite *BalanceCheckerTestSuite) TestStoppingBalance() {
	//set up nodes info, stopping node1
	nodeID1, nodeID2 := 1, 2
	suite.nodeMgr.Add(session.NewNodeInfo(int64(nodeID1), "localhost"))
	suite.nodeMgr.Add(session.NewNodeInfo(int64(nodeID2), "localhost"))
	suite.nodeMgr.Stopping(int64(nodeID1))
	suite.checker.meta.ResourceManager.AssignNode(meta.DefaultResourceGroupName, int64(nodeID1))
	suite.checker.meta.ResourceManager.AssignNode(meta.DefaultResourceGroupName, int64(nodeID2))

	// set collections meta
	cid1, replicaId1 := 1, 1
	collection1 := utils.CreateTestCollection(int64(cid1), int32(replicaId1))
	collection1.Status = querypb.LoadStatus_Loaded
	replica1 := utils.CreateTestReplica(int64(replicaId1), int64(cid1), []int64{int64(nodeID1), int64(nodeID2)})
	suite.checker.meta.CollectionManager.PutCollection(collection1)
	suite.checker.meta.ReplicaManager.Put(replica1)

	cid2, replicaId2 := 2, 2
	collection2 := utils.CreateTestCollection(int64(cid2), int32(replicaId2))
	collection2.Status = querypb.LoadStatus_Loaded
	replica2 := utils.CreateTestReplica(int64(replicaId2), int64(cid2), []int64{int64(nodeID1), int64(nodeID2)})
	suite.checker.meta.CollectionManager.PutCollection(collection2)
	suite.checker.meta.ReplicaManager.Put(replica2)

	//test scheduler busy
	idsToBalance := []int64{int64(nodeID1), int64(nodeID2)}
	collectionToBalance := suite.checker.collectionsToBalance()
	suite.ElementsMatch(idsToBalance, collectionToBalance)
}

func TestBalanceCheckerSuite(t *testing.T) {
	suite.Run(t, new(BalanceCheckerTestSuite))
}
