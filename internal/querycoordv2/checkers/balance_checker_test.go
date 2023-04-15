package checkers

import (
	etcdkv "github.com/milvus-io/milvus/internal/kv/etcd"
	"github.com/milvus-io/milvus/internal/querycoordv2/balance"
	"github.com/milvus-io/milvus/internal/querycoordv2/meta"
	. "github.com/milvus-io/milvus/internal/querycoordv2/params"
	"github.com/milvus-io/milvus/internal/querycoordv2/session"
	"github.com/milvus-io/milvus/internal/querycoordv2/task"
	"github.com/milvus-io/milvus/internal/util/etcd"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/suite"
)

type BalanceCheckerTestSuite struct {
	suite.Suite
	kv      *etcdkv.EtcdKV
	checker *BalanceChecker
	meta    *meta.Meta
	broker  *meta.MockBroker
	nodeMgr *session.NodeManager
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

	scheduler := task.NewMockScheduler(suite.T())
	balancer := suite.createMockBalancer()
	suite.checker = NewBalanceChecker(suite.meta, balancer, suite.nodeMgr, scheduler)
}

func (suite *BalanceCheckerTestSuite) createMockBalancer() balance.Balance {
	balancer := balance.NewMockBalancer(suite.T())
	balancer.EXPECT().AssignSegment(mock.Anything, mock.Anything, mock.Anything).
		Maybe().Return(func(collectionID int64, segments []*meta.Segment, nodes []int64) []balance.SegmentAssignPlan {
		plans := make([]balance.SegmentAssignPlan, 0, len(segments))
		for i, s := range segments {
			plan := balance.SegmentAssignPlan{
				Segment:   s,
				From:      -1,
				To:        nodes[i%len(nodes)],
				ReplicaID: -1,
			}
			plans = append(plans, plan)
		}
		return plans
	})
	return balancer
}

func (suite *BalanceCheckerTestSuite) TearDownTest() {
	suite.kv.Close()
}
