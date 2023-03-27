// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//	http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package balance

import (
	etcdkv "github.com/milvus-io/milvus/internal/kv/etcd"
	"github.com/milvus-io/milvus/internal/proto/datapb"
	"github.com/milvus-io/milvus/internal/querycoordv2/meta"
	. "github.com/milvus-io/milvus/internal/querycoordv2/params"
	"github.com/milvus-io/milvus/internal/querycoordv2/session"
	"github.com/milvus-io/milvus/internal/querycoordv2/task"
	"github.com/milvus-io/milvus/internal/util/etcd"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/suite"
	"testing"
)

type ScoreBasedBalancerTestSuite struct {
	suite.Suite
	balancer      *ScoreBasedBalancer
	kv            *etcdkv.EtcdKV
	broker        *meta.MockBroker
	mockScheduler *task.MockScheduler
}

func (suite *ScoreBasedBalancerTestSuite) SetupSuite() {
	Params.Init()
}

func (suite *ScoreBasedBalancerTestSuite) SetupTest() {
	var err error
	config := GenerateEtcdConfig()
	cli, err := etcd.GetEtcdClient(
		config.UseEmbedEtcd.GetAsBool(),
		config.EtcdUseSSL.GetAsBool(),
		config.Endpoints.GetAsStrings(),
		config.EtcdTLSCert.GetValue(),
		config.EtcdTLSKey.GetValue(),
		config.EtcdTLSCACert.GetValue(),
		config.EtcdTLSMinVersion.GetValue())
	suite.Require().NoError(err)
	suite.kv = etcdkv.NewEtcdKV(cli, config.MetaRootPath.GetValue())
	suite.broker = meta.NewMockBroker(suite.T())

	store := meta.NewMetaStore(suite.kv)
	idAllocator := RandomIncrementIDAllocator()
	nodeManager := session.NewNodeManager()
	testMeta := meta.NewMeta(idAllocator, store, nodeManager)
	testTarget := meta.NewTargetManager(suite.broker, testMeta)

	distManager := meta.NewDistributionManager()
	suite.mockScheduler = task.NewMockScheduler(suite.T())
	suite.balancer = NewScoreBasedBalancer(suite.mockScheduler, nodeManager, distManager, testMeta, testTarget)

	suite.broker.EXPECT().GetPartitions(mock.Anything, int64(1)).Return([]int64{1}, nil).Maybe()
}

func (suite *ScoreBasedBalancerTestSuite) TearDownTest() {
	suite.kv.Close()
}

func (suite *ScoreBasedBalancerTestSuite) TestAssignSegment() {
	cases := []struct {
		name          string
		comment       string
		distributions map[int64][]*meta.Segment
		assignments   [][]*meta.Segment
		nodes         []int64
		collectionIDs []int64
		segmentCnts   []int
		states        []session.State
		expectPlans   [][]SegmentAssignPlan
	}{
		{
			name:          "test empty cluster assigning one collection",
			comment:       "this is most simple case in which global row count is zero for all nodes",
			distributions: map[int64][]*meta.Segment{},
			assignments: [][]*meta.Segment{
				{
					{SegmentInfo: &datapb.SegmentInfo{ID: 1, NumOfRows: 5, CollectionID: 1}},
					{SegmentInfo: &datapb.SegmentInfo{ID: 2, NumOfRows: 10, CollectionID: 1}},
					{SegmentInfo: &datapb.SegmentInfo{ID: 3, NumOfRows: 15, CollectionID: 1}},
				},
			},
			nodes:         []int64{1, 2, 3},
			collectionIDs: []int64{0},
			states:        []session.State{session.NodeStateNormal, session.NodeStateNormal, session.NodeStateNormal},
			segmentCnts:   []int{0, 0, 0},
			expectPlans: [][]SegmentAssignPlan{
				{
					//as assign segments is used while loading collection,
					//all assignPlan should have weight equal to 1(HIGH PRIORITY)
					{Segment: &meta.Segment{SegmentInfo: &datapb.SegmentInfo{ID: 3, NumOfRows: 15,
						CollectionID: 1}}, From: -1, To: 1, Weight: 1},
					{Segment: &meta.Segment{SegmentInfo: &datapb.SegmentInfo{ID: 2, NumOfRows: 10,
						CollectionID: 1}}, From: -1, To: 3, Weight: 1},
					{Segment: &meta.Segment{SegmentInfo: &datapb.SegmentInfo{ID: 1, NumOfRows: 5,
						CollectionID: 1}}, From: -1, To: 2, Weight: 1},
				},
			},
		},
		{
			name: "test non-empty cluster assigning one collection",
			comment: "this case will verify the effect of global row for loading segments process, although node1" +
				"has only 10 rows at the beginning, but it has so many rows on global view, resulting in a lower priority",
			distributions: map[int64][]*meta.Segment{
				1: {
					{SegmentInfo: &datapb.SegmentInfo{ID: 1, NumOfRows: 10, CollectionID: 1}, Node: 1},
					{SegmentInfo: &datapb.SegmentInfo{ID: 2, NumOfRows: 300, CollectionID: 2}, Node: 1},
					//base: collection1-node1-priority is 10 + 0.1 * 310 = 41
					//assign3: collection1-node1-priority is 15 + 0.1 * 315 = 46.5
				},
				2: {
					{SegmentInfo: &datapb.SegmentInfo{ID: 3, NumOfRows: 20, CollectionID: 1}, Node: 2},
					{SegmentInfo: &datapb.SegmentInfo{ID: 4, NumOfRows: 180, CollectionID: 2}, Node: 2},
					//base: collection1-node2-priority is 20 + 0.1 * 200 = 40
					//assign2: collection1-node2-priority is 30 + 0.1 * 210 = 51
				},
				3: {
					{SegmentInfo: &datapb.SegmentInfo{ID: 5, NumOfRows: 30, CollectionID: 1}, Node: 3},
					{SegmentInfo: &datapb.SegmentInfo{ID: 6, NumOfRows: 20, CollectionID: 2}, Node: 3},
					//base: collection1-node2-priority is 30 + 0.1 * 50 = 35
					//assign1: collection1-node2-priority is 45 + 0.1 * 65 = 51.5
				},
			},
			assignments: [][]*meta.Segment{
				{
					{SegmentInfo: &datapb.SegmentInfo{ID: 7, NumOfRows: 5, CollectionID: 1}},
					{SegmentInfo: &datapb.SegmentInfo{ID: 8, NumOfRows: 10, CollectionID: 1}},
					{SegmentInfo: &datapb.SegmentInfo{ID: 9, NumOfRows: 15, CollectionID: 1}},
				},
			},
			nodes:         []int64{1, 2, 3},
			collectionIDs: []int64{1},
			states:        []session.State{session.NodeStateNormal, session.NodeStateNormal, session.NodeStateNormal},
			segmentCnts:   []int{0, 0, 0},
			expectPlans: [][]SegmentAssignPlan{
				{
					{Segment: &meta.Segment{SegmentInfo: &datapb.SegmentInfo{ID: 9, NumOfRows: 15, CollectionID: 1}}, From: -1, To: 3, Weight: 1},
					{Segment: &meta.Segment{SegmentInfo: &datapb.SegmentInfo{ID: 8, NumOfRows: 10, CollectionID: 1}}, From: -1, To: 2, Weight: 1},
					{Segment: &meta.Segment{SegmentInfo: &datapb.SegmentInfo{ID: 7, NumOfRows: 5, CollectionID: 1}}, From: -1, To: 1, Weight: 1},
				},
			},
		},
		{
			name: "test non-empty cluster assigning two collections at one round segment checking",
			comment: "this case is used to demonstrate the existing assign mechanism having flaws when assigning " +
				"multi collections at one round by using the only segment distribution",
			distributions: map[int64][]*meta.Segment{
				1: {
					{SegmentInfo: &datapb.SegmentInfo{ID: 1, NumOfRows: 10, CollectionID: 1}, Node: 1},
				},
				2: {
					{SegmentInfo: &datapb.SegmentInfo{ID: 2, NumOfRows: 20, CollectionID: 1}, Node: 2},
				},
				3: {
					{SegmentInfo: &datapb.SegmentInfo{ID: 3, NumOfRows: 40, CollectionID: 1}, Node: 3},
				},
			},
			assignments: [][]*meta.Segment{
				{
					{SegmentInfo: &datapb.SegmentInfo{ID: 4, NumOfRows: 60, CollectionID: 1}},
					{SegmentInfo: &datapb.SegmentInfo{ID: 5, NumOfRows: 50, CollectionID: 1}},
				},
				{
					{SegmentInfo: &datapb.SegmentInfo{ID: 6, NumOfRows: 15, CollectionID: 2}},
					{SegmentInfo: &datapb.SegmentInfo{ID: 7, NumOfRows: 10, CollectionID: 2}},
				},
			},
			nodes:         []int64{1, 2, 3},
			collectionIDs: []int64{1, 2},
			states:        []session.State{session.NodeStateNormal, session.NodeStateNormal, session.NodeStateNormal},
			segmentCnts:   []int{0, 0, 0},
			expectPlans: [][]SegmentAssignPlan{
				//note that these two segments plans are absolutely unbalanced globally,
				//as if the assignment for collection1 could succeed, node1 and node2 will both have 70 rows
				//much more than node3, but following assignment will still assign segment based on [10,20,40]
				//rather than [70,70,40], this flaw will be mitigated by balance process and maybe fixed in the later versions
				{
					{Segment: &meta.Segment{SegmentInfo: &datapb.SegmentInfo{ID: 4, NumOfRows: 60, CollectionID: 1}}, From: -1, To: 1, Weight: 1},
					{Segment: &meta.Segment{SegmentInfo: &datapb.SegmentInfo{ID: 5, NumOfRows: 50, CollectionID: 1}}, From: -1, To: 2, Weight: 1},
				},
				{
					{Segment: &meta.Segment{SegmentInfo: &datapb.SegmentInfo{ID: 6, NumOfRows: 15, CollectionID: 2}}, From: -1, To: 1, Weight: 1},
					{Segment: &meta.Segment{SegmentInfo: &datapb.SegmentInfo{ID: 7, NumOfRows: 10, CollectionID: 2}}, From: -1, To: 2, Weight: 1},
				},
			},
		},
	}

	for _, c := range cases {
		suite.Run(c.name, func() {
			suite.SetupSuite()
			defer suite.TearDownTest()
			balancer := suite.balancer
			for node, s := range c.distributions {
				balancer.dist.SegmentDistManager.Update(node, s...)
			}
			for i := range c.nodes {
				nodeInfo := session.NewNodeInfo(c.nodes[i], "127.0.0.1:0")
				nodeInfo.UpdateStats(session.WithSegmentCnt(c.segmentCnts[i]))
				nodeInfo.SetState(c.states[i])
				suite.balancer.nodeManager.Add(nodeInfo)
			}
			for i := range c.collectionIDs {
				plans := balancer.AssignSegment(c.collectionIDs[i], c.assignments[i], c.nodes)
				suite.ElementsMatch(c.expectPlans[i], plans)
			}
		})
	}
}

func TestScoreBasedBalancerSuite(t *testing.T) {
	suite.Run(t, new(ScoreBasedBalancerTestSuite))
}
