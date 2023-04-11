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

package checkers

import (
	"context"
	"github.com/milvus-io/milvus/internal/querycoordv2/session"
	"golang.org/x/exp/maps"
	"sort"
	"time"

	"github.com/milvus-io/milvus/internal/proto/querypb"
	"github.com/milvus-io/milvus/internal/querycoordv2/balance"
	"github.com/milvus-io/milvus/internal/querycoordv2/meta"
	. "github.com/milvus-io/milvus/internal/querycoordv2/params"
	"github.com/milvus-io/milvus/internal/querycoordv2/task"
	"github.com/milvus-io/milvus/pkg/log"
	"github.com/milvus-io/milvus/pkg/util/typeutil"

	"github.com/samber/lo"
	"go.uber.org/zap"
)

// BalanceChecker checks the cluster distribution and generates balance tasks.
type BalanceChecker struct {
	baseChecker
	policy                          balance.Policy
	meta                            *meta.Meta
	scheduler                       task.Scheduler
	balancedCollectionsCurrentRound typeutil.UniqueSet
	dist                            *meta.DistributionManager
	targetMgr                       *meta.TargetManager
	nodeMgr                         *session.NodeManager
}

func NewBalanceChecker(meta *meta.Meta, policy balance.Policy, scheduler task.Scheduler, dist *meta.DistributionManager,
	targetManager *meta.TargetManager, nodeManager *session.NodeManager) *BalanceChecker {
	return &BalanceChecker{
		policy:                          policy,
		meta:                            meta,
		scheduler:                       scheduler,
		balancedCollectionsCurrentRound: typeutil.NewUniqueSet(),
		dist:                            dist,
		targetMgr:                       targetManager,
		nodeMgr:                         nodeManager,
	}
}

func (b *BalanceChecker) Description() string {
	return "BalanceChecker checks the cluster distribution and generates balance tasks"
}

func (b *BalanceChecker) shouldDoBalance() (bool, []int64) {
	//if configured no auto balance, skip balance
	if !Params.QueryCoordCfg.AutoBalance.GetAsBool() {
		return false, nil
	}
	//if there are still tasks running in the scheduler, skip balance
	//to avoid increase instability
	if b.scheduler.GetSegmentTaskNum() != 0 || b.scheduler.GetChannelTaskNum() != 0 {
		return false, nil
	}
	// loading collection should skip balance
	ids := b.meta.CollectionManager.GetAll()
	loadedCollections := lo.Filter(ids, func(cid int64, _ int) bool {
		return b.meta.CalculateLoadStatus(cid) == querypb.LoadStatus_Loaded
	})
	return len(loadedCollections) == 0, loadedCollections
}

func (b *BalanceChecker) Check(ctx context.Context) []task.Task {
	ret := make([]task.Task, 0)
	shouldDoBalance, loadedCollections := b.shouldDoBalance()
	if !shouldDoBalance {
		return ret
	}
	sort.Slice(loadedCollections, func(i, j int) bool {
		return loadedCollections[i] < loadedCollections[j]
	})
	hasUnBalancedCollections := false
	for _, cid := range loadedCollections {
		if b.balancedCollectionsCurrentRound.Contain(cid) {
			log.Debug("ScoreBasedBalancer has balanced collection, skip balancing in this round",
				zap.Int64("collectionID", cid))
			continue
		}
		hasUnBalancedCollections = true
		replicas := b.meta.ReplicaManager.GetByCollection(cid)

		segmentPlans, channelPlans := make([]balance.SegmentAssignPlan, 0), make([]balance.ChannelAssignPlan, 0)
		for _, replica := range replicas {
			sPlans, cPlans := b.balanceReplica(replica)
			balance.PrintNewBalancePlans(cid, replica.GetID(), sPlans, cPlans)
			segmentPlans = append(segmentPlans, sPlans...)
			channelPlans = append(channelPlans, cPlans...)
		}
		if len(segmentPlans) != 0 || len(channelPlans) != 0 {
			log.Info("ScoreBasedBalancer has generated balance plans for", zap.Int64("collectionID", cid))
			break
		}
		segmentTasks := b.createSegmentTasksWithPriority(ctx, segmentPlans)
		channelTasks := balance.CreateChannelTasksFromPlans(ctx, b.ID(),
			Params.QueryCoordCfg.ChannelTaskTimeout.GetAsDuration(time.Millisecond), channelPlans)
		ret = append(ret, segmentTasks...)
		ret = append(ret, channelTasks...)
		b.balancedCollectionsCurrentRound.Insert(cid)
	}
	if !hasUnBalancedCollections {
		b.balancedCollectionsCurrentRound.Clear()
		log.Debug("BalanceChecker has balanced all " +
			"collections in one round, clear collectionIDs for this round")
	}

	return ret
}

func (b *BalanceChecker) createSegmentTasksWithPriority(ctx context.Context, segmentPlans []balance.SegmentAssignPlan) []task.Task {
	tasks := balance.CreateSegmentTasksFromPlans(ctx, b.ID(), Params.QueryCoordCfg.SegmentTaskTimeout.GetAsDuration(time.Millisecond), segmentPlans)
	task.SetPriorityWithFunc(func(t task.Task) task.Priority {
		if t.Priority() == task.TaskPriorityHigh {
			return task.TaskPriorityHigh
		}
		return task.TaskPriorityLow
	}, tasks...)
	return tasks
}

func (b *BalanceChecker) balanceReplica(replica *meta.Replica) ([]balance.SegmentAssignPlan, []balance.ChannelAssignPlan) {
	nodes := replica.GetNodes()
	if len(nodes) == 0 {
		return nil, nil
	}

	nodesSegments := make(map[int64][]*meta.Segment)
	stoppingNodesSegments := make(map[int64][]*meta.Segment)
	outboundNodes := b.meta.ResourceManager.CheckOutboundNodes(replica)

	// get stopping nodes and available nodes.
	for _, nid := range nodes {
		segments := b.dist.SegmentDistManager.GetByCollectionAndNode(replica.GetCollectionID(), nid)
		// Only balance segments in current targets
		segments = lo.Filter(segments, func(segment *meta.Segment, _ int) bool {
			return b.targetMgr.GetHistoricalSegment(segment.GetCollectionID(), segment.GetID(), meta.CurrentTarget) != nil
		})

		if isStopping, err := b.nodeMgr.IsStoppingNode(nid); err != nil {
			log.Info("not existed node", zap.Int64("nid", nid), zap.Any("segments", segments), zap.Error(err))
			continue
		} else if isStopping {
			stoppingNodesSegments[nid] = segments
		} else if outboundNodes.Contain(nid) {
			log.RatedInfo(10, "meet outbound node, try to move out all segment/channel",
				zap.Int64("collectionID", replica.GetCollectionID()),
				zap.Int64("replicaID", replica.GetCollectionID()),
				zap.Int64("node", nid),
			)
			stoppingNodesSegments[nid] = segments
		} else {
			nodesSegments[nid] = segments
		}
	}

	if len(nodes) == len(stoppingNodesSegments) {
		// no available nodes to balance
		log.Warn("All nodes is under stopping mode or outbound, skip balance replica",
			zap.Int64("collection", replica.CollectionID),
			zap.Int64("replica id", replica.Replica.GetID()),
			zap.String("replica group", replica.Replica.GetResourceGroup()),
			zap.Int64s("nodes", replica.Replica.GetNodes()),
		)
		return nil, nil
	}

	if len(nodesSegments) <= 0 {
		log.Warn("No nodes is available in resource group, skip balance replica",
			zap.Int64("collection", replica.CollectionID),
			zap.Int64("replica id", replica.Replica.GetID()),
			zap.String("replica group", replica.Replica.GetResourceGroup()),
			zap.Int64s("nodes", replica.Replica.GetNodes()),
		)
		return nil, nil
	}

	//print current distribution before generating plans
	balance.PrintCurrentReplicaDist(replica, stoppingNodesSegments, nodesSegments, b.dist.ChannelDistManager)
	if len(stoppingNodesSegments) != 0 {
		log.Info("Handle stopping nodes", zap.Int64("collection", replica.CollectionID),
			zap.Int64("replica id", replica.Replica.GetID()),
			zap.String("replica group", replica.Replica.GetResourceGroup()),
			zap.Any("stopping nodes", maps.Keys(stoppingNodesSegments)),
			zap.Any("available nodes", maps.Keys(nodesSegments)),
		)
		return b.getStoppedSegmentPlan(replica, nodesSegments, stoppingNodesSegments), b.getStoppedChannelPlan(replica, lo.Keys(nodesSegments), lo.Keys(stoppingNodesSegments))
	}

	return b.getNormalSegmentPlan(replica, nodesSegments), b.getNormalChannelPlan(replica, lo.Keys(nodesSegments))
}
