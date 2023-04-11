package balance

import (
	"sort"

	"github.com/milvus-io/milvus/internal/querycoordv2/meta"
	"github.com/milvus-io/milvus/internal/querycoordv2/session"
)

type Policy interface {
	AssignSegment(collectionID int64, segments []*meta.Segment, nodeInfos []*session.NodeInfo, deltaMap map[int64]int) []SegmentAssignPlan
	AssignChannel(channels []*meta.DmChannel, nodeInfos []*session.NodeInfo) []ChannelAssignPlan
	BalanceReplica(replica *meta.Replica) ([]SegmentAssignPlan, []ChannelAssignPlan)
}

type RoundRobinPolicy struct {
}

func (roundRobinPolicy *RoundRobinPolicy) AssignSegment(collectionID int64, segments []*meta.Segment,
	nodesInfo []*session.NodeInfo, deltaMap map[int64]int) []SegmentAssignPlan {
	if len(nodesInfo) == 0 {
		return nil
	}
	sort.Slice(nodesInfo, func(i, j int) bool {
		cnt1, cnt2 := nodesInfo[i].SegmentCnt(), nodesInfo[j].SegmentCnt()
		delta1, delta2 := deltaMap[nodesInfo[i].ID()], deltaMap[nodesInfo[j].ID()]
		return cnt1+delta1 < cnt2+delta2
	})
	ret := make([]SegmentAssignPlan, 0, len(segments))
	for i, s := range segments {
		plan := SegmentAssignPlan{
			Segment: s,
			From:    -1,
			To:      nodesInfo[i%len(nodesInfo)].ID(),
		}
		ret = append(ret, plan)
	}
	return ret
}

func (roundRobinPolicy *RoundRobinPolicy) AssignChannel(channels []*meta.DmChannel,
	nodesInfo []*session.NodeInfo, deltaMap map[int64]int) []ChannelAssignPlan {
	if len(nodesInfo) == 0 {
		return nil
	}
	sort.Slice(nodesInfo, func(i, j int) bool {
		cnt1, cnt2 := nodesInfo[i].ChannelCnt(), nodesInfo[j].ChannelCnt()
		delta1, delta2 := deltaMap[nodesInfo[i].ID()], deltaMap[nodesInfo[j].ID()]
		return cnt1+delta1 < cnt2+delta2
	})
	ret := make([]ChannelAssignPlan, 0, len(channels))
	for i, c := range channels {
		plan := ChannelAssignPlan{
			Channel: c,
			From:    -1,
			To:      nodesInfo[i%len(nodesInfo)].ID(),
		}
		ret = append(ret, plan)
	}
	return ret
}

func (roundRobinPolicy *RoundRobinPolicy) Balance() ([]SegmentAssignPlan, []ChannelAssignPlan) {

	return segmentPlans, channelPlans
}
