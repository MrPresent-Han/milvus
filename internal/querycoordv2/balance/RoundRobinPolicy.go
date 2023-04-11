package balance

import "github.com/milvus-io/milvus/internal/querycoordv2/meta"

type Policy interface {
	AssignSegment(collectionID int64, segments []*meta.Segment, nodes []int64) []SegmentAssignPlan
	AssignChannel(channels []*meta.DmChannel, nodes []int64) []ChannelAssignPlan
	Balance() ([]SegmentAssignPlan, []ChannelAssignPlan)
}

type RoundRobinPolicy struct {
}

func (roundRobinPolicy *RoundRobinPolicy) AssignSegment(collectionID int64, segments []*meta.Segment, nodes []int64) []SegmentAssignPlan {

}
