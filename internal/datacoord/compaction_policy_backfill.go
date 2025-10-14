package datacoord

import (
	"context"
	"fmt"
	"time"

	"go.uber.org/zap"

	"github.com/milvus-io/milvus/internal/datacoord/allocator"
	"github.com/milvus-io/milvus/pkg/v2/log"
)

type backfillCompactionPolicy struct {
	meta      *meta
	handler   Handler
	allocator allocator.Allocator
}

// Ensure backfillCompactionPolicy implements CompactionPolicy interface
var _ CompactionPolicy = (*backfillCompactionPolicy)(nil)

func newBackfillCompactionPolicy(meta *meta, allocator allocator.Allocator, handler Handler) *backfillCompactionPolicy {
	return &backfillCompactionPolicy{meta: meta, allocator: allocator, handler: handler}
}

func (policy *backfillCompactionPolicy) Enable() bool {
	return Params.DataCoordCfg.EnableAutoCompaction.GetAsBool()
}

func (policy *backfillCompactionPolicy) Name() string {
	return "BackfillCompaction"
}

func (policy *backfillCompactionPolicy) Trigger(ctx context.Context) (map[CompactionTriggerType][]CompactionView, error) {
	// Check all collections
	collections := policy.meta.GetCollections()
	events := make(map[CompactionTriggerType][]CompactionView, 0)
	newTriggerID, err := policy.allocator.AllocID(ctx)
	if err != nil {
		return nil, err
	}

	for _, collection := range collections {
		collectionID := collection.ID
		// Use CreatedAt as schema version since datacoord's collectionInfo doesn't have UpdateTimestamp
		collectionSchemaVersion := collection.Schema.GetSchemaVersion()

		// Get all segments for this collection
		partSegments := GetSegmentsChanPart(policy.meta, collectionID, SegmentFilterFunc(func(segment *SegmentInfo) bool {
			return isSegmentHealthy(segment) &&
				isFlushed(segment) &&
				!segment.isCompacting && // not compacting now
				!segment.GetIsImporting() && // not importing now
				!segment.GetIsInvisible()
		}))

		// Check each segment's schema version
		views := make([]CompactionView, 0)
		for _, group := range partSegments {
			for _, segment := range group.segments {
				segmentSchemaVersion := segment.GetSchemaVersion()

				// If segment's schema version is smaller than collection's schema version
				if segmentSchemaVersion < collectionSchemaVersion {
					log.Ctx(ctx).Info("hc===Found segment with outdated schema version",
						zap.Int64("segmentID", segment.GetID()),
						zap.Int64("collectionID", collectionID),
						zap.Uint64("segmentSchemaVersion", segmentSchemaVersion),
						zap.Uint64("collectionSchemaVersion", collectionSchemaVersion))

					// Create BackfillSegmentsView for this segment
					segmentViews := GetViewsByInfo(segment)
					view := &BackfillSegmentsView{
						label:     segmentViews[0].label,
						segments:  segmentViews,
						triggerID: newTriggerID,
					}
					views = append(views, view)
				}
			}
		}

		// Add views to events if any segments need backfill
		if len(views) > 0 {
			if events[TriggerTypeBackfill] == nil {
				events[TriggerTypeBackfill] = make([]CompactionView, 0)
			}
			events[TriggerTypeBackfill] = append(events[TriggerTypeBackfill], views...)
		}
	}

	return events, nil
}

type BackfillSegmentsView struct {
	label         *CompactionGroupLabel
	segments      []*SegmentView
	triggerID     int64
	collectionTTL time.Duration
}

func (v *BackfillSegmentsView) GetGroupLabel() *CompactionGroupLabel {
	return v.label
}

func (v *BackfillSegmentsView) GetSegmentsView() []*SegmentView {
	return v.segments
}

func (v *BackfillSegmentsView) Append(segments ...*SegmentView) {
	v.segments = append(v.segments, segments...)
}

func (v *BackfillSegmentsView) String() string {
	return fmt.Sprintf("BackfillSegmentsView: label=%s, segments=%d, triggerID=%d",
		v.label.Key(), len(v.segments), v.triggerID)
}

func (v *BackfillSegmentsView) Trigger() (CompactionView, string) {
	// For backfill compaction, we always trigger
	return v, "backfill schema version mismatch"
}

func (v *BackfillSegmentsView) ForceTrigger() (CompactionView, string) {
	return v.Trigger()
}

func (v *BackfillSegmentsView) ForceTriggerAll() ([]CompactionView, string) {
	view, reason := v.Trigger()
	return []CompactionView{view}, reason
}

func (v *BackfillSegmentsView) GetTriggerID() int64 {
	return v.triggerID
}
