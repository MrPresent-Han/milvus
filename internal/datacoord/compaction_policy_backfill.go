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

package datacoord

import (
	"context"
	"fmt"

	"go.uber.org/zap"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/datacoord/allocator"
	"github.com/milvus-io/milvus/pkg/v3/log"
	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
)

type backfillCompactionPolicy struct {
	meta      *meta
	handler   Handler
	allocator allocator.Allocator
}

var _ CompactionPolicy = (*backfillCompactionPolicy)(nil)

func newBackfillCompactionPolicy(meta *meta, allocator allocator.Allocator, handler Handler) *backfillCompactionPolicy {
	return &backfillCompactionPolicy{meta: meta, allocator: allocator, handler: handler}
}

func (policy *backfillCompactionPolicy) Enable() bool {
	return true
}

func (policy *backfillCompactionPolicy) Name() string {
	return "BumpSchemaVersion"
}

// staleFlushedSegments returns the flushed segments for collectionID whose SchemaVersion
// lags behind collectionSchemaVersion. L0 segments are excluded (deletes only, no data).
func (policy *backfillCompactionPolicy) staleFlushedSegments(collectionID int64, collectionSchemaVersion int32) []*chanPartSegments {
	return GetSegmentsChanPart(policy.meta, collectionID, SegmentFilterFunc(func(segment *SegmentInfo) bool {
		return isSegmentHealthy(segment) &&
			isFlushed(segment) &&
			!segment.isCompacting &&
			!segment.GetIsImporting() &&
			!segment.GetIsInvisible() &&
			segment.GetLevel() != datapb.SegmentLevel_L0 &&
			segment.GetSchemaVersion() < collectionSchemaVersion
	}))
}

func (policy *backfillCompactionPolicy) Trigger(ctx context.Context) (map[CompactionTriggerType][]CompactionView, error) {
	collections := policy.meta.GetCollections()
	events := make(map[CompactionTriggerType][]CompactionView)

	for _, collection := range collections {
		if collection.Schema == nil {
			continue
		}
		collectionID := collection.ID
		collectionSchemaVersion := collection.Schema.GetVersion()
		partSegments := policy.staleFlushedSegments(collectionID, collectionSchemaVersion)

		var views []CompactionView
		var collectionTriggerID int64
		for _, group := range partSegments {
			for _, segment := range group.segments {
				segmentID := segment.GetID()
				segmentViews := GetViewsByInfo(segment)
				if len(segmentViews) == 0 {
					log.Ctx(ctx).Warn("GetViewsByInfo returned empty views, skip segment",
						zap.Int64("segmentID", segmentID))
					continue
				}
				if len(segmentViews) != 1 {
					log.Ctx(ctx).Warn("GetViewsByInfo returned unexpected view count, using first view only",
						zap.Int64("segmentID", segmentID),
						zap.Int("viewCount", len(segmentViews)))
				}

				if collectionTriggerID == 0 {
					id, err := policy.allocator.AllocID(ctx)
					if err != nil {
						log.Ctx(ctx).Warn("Failed to allocate triggerID for schema version bump, skip remaining segments in current collection",
							zap.Int64("collectionID", collectionID),
							zap.Error(err))
						break
					}
					collectionTriggerID = id
				}

				log.Ctx(ctx).Info("Found segment needing schema version bump",
					zap.Int64("segmentID", segmentID),
					zap.Int64("collectionID", collectionID),
					zap.Int32("segmentSchemaVersion", segment.GetSchemaVersion()),
					zap.Int32("collectionSchemaVersion", collectionSchemaVersion))
				views = append(views, &BumpSchemaVersionView{
					label:     segmentViews[0].label,
					segments:  segmentViews,
					triggerID: collectionTriggerID,
					schema:    collection.Schema,
				})
			}
		}
		if len(views) > 0 {
			events[TriggerTypeBackfill] = append(events[TriggerTypeBackfill], views...)
		}
	}
	return events, nil
}

type BumpSchemaVersionView struct {
	label     *CompactionGroupLabel
	segments  []*SegmentView
	triggerID int64

	// schema is captured at policy-scan time so completion only advances the segment
	// to the schema version that this task reconciled.
	schema *schemapb.CollectionSchema
}

var _ CompactionView = (*BumpSchemaVersionView)(nil)

func (v *BumpSchemaVersionView) GetGroupLabel() *CompactionGroupLabel {
	return v.label
}

func (v *BumpSchemaVersionView) GetSegmentsView() []*SegmentView {
	return v.segments
}

func (v *BumpSchemaVersionView) Append(segments ...*SegmentView) {
	v.segments = append(v.segments, segments...)
}

func (v *BumpSchemaVersionView) String() string {
	return fmt.Sprintf("BumpSchemaVersionView: label=%s, segments=%d, triggerID=%d, schemaVersion=%d",
		v.label.Key(), len(v.segments), v.triggerID, v.schema.GetVersion())
}

func (v *BumpSchemaVersionView) Trigger() (CompactionView, string) {
	return v, "segment schema version behind collection schema"
}

func (v *BumpSchemaVersionView) ForceTrigger() (CompactionView, string) {
	return v.Trigger()
}

func (v *BumpSchemaVersionView) ForceTriggerAll() ([]CompactionView, string) {
	view, reason := v.Trigger()
	return []CompactionView{view}, reason
}

func (v *BumpSchemaVersionView) GetTriggerID() int64 {
	return v.triggerID
}
