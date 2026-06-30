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
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v3/msgpb"
	"github.com/milvus-io/milvus/internal/metastore/model"
	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

func TestStaleBackfillSegments(t *testing.T) {
	segments := NewSegmentsInfo()
	// stale sealed data segment: schema_version below the watermark
	segments.SetSegment(1, &SegmentInfo{SegmentInfo: &datapb.SegmentInfo{
		ID: 1, CollectionID: 100, State: commonpb.SegmentState_Flushed, SchemaVersion: 1,
	}})
	// fresh sealed data segment: already at the watermark
	segments.SetSegment(2, &SegmentInfo{SegmentInfo: &datapb.SegmentInfo{
		ID: 2, CollectionID: 100, State: commonpb.SegmentState_Flushed, SchemaVersion: 2,
	}})
	// outside the sealed data population: growing / L0 / dropped
	segments.SetSegment(3, &SegmentInfo{SegmentInfo: &datapb.SegmentInfo{
		ID: 3, CollectionID: 100, State: commonpb.SegmentState_Growing, SchemaVersion: 0,
	}})
	segments.SetSegment(4, &SegmentInfo{SegmentInfo: &datapb.SegmentInfo{
		ID: 4, CollectionID: 100, State: commonpb.SegmentState_Flushed, Level: datapb.SegmentLevel_L0, SchemaVersion: 0,
	}})
	segments.SetSegment(5, &SegmentInfo{SegmentInfo: &datapb.SegmentInfo{
		ID: 5, CollectionID: 100, State: commonpb.SegmentState_Dropped, SchemaVersion: 0,
	}})
	s := &Server{meta: &meta{segments: segments, collections: newTestCollections(100)}}

	assert.ElementsMatch(t, []int64{1}, s.StaleBackfillSegments(context.Background(), 100, 2))
	assert.Empty(t, s.StaleBackfillSegments(context.Background(), 100, 1))
}

func TestGrowingSegmentsBefore(t *testing.T) {
	ctx := context.Background()
	segments := NewSegmentsInfo()
	// growing segment born before the tick: may hold pre-V rows
	segments.SetSegment(1, &SegmentInfo{SegmentInfo: &datapb.SegmentInfo{
		ID: 1, CollectionID: 100, State: commonpb.SegmentState_Growing,
		StartPosition: &msgpb.MsgPosition{Timestamp: 41},
	}})
	// growing segment born after the tick: computes the function field on ingest
	segments.SetSegment(2, &SegmentInfo{SegmentInfo: &datapb.SegmentInfo{
		ID: 2, CollectionID: 100, State: commonpb.SegmentState_Growing,
		StartPosition: &msgpb.MsgPosition{Timestamp: 43},
	}})
	// growing segment without a start position: nothing synced yet — counting it as
	// ts 0 would stall every round behind fresh segments under sustained ingest.
	segments.SetSegment(3, &SegmentInfo{SegmentInfo: &datapb.SegmentInfo{
		ID: 3, CollectionID: 100, State: commonpb.SegmentState_Growing,
	}})
	// sealed segment: outside the growing population regardless of position
	segments.SetSegment(4, &SegmentInfo{SegmentInfo: &datapb.SegmentInfo{
		ID: 4, CollectionID: 100, State: commonpb.SegmentState_Flushed,
		StartPosition: &msgpb.MsgPosition{Timestamp: 1},
	}})
	s := &Server{meta: &meta{segments: segments, collections: newTestCollections(100)}}

	assert.ElementsMatch(t, []int64{1}, s.GrowingSegmentsBefore(ctx, 100, 42))
	assert.Empty(t, s.GrowingSegmentsBefore(ctx, 100, 20))
}

func TestUnindexedSegmentsBefore(t *testing.T) {
	ctx := context.Background()
	const (
		collID  = int64(100)
		fieldID = int64(10)
		indexID = int64(500)
		tick    = uint64(42)
	)

	segments := NewSegmentsInfo()
	// pre-tick sealed segment WITH a finished index on the field -> covered
	segments.SetSegment(1, &SegmentInfo{SegmentInfo: &datapb.SegmentInfo{
		ID: 1, CollectionID: collID, State: commonpb.SegmentState_Flushed,
		StartPosition: &msgpb.MsgPosition{Timestamp: 41},
	}})
	// pre-tick sealed segment WITHOUT the index -> reported
	segments.SetSegment(2, &SegmentInfo{SegmentInfo: &datapb.SegmentInfo{
		ID: 2, CollectionID: collID, State: commonpb.SegmentState_Flushed,
		StartPosition: &msgpb.MsgPosition{Timestamp: 41},
	}})
	// sealed segment with NO start position -> counts as pre-tick (fail-closed) -> reported
	segments.SetSegment(3, &SegmentInfo{SegmentInfo: &datapb.SegmentInfo{
		ID: 3, CollectionID: collID, State: commonpb.SegmentState_Flushed,
	}})
	// post-tick sealed segment: computed the column on ingest, exempt even unindexed
	segments.SetSegment(4, &SegmentInfo{SegmentInfo: &datapb.SegmentInfo{
		ID: 4, CollectionID: collID, State: commonpb.SegmentState_Flushed,
		StartPosition: &msgpb.MsgPosition{Timestamp: 43},
	}})
	// pre-tick but growing: outside the sealed population (the growing check owns it)
	segments.SetSegment(5, &SegmentInfo{SegmentInfo: &datapb.SegmentInfo{
		ID: 5, CollectionID: collID, State: commonpb.SegmentState_Growing,
		StartPosition: &msgpb.MsgPosition{Timestamp: 41},
	}})

	im := newSegmentIndexMeta(nil)
	im.indexes[collID] = map[UniqueID]*model.Index{
		indexID: {CollectionID: collID, FieldID: fieldID, IndexID: indexID},
	}
	seg1Indexes := typeutil.NewConcurrentMap[UniqueID, *model.SegmentIndex]()
	seg1Indexes.Insert(indexID, &model.SegmentIndex{
		SegmentID: 1, CollectionID: collID, IndexID: indexID, IndexState: commonpb.IndexState_Finished,
	})
	im.segmentIndexes.Insert(1, seg1Indexes)

	s := &Server{meta: &meta{segments: segments, collections: newTestCollections(collID), indexMeta: im}}
	assert.ElementsMatch(t, []int64{2, 3}, s.UnindexedSegmentsBefore(ctx, collID, fieldID, tick))

	// No index defined on the field at all: every pre-tick sealed segment reads as
	// unindexed (fail-closed until the index is created).
	assert.ElementsMatch(t, []int64{1, 2, 3}, s.UnindexedSegmentsBefore(ctx, collID, int64(11), tick))
}
