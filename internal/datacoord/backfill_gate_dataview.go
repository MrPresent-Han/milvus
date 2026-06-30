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

	"github.com/samber/lo"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

// bump_defence DataView: the write-side truth the coordinator's readiness witness reads
// (the watermark round's write-side check). Structurally satisfies
// coordinator.DataViewReader; mixCoord wires the Server in directly.

// StaleBackfillSegments returns the IDs of the collection's sealed data segments whose
// schema_version is strictly below the watermark, i.e. the write side has not finished
// backfilling them yet. Empty means the DataView is clear for that watermark.
//
// Growing segments are outside the sealed population (isSealedDataSegment) and thus not
// counted; besides being definitionally un-sealed, streaming-created growing segments
// currently carry SchemaVersion=0 (#48865), so counting them would block forever under
// write traffic. This is safe: the schema change seals them, and the backfill picks the
// flushed result up and advances its SchemaVersion.
func (s *Server) StaleBackfillSegments(ctx context.Context, collectionID int64, watermark int32) []int64 {
	segments := s.meta.SelectSegments(ctx, WithCollection(collectionID), SegmentFilterFunc(func(si *SegmentInfo) bool {
		return isSealedDataSegment(si) && si.GetSchemaVersion() < watermark
	}))
	out := make([]int64, 0, len(segments))
	for _, seg := range segments {
		out = append(out, seg.GetID())
	}
	return out
}

// GrowingSegmentsBefore returns the IDs of the collection's growing segments whose start
// position is strictly before ts. Growing segments carry no trustworthy schema_version
// (#48865), so the start timestamp arbitrates instead: born before the DDL timetick, the
// segment may hold pre-V rows and must first seal (the DDL fence-flushes it; once
// flipped it is a <V sealed segment and StaleBackfillSegments takes over).
//
// A growing segment with NO start position is skipped: the field is only stamped by the
// segment's first sync, so a nil position means nothing durable yet. Counting it as ts 0
// would block every round behind the newest not-yet-synced segment under sustained
// ingest; the accepted residual is pre-tick rows buffered but not yet synced during the
// DDL fence-flush window.
func (s *Server) GrowingSegmentsBefore(ctx context.Context, collectionID int64, ts uint64) []int64 {
	segments := s.meta.SelectSegments(ctx, WithCollection(collectionID), SegmentFilterFunc(func(si *SegmentInfo) bool {
		return si.GetState() == commonpb.SegmentState_Growing &&
			si.GetStartPosition() != nil && si.GetStartPosition().GetTimestamp() < ts
	}))
	return lo.Map(segments, func(seg *SegmentInfo, _ int) int64 { return seg.GetID() })
}

// UnindexedSegmentsBefore returns the IDs of the collection's sealed data segments whose
// start position is strictly before ts and that do NOT have a Finished index on fieldID.
// Segments born before the DDL timetick had their vector column materialized by the
// backfill (not computed on ingest), so serving them requires the index to exist first;
// compaction recalculates the result segment's start position from the actual row
// timestamps, so backfilled old data keeps its pre-tick start. A sealed segment with no
// start position counts as pre-tick (fail-closed; every current write path stamps it).
// A field with no index defined at all reads as everything-unindexed
// (GetIndexedSegments returns nil), holding the round until the index is created.
func (s *Server) UnindexedSegmentsBefore(ctx context.Context, collectionID, fieldID int64, ts uint64) []int64 {
	segments := s.meta.SelectSegments(ctx, WithCollection(collectionID), SegmentFilterFunc(func(si *SegmentInfo) bool {
		return isSealedDataSegment(si) &&
			(si.GetStartPosition() == nil || si.GetStartPosition().GetTimestamp() < ts)
	}))
	candidates := lo.Map(segments, func(seg *SegmentInfo, _ int) int64 { return seg.GetID() })
	indexed := typeutil.NewUniqueSet(s.meta.indexMeta.GetIndexedSegments(collectionID, candidates, []int64{fieldID})...)
	return lo.Filter(candidates, func(segID int64, _ int) bool { return !indexed.Contain(segID) })
}
