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

package coordinator

import (
	"context"
)

// Readiness: revoke ⟺ the write side (datacoord meta) shows the round's backfill is
// complete. The gate deliberately checks NO read-side state (target / dist): once every
// segment carries the backfilled data (and, for vector fields, its index), the ordinary
// load/update machinery serves it; the gate only protects the window in which the data
// itself does not exist yet.
//
// Two conditions, both answered by datacoord meta:
//
//  1. Schema watermark — every sealed data segment carries schema_version ≥ V
//     (StaleBackfillSegments == ∅; segments flushed after the DDL that still carry a
//     zero/old stamp count as stale and are advanced by the backfill), and no growing
//     segment was born before the DDL timetick (GrowingSegmentsBefore == ∅ — growing
//     segments cannot carry a trustworthy schema_version (#48865), so the START
//     TIMESTAMP arbitrates: born after the tick ⇒ the ingest path computed the field).
//  2. Vector index — for each gated field that is a VECTOR type, every sealed data
//     segment born before the DDL timetick has a Finished index on that field. Pre-tick
//     segments are exactly the ones whose vector column was materialized by the
//     backfill; serving them requires the index to exist first. Segments born after the
//     tick computed the column on ingest and are exempt. No index defined on the field
//     reads as unindexed — fail-closed until the index is created.

// DataViewReader exposes the datacoord segment meta (the write-side truth) to the
// readiness check. datacoord's Server satisfies it structurally; mixCoord wires it.
type DataViewReader interface {
	// StaleBackfillSegments returns the IDs of the collection's sealed data segments
	// whose schema_version is strictly below the watermark, i.e. the write side has not
	// finished backfilling them.
	StaleBackfillSegments(ctx context.Context, collectionID int64, watermark int32) []int64
	// GrowingSegmentsBefore returns the IDs of the collection's growing segments whose
	// start position is strictly before ts.
	GrowingSegmentsBefore(ctx context.Context, collectionID int64, ts uint64) []int64
	// UnindexedSegmentsBefore returns the IDs of the collection's sealed data segments
	// whose start position is strictly before ts (nil start position counts as before —
	// fail-closed) and that do NOT have a Finished index on fieldID. A field with no
	// index defined at all reads as everything-unindexed.
	UnindexedSegmentsBefore(ctx context.Context, collectionID, fieldID int64, ts uint64) []int64
}

// ReadinessProvider answers "can ALL in-scope segments correctly serve this round?".
// Stateless: every answer is re-derived from current datacoord meta.
type ReadinessProvider struct {
	dataView DataViewReader
}

// NewReadinessProvider constructs a provider over the datacoord meta view.
func NewReadinessProvider(dataView DataViewReader) *ReadinessProvider {
	if dataView == nil {
		panic("bump_defence readiness provider requires a data-view reader")
	}
	return &ReadinessProvider{dataView: dataView}
}

// IsRoundReady reports whether the round's backfill is complete on the write side. A
// round reveals atomically — it returns false until every condition holds.
func (p *ReadinessProvider) IsRoundReady(ctx context.Context, round *BackfillRound) bool {
	if round == nil || len(round.Fields) == 0 {
		return true
	}
	if round.Scope.Kind != ScopeWatermark {
		// Unknown scope can never be certified; Reload drops such rounds, so this is a
		// defensive fail-closed backstop.
		return false
	}
	// Condition 1a: every sealed data segment carries the bumped schema. The sealed
	// population is not monotone (late imports, post-tick flushes stamped zero), so this
	// is re-checked every sweep; one landing after a revoke is the accepted residual.
	if len(p.dataView.StaleBackfillSegments(ctx, round.CollectionID, round.Scope.Watermark)) > 0 {
		return false
	}
	tick := round.Scope.SchemaChangeTimeTick
	if tick == 0 {
		// No timetick recorded (defensive tolerance): the growing and index conditions
		// have no time axis to cut on, so the watermark alone decides.
		return true
	}
	// Condition 1b: growing segments carry no trustworthy schema_version, so the start
	// timestamp arbitrates — one born before the tick may hold pre-V rows and must first
	// seal (then condition 1a takes over).
	if len(p.dataView.GrowingSegmentsBefore(ctx, round.CollectionID, tick)) > 0 {
		return false
	}
	// Condition 2: backfilled VECTOR fields additionally need their index on every
	// pre-tick sealed segment before the round may serve.
	for _, fieldID := range round.Scope.VectorFields {
		if len(p.dataView.UnindexedSegmentsBefore(ctx, round.CollectionID, fieldID, tick)) > 0 {
			return false
		}
	}
	return true
}
