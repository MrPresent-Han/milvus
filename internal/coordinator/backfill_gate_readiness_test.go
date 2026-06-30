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
	"testing"

	"github.com/stretchr/testify/assert"
)

type fakeDataView struct {
	stale           map[int64][]int64 // collectionID -> stale sealed segments
	growingBefore   map[int64][]int64 // collectionID -> growing segments born before any tick
	unindexedBefore map[int64][]int64 // fieldID -> pre-tick sealed segments lacking its index
}

func (f *fakeDataView) StaleBackfillSegments(_ context.Context, collectionID int64, _ int32) []int64 {
	return f.stale[collectionID]
}

func (f *fakeDataView) GrowingSegmentsBefore(_ context.Context, collectionID int64, _ uint64) []int64 {
	return f.growingBefore[collectionID]
}

func (f *fakeDataView) UnindexedSegmentsBefore(_ context.Context, _, fieldID int64, _ uint64) []int64 {
	return f.unindexedBefore[fieldID]
}

func scalarRoundWM(coll, round int64, wm int32, fields ...int64) *BackfillRound {
	return &BackfillRound{CollectionID: coll, RoundID: round, Fields: fields, Scope: NewWatermarkScope(wm, 40 /*tick*/, nil)}
}

func vectorRoundWM(coll, round int64, wm int32, vectorFields []int64, fields ...int64) *BackfillRound {
	return &BackfillRound{CollectionID: coll, RoundID: round, Fields: fields, Scope: NewWatermarkScope(wm, 40 /*tick*/, vectorFields)}
}

func TestReadiness_Watermark(t *testing.T) {
	ctx := context.Background()
	round := scalarRoundWM(1, 1, 3 /*V*/, 10)

	// A stale sealed segment (schema_version < V: not yet backfilled, late import,
	// zero-stamped seal) -> hold.
	staleDV := &fakeDataView{stale: map[int64][]int64{1: {103}}}
	assert.False(t, NewReadinessProvider(staleDV).IsRoundReady(ctx, round))

	// A growing segment born before the DDL tick may hold pre-V rows (growing carries
	// no trustworthy schema_version, the start timestamp arbitrates) -> hold.
	flippingDV := &fakeDataView{growingBefore: map[int64][]int64{1: {104}}}
	assert.False(t, NewReadinessProvider(flippingDV).IsRoundReady(ctx, round))

	// All sealed segments at >= V and no pre-tick growing -> revoke.
	assert.True(t, NewReadinessProvider(&fakeDataView{}).IsRoundReady(ctx, round))

	// Zero tick (defensive tolerance) disables the growing + index conditions; the
	// watermark alone decides.
	legacy := &BackfillRound{CollectionID: 1, RoundID: 2, Fields: []int64{10}, Scope: NewWatermarkScope(3, 0, []int64{10})}
	legacyDV := &fakeDataView{
		growingBefore:   map[int64][]int64{1: {104}},
		unindexedBefore: map[int64][]int64{10: {105}},
	}
	assert.True(t, NewReadinessProvider(legacyDV).IsRoundReady(ctx, legacy))
}

func TestReadiness_VectorIndex(t *testing.T) {
	ctx := context.Background()
	round := vectorRoundWM(1, 1, 3 /*V*/, []int64{11}, 10, 11)

	// A pre-tick sealed segment without a Finished index on the gated vector field
	// (including "no index defined at all") -> hold: its vector column was backfilled,
	// serving it needs the index first.
	unindexedDV := &fakeDataView{unindexedBefore: map[int64][]int64{11: {105}}}
	assert.False(t, NewReadinessProvider(unindexedDV).IsRoundReady(ctx, round))

	// The scalar gated field never consults the index condition.
	scalarOnlyDV := &fakeDataView{unindexedBefore: map[int64][]int64{10: {105}}}
	assert.True(t, NewReadinessProvider(scalarOnlyDV).IsRoundReady(ctx, round))

	// Index coverage complete on every pre-tick segment -> revoke.
	assert.True(t, NewReadinessProvider(&fakeDataView{}).IsRoundReady(ctx, round))
}

func TestReadiness_Trivial(t *testing.T) {
	ctx := context.Background()
	p := NewReadinessProvider(&fakeDataView{})

	// Nil round / empty fields pass trivially.
	assert.True(t, p.IsRoundReady(ctx, nil))
	assert.True(t, p.IsRoundReady(ctx, &BackfillRound{CollectionID: 1, RoundID: 1}))

	// A non-watermark scope kind can never be certified -> hold (defensive backstop;
	// Reload drops such rounds).
	unknown := &BackfillRound{CollectionID: 1, RoundID: 2, Fields: []int64{10}, Scope: BackfillScope{Kind: BackfillScopeKind(99)}}
	assert.False(t, p.IsRoundReady(ctx, unknown))
}

func TestNewReadinessProviderRequiresReader(t *testing.T) {
	assert.Panics(t, func() { NewReadinessProvider(nil) })
}
