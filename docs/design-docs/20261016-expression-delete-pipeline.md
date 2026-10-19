# Segcore Expression Delete Pipeline

Date: 2026-10-16

Discussion checkpoint: 2026-10-19

Status: Draft — agreed architectural direction; not implemented

Scope: Delegator handoff through single-segment execution and in-memory delete application, with system-level discussion checkpoints for compaction, drop-field delete drainage, scheduling statistics, and target-QN segment loading

## 1. Goal and agreed decisions

Execute a delete predicate on each target segment and apply its matching row offsets directly to `DeletedRecord`, without materializing primary keys for a round trip through Proxy or looking those keys up again during application.

The agreed direction is:

- Delegator receives an expression plan and a delete timestamp, rather than an expanded list of matching PKs, for this new path.
- Reuse expression execution to produce a segment-local selection bitmap; predicates may reference any supported scalar field, such as `age < 18`.
- Add a dedicated `DeleteApplyNode` that consumes the selection and calls a segment-level `ApplyDeletesMap` entry.
- Use the same logical pipeline for Growing and Sealed segments.
- Preserve the current `DeletedRecord` representation and query algorithms; add or refactor its write entry to accept selected offsets directly.
- Retain existing PK-based delete and delta-load entries for compatibility.
- Restrict the native predicate-delete path to a deliberately small, explicitly supported expression set with stable, verified semantics across query execution and compaction. Expressions outside that set retain the existing expression-query-to-PK-delete path; do not approximate them.
- Before physical field removal, require complete durable coverage of pre-barrier predicate deletes in the affected segments' delta logs or rewritten data. Prefer checkpoint-style drainage of both PK and predicate deletes rather than separate type-specific drain progress; see Section 9.1.3.
- During target-QN segment loading, apply predicate deltas through the existing Segcore expression execution paths: use suitable scalar indexes without reading the corresponding raw columns, otherwise read required data on demand. Do not introduce an additional forced eager-load step; see Section 9.3.

This is primarily a local execution design, not a complete replacement for the distributed delete protocol. Durable ordering, replay, compaction, and release/reload behavior must be designed before production integration. The current direction retains the existing loaded-data prerequisite for general predicate deletes; extending them to unloaded data is not an initial requirement.

## 2. Motivation and industry precedent

The existing expression-delete path retrieves matching PKs before applying PK-based deletes. At very high match counts, PK materialization, result-size limits, transfer, and repeated PK-to-offset lookup can become substantial costs.

### 2.1 Shared convention and comparison scope

The mature shared pattern is **evaluate a condition -> identify local rows -> mutate through a storage-owned entry**. An explicit SQL mutation operator is one realization; a segment-local query-delete loop is another.

The comparison below inspects fixed release tags, not moving branches. It establishes implementation precedent, not comparative performance or proof of Milvus correctness.

| Reference | Why it is relevant | Main difference from this proposal |
| --- | --- | --- |
| DuckDB v1.4.0 | Closest executor/sink decomposition | Consumes row-ID vectors, not Milvus bitmaps |
| Lucene 9.12.0 | Closest segment-local delete-by-query application | Uses query iterators and live-doc bits, not timestamped `DeletedRecord` history |
| PostgreSQL 17.0 | Mature selection/mutation boundary and conflict handling | Ordinary heap deletion uses tuple identifiers and transaction semantics |
| ClickHouse v25.8.1.5101-lts | Compaction-side precedent: execute delete conditions as filters during data-part rewriting | Reuses its own expression machinery; does not establish parity for a separate Go evaluator |

### 2.2 DuckDB: borrow the explicit delete sink

```text
Child selection plan -> row-ID column in DataChunk
  -> PhysicalDelete::Sink -> DataTable::Delete
```

`PlanDelete` attaches the child plan to `PhysicalDelete`. Its sink reads `chunk.data[row_id_index]` and passes a batch of row IDs to the table deletion interface; it need not recover primary keys first. Extra paths handle indexes and `RETURNING`. Sources: [plan construction, v1.4.0](https://github.com/duckdb/duckdb/blob/v1.4.0/src/execution/physical_plan/plan_delete.cpp), [PhysicalDelete, v1.4.0](https://github.com/duckdb/duckdb/blob/v1.4.0/src/execution/operator/persistent/physical_delete.cpp).

**Adopt:** a named terminal mutation operator, batched local row identity, and a separate storage deletion entry. This is the strongest precedent for `DeleteApplyNode -> Segment::ApplyDeletesMap`.

**Adapt:** retain Milvus's existing bitmap representation rather than introducing a row-ID column merely to imitate DuckDB. Keep counts and snapshot updates behind the storage boundary.

**Do not copy:** its transaction machinery, index maintenance, or locking scope. A sink abstraction alone does not supply rollback or atomic publication.

### 2.3 Lucene: borrow segment-local query execution and bounded application

```text
IndexWriter.deleteDocuments(Query...)
  -> buffered query deletes
  -> per-segment query rewrite / Weight / Scorer
  -> matching local doc IDs
  -> ReadersAndUpdates.delete(docID) -> PendingDeletes
```

The public entry buffers query deletes rather than asking the caller for matching IDs. See [IndexWriter, 9.12.0](https://github.com/apache/lucene/blob/releases/lucene/9.12.0/lucene/core/src/java/org/apache/lucene/index/IndexWriter.java).

`FrozenBufferedUpdates::applyQueryDeletes` runs queries against segment readers without scoring, iterates matching local doc IDs, and calls segment delete bookkeeping. It checks delete generations; segment-private queries additionally use a document limit, remapped when flush sorting changed the ID space. See [FrozenBufferedUpdates, 9.12.0](https://github.com/apache/lucene/blob/releases/lucene/9.12.0/lucene/core/src/java/org/apache/lucene/index/FrozenBufferedUpdates.java).

`ReadersAndUpdates.delete` delegates to `PendingDeletes`. That class maintains writable live-doc bits with copy-on-write and counts only newly cleared bits. Sources: [ReadersAndUpdates, 9.12.0](https://github.com/apache/lucene/blob/releases/lucene/9.12.0/lucene/core/src/java/org/apache/lucene/index/ReadersAndUpdates.java), [PendingDeletes, 9.12.0](https://github.com/apache/lucene/blob/releases/lucene/9.12.0/lucene/core/src/java/org/apache/lucene/index/PendingDeletes.java).

**Adopt:** evaluate the condition where segment data lives, consume local positions directly, and leave delete-state ownership to storage.

**Adapt:** its bounded application illustrates why Milvus needs a captured row range and stable offset layout during concurrent insertion or reordering. This is an analogy, not an equivalence between Lucene generations and Milvus timestamps.

**Do not copy:** a live-doc bitmap as the entire Milvus delete representation. Milvus must retain `(delete_ts, offset)` history and timestamped query behavior. Lucene also does not establish that this operation must be an SQL-style execution operator.

### 2.4 PostgreSQL: borrow the boundary, not heap transaction semantics

```text
Selection subplan -> row-locating information
  -> ModifyTable / ExecDelete -> table_tuple_delete
```

For ordinary tables, `ExecDeleteAct` invokes `table_tuple_delete` with a tuple identifier and snapshot information. `ExecDelete` handles outcomes such as concurrent update/deletion, including rechecks or serialization errors where required. See [nodeModifyTable.c, REL_17_0](https://github.com/postgres/postgres/blob/REL_17_0/src/backend/executor/nodeModifyTable.c).

**Adopt:** selection produces row identity, while mutation explicitly handles storage outcomes. This supports an error/completion contract for `DeleteApplyNode`, rather than assuming successful filtering guarantees successful application.

**Do not copy:** heap TIDs, row locks, triggers, or EvalPlanQual into Segcore. PostgreSQL uses a generic modifying operator, so it supports the responsibility boundary, not the necessity of a dedicated operator class or bitmap ABI.

### 2.5 ClickHouse: borrow predicate filtering during data rewriting

**Primary precedent for the compaction direction in Section 9.1.** ClickHouse's `ALTER TABLE ... DELETE WHERE` rewrites affected data parts. See the [official DELETE mutation documentation](https://clickhouse.com/docs/reference/statements/alter/delete).

```text
Delete mutation predicate -> retaining-row filter
  -> FilterStep in the mutation execution pipeline -> rewritten data part
```

In the fixed release, `MutationsInterpreter` wraps a DELETE predicate with `isZeroOrNull` to retain non-matching or NULL rows, and `addStreamsForLaterStages` inserts `FilterStep` for deletion stages. See [MutationsInterpreter.cpp, v25.8.1.5101-lts](https://github.com/ClickHouse/ClickHouse/blob/v25.8.1.5101-lts/src/Interpreters/MutationsInterpreter.cpp).

**Adopt:** apply predicate filtering in the rewrite pipeline before writing surviving rows.

**Adapt:** evaluate a restricted Milvus expression set over the Go compactor's record batches, preserving Milvus timestamp and schema semantics.

**Do not copy:** ClickHouse's mutation ordering, transaction behavior, expression language, or full execution framework. A separate Go evaluator and strict allowlist are Milvus-specific choices, not an industry-standard consequence of this precedent.

**Evidence boundary:** this is direct implementation evidence for filter-then-rewrite. It neither proves Go/Segcore semantic equivalence nor establishes a performance benefit for Milvus; divergence could retain deleted rows or physically remove live rows.

### 2.6 Decisions derived from the comparison

| Milvus decision | Reference principle | Applicability boundary |
| --- | --- | --- |
| Explicit `DeleteApplyNode` | DuckDB delete sink; PostgreSQL mutation boundary | Fits the existing Segcore execution framework; does not create transaction guarantees |
| Segment-local predicate application without PK expansion | Lucene delete-by-query; DuckDB local row IDs | Remove the PK round trip, not predicate-evaluation cost |
| Bitmap-based `ApplyDeletesMap` | Local-position consumption in both models | Milvus-specific representation; define polarity, bounds, and ownership explicitly |
| Captured row range and pinned offset layout | Lucene bounded application and sort remapping | Preserve Milvus's own strict insert/delete timestamp boundary |
| Storage-owned deduplication and visibility state | All three separate selection from storage mutation | Preserve `DeletedRecord`; do not replace its history with Lucene live-doc bits |
| Predicate filtering in compaction's rewrite pipeline | ClickHouse DELETE mutation stages | Borrow the decomposition, not its evaluator or mutation protocol; independently verify Go/Segcore parity |

These sources strongly support the architectural decomposition. They do **not** establish a standard interface named `ApplyDeletesMap`, prove Growing/Sealed concurrency safety, or solve Milvus WAL/L0 replay. Those remain Milvus-specific contracts to implement and validate.

## 3. Current implementation

The relevant source paths are:

- [SegmentInterface.h](../../internal/core/src/segcore/SegmentInterface.h): the current `Delete` entry accepts PKs and timestamps, not a selection bitmap.
- [ChunkedSegmentSealedImpl.cpp](../../internal/core/src/segcore/ChunkedSegmentSealedImpl.cpp): Sealed deletion invokes `DeletedRecord<true>::StreamPush` with PKs; its lookup callback resolves those PKs to segment offsets.
- [SegmentGrowingImpl.cpp](../../internal/core/src/segcore/SegmentGrowingImpl.cpp): Growing deletion uses its insert-record PK lookup and `DeletedRecord<false>`.
- [DeletedRecord.h](../../internal/core/src/segcore/DeletedRecord.h): `InternalPush` performs PK lookup through `search_pk_func_`, then updates the timestamp/offset skip list, deletion mask, counters, and resource accounting; `StreamPush` also maintains delete snapshots.
- [FilterBitsNode.cpp](../../internal/core/src/exec/operator/FilterBitsNode.cpp) and [MvccNode.cpp](../../internal/core/src/exec/operator/MvccNode.cpp): predicate evaluation and query visibility filtering already operate on segment-aligned bitmaps.
- [QueryContext.h](../../internal/core/src/exec/QueryContext.h): the query execution context exposes a const segment pointer; a mutation pipeline needs an explicit write-capable entry/context.
- [segment.go](../../internal/querynodev2/segments/segment.go): `LocalSegment.Delete` pins segment lifetime and serializes delete mutation with `deltaMut`.

The current interfaces do not already implement expression-delete application. This proposal must not be confused with an existing capability.

## 4. Proposed pipeline

```text
Delegator: expression plan + delete_ts + execution context
  -> target worker / pinned segment
  -> FilterBitsNode: evaluate predicate
  -> delete-specific visibility filtering
  -> DeleteApplyNode: consume matching row bitmap
  -> Segment::ApplyDeletesMap
  -> shared DeletedRecord offset-application logic
```

The expression belongs to the plan and predicate evaluator. Operators downstream of filtering consume its bitmap result, not a repeatedly interpreted expression string. No PK projection is required between selection and application.

### 4.1 Selection and visibility

For segment row offset `i`, the intended eligibility is:

```text
predicate(i) is TRUE
AND i belongs to the captured execution row range
AND insert_ts(i) < delete_ts
AND the row is eligible under the existing deletion bookkeeping rules
```

Predicate FALSE and NULL do not select rows for deletion, consistent with [SQL DELETE predicate semantics](https://duckdb.org/docs/current/sql/statements/delete).

The existing filter bitmap uses **1 = excluded**. The proposed application input uses **1 = selected for deletion**. Normalize this polarity at one explicit boundary, restricted to the captured row range; never mutate a shared expression-cache bitmap or invert twice.

Reuse MVCC primitives, but do not blindly reuse the query node unchanged: current PK-delete application requires `insert_ts < delete_ts`, whereas query visibility may include rows inserted at the query timestamp. The equality boundary must be preserved explicitly.

### 4.2 DeleteApplyNode

`DeleteApplyNode` is a terminal mutation operator, analogous in responsibility to a database delete sink. It:

1. Consumes a validated selection bitmap and the fixed delete timestamp from execution context.
2. Calls the segment's `ApplyDeletesMap` entry without extracting PK values.
3. Reports application outcome and, if needed, newly applied row count through an explicit result contract.

It must not directly manipulate `DeletedRecord` internals, assume a particular PK representation, or acquire mutation privileges through `const_cast` on the existing query context.

### 4.3 ApplyDeletesMap

`ApplyDeletesMap` is the working name for the new **bitmap/offset-based** application entry. Here, “Map” means a selection bitmap aligned with segment row offsets, not a PK map or hash table.

Its conceptual contract is:

```text
ApplyDeletesMap(selected_rows, delete_ts, captured_row_range)

selected_rows[i] == 1 -> apply a delete to segment row i at delete_ts
```

This notation defines behavior, not a finalized C++ signature or ABI. A batch-local bitmap would additionally need an explicit base offset; absolute segment offsets must never be inferred ambiguously.

The segment layer validates row bounds and the execution view, then invokes a shared offset-based helper in `DeletedRecord`. Refactor the existing PK callback and the new bitmap path to share:

- Existing duplicate suppression and timestamp/offset record insertion.
- Growing-mask resizing or Sealed-mask bounds handling.
- Delete counts and memory/resource accounting.
- Latest and historical delete-snapshot maintenance.

The existing `StreamPush` and `LoadPush` remain available. `DeletedRecord` does not parse predicates and does not need a new storage representation, but it does need a small write-entry refactor: calling today's PK-based entry would reintroduce the lookup this design removes.

Snapshot maintenance should be amortized over an application batch, not performed by cloning a full snapshot for every selected row. Exact batching/publication granularity remains an implementation decision subject to the existing visibility contract.

## 5. Growing and Sealed segments

The plan and `DeleteApplyNode` should be segment-type agnostic. Segment-specific behavior stays behind the segment interface and existing `DeletedRecord<is_sealed>` specialization.

| Concern | Growing | Sealed |
| --- | --- | --- |
| Legacy PK lookup | Insert-record PK mapping | Sorted PK column or another supported PK representation |
| New expression path | Predicate result already identifies offsets; no PK lookup | Predicate result already identifies offsets; no PK lookup |
| Row bounds | Capture an active prefix; concurrent inserts must not extend the delete's selected range | Capture a stable row layout and its timestamp view |
| Delete-mask handling | Preserve dynamic capacity and existing resource accounting | Preserve fixed-row bounds and existing resource accounting |

The operator does not need separate Growing/Sealed implementations merely because their legacy PK lookup differs. It still relies on each segment providing correct expression access, timestamp filtering, lifetime safety, and mutation serialization.

## 6. Correctness boundaries

### Stable row identity and timestamp

- A row offset identifies a row in one segment layout; it is not a durable or cross-segment identifier.
- Pin segment lifetime and the relevant data/schema layout throughout evaluation and application. Reopen, replacement, release, or compaction must not silently reinterpret selected offsets.
- Use one fixed `delete_ts` and captured execution view. Growing inserts outside the selected prefix, or with `insert_ts >= delete_ts`, must not be deleted accidentally.
- Preserve the existing write serialization discipline, including interaction with PK deletes and loaded deltas; a read-only query pipeline alone does not provide it.

### DeletedRecord compatibility

- Preserve the current skip-list ordering, row-level duplicate suppression, deletion mask, and resource accounting.
- Preserve timestamp-based historical queries and latest-snapshot queries. Do not replace timestamped records with only a timeless OR-ed bitmap.
- Keep expression-result cache ownership separate from mutable deletion state.
- Preserve current behavior for replayed, duplicate, and already-deleted offsets; this proposal does not introduce a new tombstone-history model.

### Failure and completion

An execution framework does not automatically roll back storage mutation. Cancellation or an exception after applying some rows may leave partial effects unless a separate publication/rollback design prevents them.

Before implementation, specify the permitted application unit, error behavior, and completion/visibility boundary. Do not claim statement-wide atomicity, all-or-nothing segment application, or automatic rollback from the presence of a terminal operator alone.

## 7. Alternatives and expected effect

**Keep PK materialization:** compatible with the current path, but retains result expansion, transfer, and another lookup; retain it as a legacy path rather than the selected native expression path.

**Evaluate locally, extract PKs, then call existing Delete:** reduces external transfer but still materializes PKs and looks them up again; rejected as the target design.

**Apply deletes in the filter operator or an opaque callback:** possible for a prototype, but mixes selection with mutation or hides write semantics; prefer an explicit `DeleteApplyNode` and storage entry.

Expected benefits are elimination of the matching-PK round trip and repeated PK lookup. Expression evaluation, bitmap handling, and per-deleted-row bookkeeping remain; deleting millions of rows is not O(1), and no speedup or resource reduction is yet measured for this proposal.

The [sorted INT64 membership design](design_docs/20260929-sorted-int64-membership.md) and [PR #53901](https://github.com/milvus-io/milvus/pull/53901) optimize predicate evaluation in a different layer. PK-lookup microbenchmarks from the separate Segcore delete-merge experiment likewise do not establish the performance of this new end-to-end pipeline.

## 8. Verification plan

Implementation must establish equivalence with the current delete semantics, not only demonstrate successful execution:

1. Compare predicate selection and post-delete visibility against an independent row/timestamp oracle and the legacy PK path for equivalent inputs.
2. Cover TRUE/FALSE/NULL predicates, zero/all matches, bitmap polarity, range boundaries, and chunk/base-offset boundaries.
3. Cover inserts before, equal to, and after `delete_ts`; repeated PKs, multiple physical versions, and already-deleted rows.
4. Exercise Growing concurrent append and Sealed multi-chunk layouts, both INT64 and VARCHAR PKs, without requiring PK lookup on the new path.
5. Verify historical and latest-snapshot queries before and after application, including interaction with existing PK deletes and loaded deltas.
6. Inject cancellation, application errors, release/reopen races, and concurrent mutation; verify the explicitly chosen failure and publication contract.
7. Measure expression, apply, snapshot, and total time separately; capture CPU, memory, allocation, and concurrent-search latency against the legacy path.
8. For every allowlisted expression/type combination, compare Go compaction selection against Segcore and an independent oracle, including boundary values and NULL behavior where supported. Verify that an unsupported subexpression routes the whole request through the legacy PK path before predicate persistence/application.
9. Validate the drop-field drain contract in Section 9.1.3 with delayed Streaming/L0 work, old growing data, unequal per-channel progress, mixed PK/predicate backlog, segment replacement, and restart between the two drain phases or before metadata publication. Cover an empty L0 inventory while deletes remain in Streaming and mismatched checkpoint/L0 observations. Physical column removal must not precede complete durable coverage and predicate evaluation.
10. For predicate-delta loading, verify index-only, raw-data, and mixed access paths against the same row/timestamp oracle. With suitable exact scalar indexes, check that `c > 18 && a == 18` does not read the corresponding raw columns for predicate evaluation; also cover cold indexes, lazy raw columns, NULL semantics, prefetch, and read/cancellation failures. Measure load/recover latency, storage reads, and peak memory against equivalent PK-delta replay rather than assuming a benefit.

## 9. System-level integration checkpoints

The query-side Segcore change is only the local application mechanism. Compaction, scheduling statistics, and target-QN segment loading require integration before supported predicate deletes can replace expanded PK deletes end to end. Compaction evaluation now favors a restricted Go-layer evaluator; its exact supported set and execution contract remain open. The scheduling-statistics and segment-load discussions have converged on the directions recorded below, without finalized implementation protocols.

### 9.1 Applying predicate deletes during compaction

The current Go compaction path reads delta logs as `(pk, delete_ts)` pairs, builds a PK-to-timestamp map, and filters input rows with that map. It does not have the Segcore expression framework and therefore cannot execute a persisted predicate delete directly. See [compaction/common.go](../../internal/compaction/common.go) and [mix_compactor.go](../../internal/datanode/compactor/mix_compactor.go).

[milvus-storage PR #556](https://github.com/milvus-io/milvus-storage/pull/556) explored a storage-side solution. Its manifest-aware masked reader recognizes predicate delta logs, evaluates SQL predicates with Arrow compute, and can return either a compaction-oriented keep mask or per-row delete timestamps. This would let a Go compactor consume already-filtered batches without embedding an expression engine in Go.

That PR is a closed, unmerged prototype rather than an available dependency. It is historical context, not the current preferred execution location. **The stateless-Segcore extraction approach has been abandoned.**

#### 9.1.1 Preferred direction: a restricted Go-layer predicate evaluator

The current preference is a lightweight evaluator in the Go compaction layer, without moving expression execution into milvus-storage or extracting Segcore's execution framework. This is an architectural direction, not an implemented engine or a claim that the engineering cost is already established.

Mix compaction already reads `storage.Record` batches with Arrow columns, filters PK deletes and TTL through `EntityFilter`, and writes surviving rows. See [mix_compactor.go](../../internal/datanode/compactor/mix_compactor.go), [merge_sort.go](../../internal/datanode/compactor/merge_sort.go), and [entity_filter.go](../../internal/compaction/entity_filter.go). The proposed extension is:

```text
Read batch -> Go predicate evaluation with delete-timestamp eligibility
  -> combine with existing PK-delete / TTL filtering -> write surviving rows
```

This follows the **ClickHouse filter-then-rewrite precedent in Section 2.5**, not its concrete engine. Reusing Milvus's typed expression representation is a candidate way to avoid a second expression language; it does not by itself guarantee identical evaluation semantics. The evaluator need not reproduce ANN execution or query scheduling.

#### 9.1.2 Agreed safety boundary: strict supported set and legacy fallback

- **Default deny for the new path.** Predicate delete supports only an explicitly allowlisted, deliberately restricted expression set whose semantics are stable and verified across query-side application and compaction. The exact initial set is not selected yet.
- **Validate the complete typed expression.** Eligibility covers operators, operand types, NULL behavior, conversions, and composition, not merely an operator-name list. One unsupported node or combination makes the whole expression ineligible; do not partially apply a supported subset.
- **Preserve the existing route.** An expression accepted by today's delete API but unsupported by the new predicate path is evaluated through the existing query path to retrieve matching PKs, followed by ordinary PK deletion. Existing validation and operational limits still apply; fallback does not promise that every large delete will succeed.
- **Choose before side effects.** This is request routing before predicate persistence/application, not a recovery mechanism after a predicate delete has committed or partly executed. A compactor encountering an unsupported persisted predicate must not silently ignore or approximate it; its error/recovery contract remains to be specified.
- **Keep semantic parity as the admission gate.** A broader Segcore expression capability does not automatically expand predicate-delete support. Extending the allowlist requires cross-evaluator validation first.

Timestamp semantics, evaluator errors, version compatibility, and L0 dispatch behavior remain open integration work. Section 9.1.3 records the agreed drop-field safety requirement and preferred drain granularity; its progress and recovery protocol remains to be designed. A narrow expression set reduces the semantic surface; it does not remove those obligations. No detailed operator set, evaluator ABI, or routing implementation is specified here.

#### 9.1.3 Drop-field safety: minimum requirement and checkpoint-style drainage

Discussion checkpoint: **2026-10-19**. The correctness requirement below is agreed; unified drainage is the preferred implementation direction, not an implemented guarantee.

**Minimum correctness requirement.** Before a rewrite physically removes a field, all committed predicate deletes preceding the drop-schema barrier must be completely and durably associated with every applicable target segment's delta logs, or already incorporated into its durable rewritten data. They must not remain exclusively in Streaming buffers or L0. Coverage includes pre-barrier growing data and must remain valid across segment replacement, not merely cover segments already sealed when Drop begins. File upload alone is insufficient: the corresponding segment metadata must also be durably published.

- The barrier belongs to each affected VChannel; ordering comes from its PChannel/WAL. Track each VChannel's own drop boundary and aggregate completion across channels, rather than replacing those boundaries with one maximum timestamp.
- Establish the boundary by entering the dropping phase and rejecting subsequent predicates that reference the removed field. Extend the drop completion gate while keeping schema-evolution admission occupied; do not try to drain an unbounded stream before establishing the boundary.
- Drain all pre-barrier predicates without selecting only those that reference the dropped field. Field-dependency analysis and extra expression metadata are unnecessary for this conservative condition.
- Durable segment delta logs may still contain predicates. In that case, drop-field compaction must evaluate them while the required old column data and internal schema remain available, then remove the column. A prior physical-delete compaction or conversion to PK/offset records is not mandatory. Historical schema metadata alone cannot recover physically discarded column data.

Completing one compaction over currently known deltas is not evidence that this condition holds. QN in-memory application, WAL consumption, and L0 persistence alone are also insufficient. A failure, timeout, or restart must not turn incomplete drainage into successful completion.

**Two required drain phases.** For each affected VChannel, capture its drop-schema WAL boundary as `T[v]` and require both phases for the same fixed pre-barrier delete prefix:

1. **Streaming -> L0:** force progress and wait until all pre-barrier PK and predicate deletes have durable L0 output and completed coordinator metadata registration. WAL append success or consumption through the barrier is not this completion condition. A validated persisted channel checkpoint may provide evidence for this phase; predicate messages must participate in the same retention and publication rules as PK deletes.
2. **L0 -> target segment delta logs:** wait until those deletes have been fully dispatched into every applicable target segment's delta logs and the corresponding metadata has been durably published, or their effects are already incorporated into durable rewritten data. L0 output alone, worker completion without metadata publication, or QN in-memory application is insufficient.

The aggregate gate is:

```text
DropDrainReady = AND over affected VChannels v:
    StreamingToL0Ready(v, T[v]) AND L0ToTargetReady(v, T[v])
```

Only after every channel satisfies both conditions may the drop drain gate release; any rewrite that physically removes the field must honor this gate and evaluate remaining predicate deltas before removing the column. An empty L0 inventory does not prove readiness while older deletes may still be buffered in Streaming. The check covers the entire pre-barrier prefix through both phases, not merely the L0 files visible at one observation. Post-barrier deletes do not extend the fixed prefix being drained.

**Preferred implementation: drain PK and predicate deletes together.** Only predicates introduce the missing-field hazard, so predicate-only drainage is the correctness minimum, not a requirement to build a selective mechanism. Unified drainage is more consistent with checkpoint-style progress: prove that the complete delete prefix through a fixed boundary has reached the required durable destination, instead of maintaining separate completion exceptions for PK and predicate records.

The benefit of distinguishing the two types is limited unless PK backlog demonstrably dominates Drop latency. Selective drainage would need type-specific completion accounting and, if batches or L0 files contain both forms, selective processing or splitting to avoid waiting for PK work. Unified drainage can reuse the batch-oriented approach with one completion condition. Its tradeoff is that PK backlog also enters the DDL wait path; this may move existing background work onto that path rather than increase total work. No latency or resource advantage has been measured.

**Reuse boundary.** The [online schema evolution design](design_docs/20260715-online-schema-evolution.md), especially Sections 7 and 9.1, supplies the staged-drop and admission framework. The current [WAL L0 materializer](design_docs/wal/l0_materializer.md) and [implementation](../../internal/streamingnode/server/wal/vchannel/l0materializer/wal.go) provide ordered per-VChannel batches and durable prefix progress for the first phase, not proof of target-segment delta publication.

There is also an existing `VchannelInfo.DeleteCheckpoint`: [GetQueryVChanPositions](../../internal/datacoord/handler.go) derives it from the minimum start position of the L0 segments in its recovery view, falling back to the channel seek position when none remain. QueryCoord forwards it to [Delegator.SyncTargetVersion](../../internal/querynodev2/delegator/delegator_data.go) for delete-buffer cleanup. The preferred direction is to reuse and strengthen this channel-checkpoint-plus-pending-L0 mechanism rather than introduce an independent checkpoint system.

That existing query-recovery response is not yet an established Drop readiness contract: its partition/recovery filters and separately sampled segment/checkpoint state need a complete, coherent view for this use. [L0 task completion](../../internal/datacoord/compaction_task_l0.go) already updates target deltalog metadata and retires inputs, but the underlying [catalog batch writes](../../internal/metastore/kv/datacoord/kv_catalog.go) require crash/restart verification before asserting atomic end-to-end completion. The two-phase requirement is settled; exact progress representation, ownership, recovery, and handling of replacement segments remain deferred. No current checkpoint value alone is claimed to prove the new predicate-delete Drop guarantee.

### 9.2 Providing delete-impact statistics to MixCoord

Decision status: **architectural direction settled on 2026-10-19**. Use cumulative applied-delete counts from loaded sealed segments, collected through a dedicated segment-stats RPC and independently scheduled polling. Detailed wire contracts and implementation remain deferred.

Current single-compaction scheduling uses `Statistics.DeleteNumRows / Segment.NumOfRows`, together with delta-log count and size. `DeleteNumRows` is the sum of `EntriesNum` across delta binlogs, so it is an approximate signal even for PK deletes rather than a count of unique physical rows. See [data_coord.proto](../../pkg/proto/data_coord.proto) and [compaction_trigger.go](../../internal/datacoord/compaction_trigger.go).

For a predicate delta log, one `(expr, delete_ts)` entry may match zero rows or most of the segment. Recording `EntriesNum = 1` therefore makes the current ratio unusable: MixCoord cannot distinguish a negligible delete from an 80% delete without evaluating the predicate or receiving a materialized per-segment impact statistic.

Compaction-time evaluation alone cannot solve scheduling because the result arrives after MixCoord has already decided to compact. The system needs a pre-compaction signal about deleted rows per segment. The agreed direction below uses cumulative observations of applied deletes, not a persistent ledger of per-delete increments.

#### 9.2.1 Agreed direction: collect cumulative counts from loaded sealed segments

MixCoord collects the actual cumulative deleted-row count of each loaded sealed segment from the query side and uses it as a reconstructible compaction-scheduling hint.

```text
Committed WAL delete
  -> query-side segment applies the delete
  -> DeletedRecord contains the applied, deduplicated deletion state
  -> expose the segment's cumulative deleted-row count through a dedicated stats RPC
  -> MixCoord collects stats on an independent polling schedule
  -> MixCoord updates its scheduling-statistics cache
```

- **Report state, not event increments.** Each accepted report replaces the previous observation for that segment; it is not added to it. A later cumulative report covers missed intermediate reports. This avoids maintaining per-delete increment histories and exactly-once increment accounting at MixCoord.
- **Count unique physical rows.** The source is actual segment deletion bookkeeping, deduplicated by row offset, not the expression's raw match count, the number of delete messages, or the number of distinct PK values. Row-level deduplication remains necessary even though the coordinator no longer merges event deltas.
- **Do not sum query replicas.** Replicas report observations of the same logical segment, not independent deleted-row contributions. Source selection, freshness, and segment-incarnation validation still need a small reporting contract; arrival order alone does not establish which observation is newest.
- **Keep scheduling off the commit path.** Delete completion does not wait for a stats report, stats-specific etcd/S3 persistence, or all query replicas to apply. The observation comes from actual application of committed deletes, so MixCoord does not need a separate proof that a precomputed count corresponds to a committed delete.
- **Rebuild rather than persist this stats cache.** After MixCoord restart, recollect from currently loaded segments. Missing observations temporarily reduce scheduling precision; they do not invalidate deletion state. No new per-delete stats files, pending/ready markers, or durable stats ledger are required by this direction. Existing WAL and segment metadata durability are unchanged.
- **Separate statistics from distribution.** Count changes do not mark segment distribution dirty or advance its change version. The stats collector has its own refresh schedule and cache-update semantics; transport connections and node discovery may still be shared.

The distinction is **valid but potentially stale**, not **prepared but possibly uncommitted**: a count describes the segment's applied deletion state at observation time, not necessarily the latest WAL state. Statistics must not decide which deletes compaction applies; compaction must independently honor authoritative persisted deletes.

#### 9.2.2 Existing implementation support and loaded-data boundary

The local code provides a basis for the count, but not the complete reporting feature:

- `ChunkedSegmentSealedImpl::get_deleted_count()` returns `deleted_record_.size()`, and `DeletedRecord::InternalPush` suppresses repeated deletion of an already-marked row offset. The new offset-based expression path must preserve this bookkeeping. See [ChunkedSegmentSealedImpl.cpp](../../internal/core/src/segcore/ChunkedSegmentSealedImpl.cpp) and [DeletedRecord.h](../../internal/core/src/segcore/DeletedRecord.h).
- Current QN `GetSegmentInfo` reports `NumRows = segment.InsertCount()`; that field is not a deleted-row count. `SegmentVersionInfo` also has no deleted-count field. Reporting needs to be extended rather than assumed to exist. See [services.go](../../internal/querynodev2/services.go) and [query_coord.proto](../../pkg/proto/query_coord.proto).
- Current general predicates such as `age < 18` and `id_pk % 5 == 1` take `complexDelete -> QueryStream -> matching PKs`, requiring loaded, queryable target data. Pure PK membership or equality can bypass the query and do not impose the same load requirement. See [task_delete.go](../../internal/proxy/dml/task_delete.go) and the load gate in [querycoordv2/utils/util.go](../../internal/querycoordv2/utils/util.go).

Retaining the existing loaded-data prerequisite for general predicate deletes means this statistics source does **not** further restrict their current deletion capability. It does not imply that a collection remains loaded after a delete: release, restart, or loading only some partitions can leave statistics unavailable or stale. Unknown is not zero, and this direction does not claim current exact counts for unloaded segments or offline PK deletes.

Counts are segment-local: statistics for a dropped or compacted-away source segment cease to describe an active scheduling candidate. A replacement segment is observed from its own applied state, not assigned the source segment's deleted count. Ordinary cache invalidation remains necessary; the per-delete file lifecycle is what disappears.

#### 9.2.3 Alternatives considered and set aside

The following records the discussion history, not additional requirements for the selected direction:

| Alternative | Why it is not selected for the current direction |
| --- | --- |
| Count each expression before committing it, then persist per-segment increments in etcd | Needs overlap handling, event deduplication, and association with a successful delete; synchronous persistence couples scheduling metadata to delete latency, while asynchronous persistence needs loss recovery |
| Save prepared count sets to S3 and publish them after WAL commit | Avoids a large metastore payload, but introduces prepared-object retention, commit-to-file association, publication recovery, and safe orphan cleanup |
| Use a durable broadcast/job workflow to coordinate that publication | Offers useful recovery mechanisms, but adds job-like coordination to ordinary predicate deletes solely to make scheduling statistics durable |

For the S3 alternative, the discussion required channel-independent commit/publication, stable VChannel identity rather than a Delegator or replica identity, and only a small stats reference in WAL rather than a large `{segment -> count}` table. A timeout alone could not authorize deletion of a prepared file whose delete might still commit. These requirements explain its complexity; they are not carried into the selected reporting approach.

The user-provided scale target is **1 million collections and 100 million segments**. Avoid retaining one count set per delete or assuming a dense coordinator-resident table for every segment. The current preference is bounded state for the observed loaded working set; its memory and reporting costs still need measurement. Earlier compressed-set size estimates were illustrative, not a validated capacity model.

#### 9.2.4 Industry precedent and applicability

PostgreSQL exposes estimated dead-row counts, allows cumulative statistics to lag, and resets counters after an unclean shutdown. Its documentation warns that resetting counters used by autovacuum can delay maintenance and cause bloat. See [PostgreSQL cumulative statistics](https://www.postgresql.org/docs/current/monitoring-stats.html).

- **Adopt:** maintenance scheduling can use derived statistics without making those statistics the source of deletion truth.
- **Adapt:** use Milvus's applied, offset-deduplicated segment count, collected from loaded query-side state; it is exact for the observed applied state rather than a raw predicate estimate.
- **Do not copy:** PostgreSQL's storage, persistence, or vacuum protocols. Its behavior does not establish that loaded-only collection or Milvus's report-selection rules are an industry standard.

TiKV's Region heartbeat carries both region metadata and runtime statistics, while its Region epoch represents peer-configuration and split/merge changes. This demonstrates that sharing transport does not require sharing change semantics. See [RegionHeartbeatRequest](https://github.com/pingcap/kvproto/blob/master/proto/pdpb.proto) and [RegionEpoch](https://github.com/pingcap/kvproto/blob/master/proto/metapb.proto). Adopt that distinction, not its Raft protocol or a requirement to combine messages. The dedicated RPC selected here is a Milvus-specific architectural choice, not a claim that different refresh frequencies inherently require separate RPC methods.

#### 9.2.5 Settled collection interface boundary

Use a **dedicated segment-stats RPC from MixCoord to QN**, with an independent polling loop. Return cumulative observations of loaded sealed segments; this is not a new message emitted for every delete operation.

The interface boundary is:

- Distribution describes segment placement, loaded copies, and associated load state. Segment stats describe applied deletion counts and the freshness of those observations.
- Stats refresh must not depend on `GetDataDistribution` reporting a topology/load change. A count-only change must not rebuild distribution indexes or advance the distribution change version.
- Collection cadence, batch size, timeout, and retry behavior can be controlled independently from distribution polling. Their values are not selected yet. Separate RPCs alone do not guarantee resource isolation; the implementation must honor this scheduling separation.
- Existing QN node discovery and client connections may be reused. A dedicated RPC does not require separate network connections, a Proxy relay, or a Delegator aggregation hop.
- MixCoord exposes accepted observations to its compaction scheduler as derived in-memory statistics. Placement information may help validate segment identity/liveness without making stats updates distribution mutations.

The code investigation identified reusable infrastructure and precedents, not a ready-made deleted-count feature:

- QueryCoord already polls QNs with `GetDataDistribution` and maintains in-memory segment observations. Its full/delta reporting is useful implementation context. See [dist_handler.go](../../internal/querycoordv2/dist/dist_handler.go).
- MixCoord's `GetLoadSegmentInfo` delegates to QueryCoord's distribution cache, and DataCoord's force-merge policy already obtains replica and node information through MixCoord. These establish a cross-module consumption precedent, not a reason to store deleted counts in distribution. See [mix_coord.go](../../internal/coordinator/mix_coord.go), [QueryCoord services.go](../../internal/querycoordv2/services.go), and [compaction_policy_forcemerge.go](../../internal/datacoord/compaction_policy_forcemerge.go).
- Directly adding a deleted-count field to the current distribution response would be insufficient: ordinary deletes do not mark that report dirty, and forcing them through its current update path would rebuild node segment indexes and change the distribution version. See [data_distribution_report.go](../../internal/querynodev2/data_distribution_report.go), [QN services.go](../../internal/querynodev2/services.go), and [segment_dist_manager.go](../../internal/querycoordv2/meta/segment_dist_manager.go).

The earlier suggestion to prefer extending `GetDataDistribution` is **superseded**. Sharing one heartbeat with separate stats and distribution sections is technically possible, including different payload refresh frequencies, but is not the selected approach: stats have different consumers, freshness requirements, and scaling costs, so their collection should not depend on the distribution poll's timing or response processing.

#### 9.2.6 Deferred details; no detailed design yet

Remaining work is to define the RPC name and request/response fields, batching and refresh parameters, collector ownership inside MixCoord, source/progress validation across replicas and reloads, cache bounds, and fallback scheduling when counts are absent or stale. The dedicated-RPC and independent-polling choice is settled; timestamp watermark semantics and fallback policy are not. Statistics loss may delay compaction and space reclamation; it must not cause missed delete application or incorrect query visibility.

This checkpoint settles the **statistics source, consistency model, and collection-interface boundary**, not the compaction-side expression evaluator in Section 9.1 or durable expression replay. No code implementation is authorized or implied by settling this part of the design.

### 9.3 Target-QN load phase: reuse expression access paths

Decision status: **architectural direction settled on 2026-10-19; not implemented**.

**Scope.** This section covers the target QN loading segment data/index resources and applying the segment's own delta logs. Delegator L0 forwarding, streaming-tail catch-up, and publication coordination remain external integration boundaries; they are not redesigned here.

**Agreed execution policy.** The loader recognizes persisted predicate records and invokes the existing Segcore expression execution framework, followed by delete-specific timestamp/visibility filtering and `ApplyDeletesMap`. Preserve each record's original `delete_ts` and load/replay bookkeeping. Do not expand matches into PKs or require a separate eager load of all fields referenced by the predicate before evaluating it.

- **Use suitable scalar indexes.** When the existing expression path can answer the predicate exactly through scalar indexes, including required NULL semantics, use their result bitmaps without reading the corresponding raw columns for predicate evaluation. For example, `c > 18 && a == 18` can combine the two indexes' result bitmaps with AND when both operations are supported.
- **Otherwise access required data on demand.** Reuse existing raw-column access, lazy materialization, chunk access, and prefetch behavior. Mixed index/raw execution is valid. Do not change a field's lazy-load configuration solely because it participates in a predicate delta.
- **Keep dependencies explicit.** This is independence from preloading raw column contents, not independence from the segment's schema, stable row layout, index/data availability, or MVCC timestamp information. Indexes themselves may require I/O, and candidate-only indexes may still require raw-data rechecks.
- **Finish application within load.** Complete the segment's predicate-delta application before this target-QN load stage reports success. Do not defer deletion correctness to the first user query. This reconstructs in-memory deletion state; it does not replace the durable drain requirement in Section 9.1.3.

**Existing implementation basis.** [segment_loader.go](../../internal/querynodev2/segments/segment_loader.go) loads segment resources and then its delta logs. [UnaryExpr.cpp](../../internal/core/src/exec/expression/UnaryExpr.cpp) selects scalar-index or raw-data evaluation, while [ConjunctExpr.cpp](../../internal/core/src/exec/expression/ConjunctExpr.cpp) combines child results. [Expr.h](../../internal/core/src/exec/expression/Expr.h) gates raw-data prefetch on the selected execution path. [ChunkedColumnGroup.h](../../internal/core/src/mmap/ChunkedColumnGroup.h) provides lazy reader/cache initialization and chunk access. These are reusable capabilities, not an already implemented predicate-delta loader.

**Industry precedent.** PostgreSQL's [multiple-index bitmap combination](https://www.postgresql.org/docs/current/indexes-bitmap-scans.html) supports the mature principle of combining per-index row selections. Adopt that principle and adapt it to Milvus's segment-local bitmaps and timestamped deletion state; do not copy PostgreSQL's heap-access or transaction workflow, or infer Milvus performance from it.

**Latency and memory boundary.** Avoiding an additional forced eager-load stage neither disables existing prefetch nor guarantees bounded peak memory: the reused raw-data path may prefetch many chunks, and a predicate requiring a full cold-column scan moves that I/O onto load/recover latency. Compare expression evaluation plus delete bookkeeping against PK-delta reading, decoding, PK-to-offset lookup, and the same bookkeeping. Index-only execution can avoid predicate-column reads, but no load-time speedup or memory reduction has been measured. Resource budgeting and exact loader/CGo wiring remain implementation work; the access-path policy above is settled.

## 10. Deferred integration and open contracts

This document does not fully specify WAL payloads, deterministic replay, L0 persistence and catch-up during segment load, release/reload behavior, CDC, compaction execution, or replica-wide visibility. Section 9.1.3 settles the drop-field safety requirement, both mandatory drain phases, and the all-VChannel release condition, while preferring unified drainage and reuse of existing checkpoint/L0 progress. Section 9.3 settles target-QN predicate-delta loading's access-path policy: reuse existing expression execution without an additional forced eager load. The detailed end-to-end drain protocol and loader integration remain open. These remain required integration work, not benefits already delivered by local bitmap application. Support for accepting general predicate deletes against unloaded data is deferred; it is not implied by the statistics proposal.

Before coding this part, finalize:

- The delete plan/context and write-capable segment handle, including the CGo boundary.
- Bitmap ownership, whole-segment versus batch-local input, and the applied-row-count definition.
- Mutation serialization, snapshot publication granularity, and cancellation/partial-application behavior.
- The restricted Go compaction evaluator's exact supported expression/type set, parity tests, and schema/version contract; preserve the agreed legacy fallback for expressions outside that set.
- The per-VChannel completion proof for both Streaming -> L0 and L0 -> target segment delta-log publication, including coherent checkpoint/L0 observations, old growing data, replacement segments, retry/restart, and gating every rewrite that can remove the field; preserve the two-phase requirement and unified-drain preference in Section 9.1.3.
- The dedicated stats RPC's detailed contract, source/freshness validation, cache invalidation, and missing-statistics behavior for the cumulative loaded-segment counts selected in Section 9.2; preserve independent polling and do not reintroduce per-delete durable statistics as an implicit requirement.
- The predicate-delta loader/CGo wiring, replay bookkeeping, failure handling, and resource budgeting; preserve the settled access-path policy and target-QN-only scope in Section 9.3. Delegator L0/streaming catch-up remains a separate integration contract.

No implementation or performance validation accompanies this draft.
