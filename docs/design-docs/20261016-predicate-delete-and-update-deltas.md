# Predicate Deletes and Delta-Based Updates

Date: 2026-10-16

Discussion checkpoint: 2026-10-22

Status: Draft — predicate-delete data flow, progress-bound best-effort statistics, predicate-backlog signals, asynchronous recovery direction, shared logical mutation-message structure, single-mutation request boundary, TimeTick-preserving buffer/file ordering, VChannel delta-prefix publication and recovery contract, independently batched complete and ordered L0 dispatch, shared ordered batch delta execution with safe temporal bypass, separate L0 dispatch, Delta Consolidation, and full-rewrite maintenance, bounded assignment-expression scope, constant-assignment update semantics, and snapshot-aware base-index/delta query merging agreed; detailed implementation protocols remain open; not implemented

Scope: Joint design for predicate deletes and logical in-place updates through deltas, including segment application, ordering, persistence, load/recovery, compaction, and schema safety. Predicate-delete decisions are more developed; update benefits and detailed mechanisms still require evaluation.

**Critical implementation-design item: cross-type delta publication atomicity.** The complete-prefix requirement, independent VChannel-level publication interface, strict contiguous range extension, `RangeApplied`/`RangeMismatch` results, and recovery-first prefix lookup are agreed. Durable metadata layout, atomic commit mechanics, consumer acquisition, and ownership fencing still require detailed design and verification. Treat Sections 9.7.2–9.7.4 as a correctness-critical workstream, not a minor writer extension or an implemented guarantee.

## 1. Goal and agreed decisions

Reduce mutation amplification without changing the observable semantics of committed operations. Predicate deletes avoid expanding matching rows into PK lists; delta-based updates aim to avoid reconstructing and reinserting an entire row when only some fields change. The two features must be designed together because an update can change whether a subsequent delete predicate matches.

### 1.1 Predicate-delete direction

Execute a delete predicate on each target segment and apply its matching row offsets directly to `DeletedRecord`, without materializing primary keys for a round trip through Proxy or looking those keys up again during application.

The agreed direction is:

- Delegator receives an expression plan and a delete timestamp, rather than an expanded list of matching PKs, for this new path.
- Reuse expression execution to produce a segment-local selection bitmap; predicates may reference any supported scalar field, such as `age < 18`.
- Add a dedicated `DeleteApplyNode` that consumes the selection and calls a segment-level `ApplyDeletesMap` entry.
- Use the same logical pipeline for Growing and Sealed segments.
- Preserve the current `DeletedRecord` representation and query algorithms; add or refactor its write entry to accept selected offsets directly.
- Retain existing PK-based delete and delta-load entries for compatibility.
- Restrict the native predicate-delete path to a deliberately small, explicitly supported expression set with stable, verified semantics across query execution and compaction. Expressions outside that set retain the existing expression-query-to-PK-delete path; do not approximate them.
- For a supported predicate, append the mutation without a matching-PK query, a count pre-query, or ingress segment-related statistics. Select the path by expression support, not estimated match count.
- Before physical field removal, require complete durable coverage of pre-barrier predicate deletes in the affected segments' delta logs or rewritten data. Prefer checkpoint-style drainage of both PK and predicate deletes rather than separate type-specific drain progress; see Section 9.1.3.
- During target-QN segment loading, apply predicate deltas through the existing Segcore expression execution paths: use suitable scalar indexes without reading the corresponding raw columns, otherwise read required data on demand. Do not introduce an additional forced eager-load step; see Section 9.3.
- Separate base load completion from background mutation application for both `lazy-load=false` and `lazy-load=true` in the proposed recovery architecture. Gate each query by its required `GuaranteeTs` and the relevant complete apply progress, rather than waiting for the entire backlog before admitting any query. This does not waive the request's consistency requirement; see Sections 9.3.1–9.3.3.

The local pipeline is not a complete replacement for the distributed delete protocol. Durable ordering, replay, compaction, and release/reload behavior must be designed before production integration. The current direction retains the existing loaded-data prerequisite for general predicate deletes; extending them to unloaded data is not an initial requirement.

### 1.2 Update-delta direction and shared boundary

- **Logical in-place update, not object overwrite.** Apply changes to existing segment rows through update deltas; do not overwrite the original S3 data files on each foreground update. Preserve base data, persist changed fields plus necessary identity/version metadata, and merge through later maintenance.
- **Co-design the mutation lifecycle.** Online application, load/recovery, and compaction must produce results equivalent to the same VChannel's committed logical mutation order and timestamp visibility. Predicate evaluation must see earlier visible updates; see Section 6.
- **Share contracts, not necessarily representations.** Delete changes row visibility; update changes field values. This agreement does not put field updates into `DeletedRecord`, select an update operator, or require identical delta formats/RPCs.
- **Separate row selection from column assignment.** One update may select multiple rows through a supported `WHERE` predicate and assign constants to multiple columns through one `SET` patch. Initial scope excludes assignments that read old field values; see Section 10.4.
- **Share the logical mutation-message structure.** Predicate delete and update reuse the WAL envelope and common collection/schema identity, partition scope, and typed predicate. Update additionally carries field assignments; neither carries an ingress matched-PK list or segment statistics. These are logical contracts, not finalized protobuf definitions; see Section 9.5.
- **Keep one logical mutation per request.** SDK, Proxy, and StreamingNode accept one native UPDATE or predicate DELETE per logical request, represented by one message per target VChannel, not a batch of independent mutation statements. One statement may still match many rows and assign multiple columns. User-controlled DML transactions are outside this design; existing internal write transactions remain separate infrastructure; see Section 9.5.2.
- **Preserve ordered commands through buffering and persistence.** The persistence buffer does not evaluate predicates or collapse intermediate updates. Update and predicate-delete deltas use separate file types, retaining each record's effective mutation TimeTick for ordered merge during load and compaction; see Section 9.7.
- **Share ordered batch delta execution across data rewrites.** Sort, Mix, full-rewrite bump-schema-version, and clustering compaction use one logical `DeltaExecutor` to apply a TimeTick-ordered sequence of PK deletes, predicate deletes, and updates to each applicable batch. Skip delta execution only when the batch cannot be affected, not merely because row-write and delta-time intervals do not overlap; see Section 9.1.4.
- **Separate dispatch, Delta Consolidation, and full rewrite.** L0 dispatch distributes deltas without expression evaluation; Delta Consolidation is a distinct background column-maintenance process, not segment-merging compaction. It rewrites selected columns and switches their manifest mappings while preserving segment identity, row count/order/offsets, and unchanged columns and indexes. A column can be split out of a shared group without rewriting the group's other columns; its old bytes become logically hidden in the new manifest, not physically reclaimed. Normal Mix/full-rewrite scheduling remains separate, with no extra trigger solely from delta accumulation. Phase one excludes vector updates, preserving unchanged vector data and ANN indexes; physical removal of deleted rows remains a full-rewrite responsibility. See Section 9.1.7.
- **Coordinate all delta types at publication.** Reuse PK-delete sync trigger categories, but use a common per-VChannel sync cut across PK deletes, predicate deletes, and updates. Atomically advance `delta_prefix_barrier[v]` only over a complete, durably published prefix; L0 delta compaction must not consume independently visible partial files. Sparse types must also sync through the cut; small-file amplification is an explicit cost; see Sections 9.7.1–9.7.2.
- **Publish ranges through a dedicated VChannel interface.** Keep the L0 segment storage representation, but introduce a separate `PublishDeltaPrefix` contract rather than broaden the single-segment `SaveBinlogPaths` RPC. New ranges must extend the current prefix without gaps or overlap; already-covered ranges return `RangeApplied` without republishing. SN recovery reads each VChannel's published prefix from MixCoord before resuming materialization. These are agreed contracts, not implemented APIs; see Sections 9.7.3–9.7.4.
- **Keep assignment capability deliberately small.** Represent an assignment as `field_id + value_expr`, initially admitting only typed literals, including explicit NULL for supported nullable fields. The long-term core adds same-row field references and basic arithmetic; a few deterministic built-ins may be added for demonstrated needs. Full SQL expression parity is not a goal; see Sections 2.10 and 10.8.
- **Separate assignment evaluation without duplicating a complete execution framework.** Delete and update share the WHERE contract; update alone computes assigned values and applies field patches. Phase-one constants require only filling selected rows, not a general value-expression executor. Reuse suitable type and field-access infrastructure while keeping WHERE and SET capability allowlists independent; see Section 9.6.1.
- **Preserve field-level patch semantics under concurrency.** Independent patches to different columns retain both changes; assignments to the same row and column resolve in committed logical mutation order. Predicate evaluation and patch application must observe earlier applicable mutations consistently across online execution, load/recovery, and compaction; see Section 10.5.
- **Keep versioned field values and correct base filtering results.** Reuse the base scalar index or raw chunks, remove old matches for rows with snapshot-visible updates to the referenced field, and supplement matches evaluated against their latest visible values. Correct value-dependent evaluation before logical combination or candidate pruning, including the WHERE of subsequent mutations. Initially, scan updated-field values without requiring a separate delta index; see Sections 10.6–10.7 and the industry comparison in Section 2.9.
- **Design update-version storage around query access.** Locate a field, access updated rows by offset window or discrete selection, and resolve each cell's latest visible version. Ordered versions support predecessor lookup; frequent appends must avoid repeated whole-history relocation. Field/offset containers, segmented-growth mechanics, and concurrent publication are not settled; see Section 10.9. NULL handling remains a separate detailed-design topic.
- **Bound version retention by the serving generation.** Preserve the old segment's supported multiversion visibility while it serves queries and remains pinned by readers. A full rewrite establishes a new baseline without retaining the intermediate history it has folded; do not introduce UNDO or long-term time travel. Safe handoff and reclamation remain required; see Section 9.1.5.
- **Keep compaction execution separate from query-side recovery.** Reuse channel-level exclusion between L0 dispatch and data rewrites. A full rewrite consumes its fixed input deltas into the output base, without carrying those consumed records forward as pending deltas. This does not authorize an update-only Delta Consolidation operation to discard outstanding deletes. QN load/handoff is a separate stage; see Sections 9.1.6–9.1.7.
- **Evaluate benefits before detailed update design.** Section 10 records opportunities and tradeoffs, not measured improvements or a settled implementation. Exact field/type and predicate support, physical row targeting, index-overlay implementation, concurrency control, and update-delta representation remain open.

Sections 4–5 and 9 retain the established predicate-delete directions. Their choices are not automatically extended to updates: for example, a cumulative deleted-row count is not an update-compaction cost model, and the settled delete drain is not yet a complete update/schema protocol.

### 1.3 Shared predicate-delta dispatch direction

- **Use coarse scope, not an ingress-time matched-segment table.** Persist predicate mutations with their VChannel/partition scope and logical timestamp; choose candidate segments from current storage metadata when L0 dispatch runs, not from the QueryView seen by the original request.
- **Keep L0 lightweight.** L0 dispatch does not execute predicates, read business columns to discover matches, or depend on the collection remaining loaded. Actual evaluation belongs to online/load execution and background data compaction.
- **Decouple publication batches from dispatch batches.** SN's published prefix bounds what MixCoord can consume, not how it must batch dispatch. MixCoord may split or combine published ranges while preserving complete cross-type intervals, ordered downstream visibility, and coverage of every applicable target. Published progress and completed-dispatch progress are distinct; see Section 9.4.5.
- **Accept conservative fanout.** Without PK expansion or predicate evaluation, exact matched-segment dispatch is not promised. Use channel/partition and valid data-time bounds; retain candidates that cannot safely be excluded.
- **Reject a separate ancestry-repair mechanism.** Do not persist a request-time segment list and chase its descendants across compactions. After dispatch, predicate deltas become segment-owned inputs to subsequent data compaction. Durable publication, ordering, and completion still require verification; see Section 9.4.

### 1.4 End-to-end predicate-delete data flow

Discussion checkpoint: **2026-10-20**. This consolidates the agreed responsibility split, not an implemented protocol. The industry principles and their limits are recorded in Sections 2, 9.2.4, and 9.3.1; the following composition is Milvus-specific.

```text
Proxy: validate supported predicate and scope; no matching-PK/count query
  -> StreamingClient -> StreamingNode -> committed VChannel WAL mutation
       |-> online consumer / Delegator delete buffer
       |     -> scoped serving Growing and Sealed segments
       |     -> local expression evaluation -> ApplyDeletesMap
       |
       `-> persistence consumer / L0 materializer
             -> durable L0 files + complete cross-type prefix publication
             -> lightweight scope-based dispatch using current storage metadata
             -> segment-owned predicate delta files + published metadata
                  |-> DataNode data compaction: evaluate and rewrite
                  `-> QN load/recover: base loading + background ordered apply
                        -> per-query admission by complete apply progress
```

1. **Admission and WAL append.** Proxy validates the typed expression, schema, and operation scope, then submits a supported predicate directly to WAL without retrieving PKs or collecting segment statistics. Unsupported expressions retain the legacy PK path. The initial general-predicate admission prerequisite remains loaded data; this is not a requirement that the collection stay loaded afterward. Commitment and ordering are per relevant VChannel, not a new collection-wide atomicity guarantee.
2. **Online consumption and application.** The query-side consumer retains predicates in the Delegator's delete buffer for catch-up and forwards applicable records to serving segments in the relevant QueryView, including local Growing data on SN and Sealed data on target QNs. Segment-local execution evaluates the predicate and applies selected offsets. Online forwarding is scoped, not a fanout to unrelated collections or partitions.
3. **Durable L0 production.** The persistence consumer materializes committed predicate deletes into L0 output. With updates enabled, Section 9.7 requires the dedicated VChannel-level publication of a complete prefix across PK-delete, predicate-delete, and update deltas before L0 compaction may consume that prefix. SN recovery first reads the durable published prefix from MixCoord and does not rematerialize covered ranges. This and the online delete buffer are separate responsibilities: the current `deleteBuffer.Put` path must not be described as the owner that flushes L0 files. Extending the existing WAL L0 materializer and implementing cross-type publication remain implementation work.
4. **L0-to-segment dispatch.** Predicate L0 dispatch selects candidate Sealed segments by current storage metadata, channel/partition scope, and safe logical data-time bounds. “Candidate” does not mean the expression is known to match. Persist a segment-owned predicate delta file type that can batch multiple records, with discoverable segment metadata; do not create a request-time segment association table or evaluate expressions in L0. Dispatch batches need not align with SN publication batches, but must preserve the complete intervals, ordered visibility, and target coverage in Section 9.4.5. Publication must complete before the corresponding L0 input can be retired. This storage operation is not a second online broadcast to all already-loaded segments.
5. **Scheduling and offline compaction.** MixCoord does not need to know a predicate's match count to schedule work. Accurate applied-delete counts, when available, are progress-bound best-effort hints; outstanding predicate counts describe replay burden rather than deleted rows. Section 9.1.7 separates update-driven Delta Consolidation maintenance from normally triggered Mix/full rewrites; backlog alone is not a new Mix trigger. DataNode reads durable inputs and evaluates predicates without a loaded-QN dependency. Executing 100 predicates and deleting zero rows during a scheduled full rewrite is an accepted outcome: incorporating those records into durable output can still remove future replay work.
6. **Load and recovery.** For either lazy-load setting, separate base loading from ordered predicate application; overlap work where initialized resources permit. Base loading may return without waiting for full predicate catch-up, but it does not make every query immediately serviceable. Each query must satisfy the complete relevant apply-progress gate in Section 9.3. Suitable scalar indexes avoid raw-column reads; otherwise reuse on-demand access without an extra forced eager-load stage.

**Shared safety boundary.** Online application, recovery, and compaction must preserve the same committed logical mutation order, including updates that change later predicate matches. WAL commitment, QN apply progress, durable L0 completion, and durable target-segment coverage are different milestones. Stats and read-side progress cannot replace the two durable drain phases required before physical field removal. Base-load completion alone does not guarantee Strong-read readiness, a rolling-upgrade deadline, or availability targets.

**Current-code cross-check.** [Delegator delete consumption](../../internal/querynodev2/delegator/delegator_data.go) currently buffers PK deletes and forwards them, while the separate [WAL L0 materializer](../../internal/streamingnode/server/wal/vchannel/l0materializer/wal.go) retains delete handles and produces durable L0 work. The current [segment loader](../../internal/querynodev2/segments/segment_loader.go) loads resources followed by delta logs. These are reuse boundaries, not evidence that the proposed predicate flow or asynchronous query gate already exists.

## 2. Motivation and industry precedent

The existing expression-delete path retrieves matching PKs before applying PK-based deletes. At very high match counts, PK materialization, result-size limits, transfer, and repeated PK-to-offset lookup can become substantial costs.

### 2.1 Shared convention and comparison scope

The mature shared pattern is **evaluate a condition -> identify local rows -> mutate through a storage-owned entry**. An explicit SQL mutation operator is one realization; a segment-local query-delete loop is another.

The delete comparisons below inspect fixed release tags. Section 2.7 separately identifies Kudu's versioned operational documentation and moving-source references; Section 2.9 compares production base-plus-change query mechanisms, and Section 2.10 compares assignment interfaces and their capability boundaries. These establish implementation precedent, not comparative performance or proof of Milvus correctness.

| Reference | Why it is relevant | Main difference from this proposal |
| --- | --- | --- |
| DuckDB v1.4.0 | Closest executor/sink decomposition | Consumes row-ID vectors, not Milvus bitmaps |
| Lucene 9.12.0 | Closest segment-local delete-by-query application | Its shared index-engine read/write model does not establish Milvus's disaggregated L0 dispatch protocol |
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

**Distributed applicability limit:** Lucene's writer and segment readers operate within one index engine, unlike Milvus's separated online query and background storage paths. This precedent supports local evaluation, not routing through loaded QNs, requiring a loaded collection for L0 dispatch, or maintaining query-derived segment associations across compaction.

### 2.4 PostgreSQL: borrow the boundary, not heap transaction semantics

```text
Selection subplan -> row-locating information
  -> ModifyTable / ExecDelete -> table_tuple_delete
```

For ordinary tables, `ExecDeleteAct` invokes `table_tuple_delete` with a tuple identifier and snapshot information. `ExecDelete` handles outcomes such as concurrent update/deletion, including rechecks or serialization errors where required. See [nodeModifyTable.c, REL_17_0](https://github.com/postgres/postgres/blob/REL_17_0/src/backend/executor/nodeModifyTable.c).

**Adopt:** selection produces row identity, while mutation explicitly handles storage outcomes. This supports an error/completion contract for `DeleteApplyNode`, rather than assuming successful filtering guarantees successful application.

**Do not copy:** heap TIDs, row locks, triggers, or EvalPlanQual into Segcore. PostgreSQL uses a generic modifying operator, so it supports the responsibility boundary, not the necessity of a dedicated operator class or bitmap ABI.

PostgreSQL hot standby provides a separate recovery precedent: a readable consistent state need not be fully caught up with the primary. Section 9.3.1 records that behavior and distinguishes it from Milvus's proposed per-request progress gate.

### 2.5 ClickHouse: borrow ordered filtering and field transformation during rewriting

**Primary precedent for the compaction direction in Section 9.1.** ClickHouse's `ALTER TABLE ... DELETE WHERE` rewrites affected data parts. See the [official DELETE mutation documentation](https://clickhouse.com/docs/reference/statements/alter/delete).

```text
Delete mutation predicate -> retaining-row filter
  -> FilterStep in the mutation execution pipeline -> rewritten data part
```

In the fixed release, `MutationsInterpreter` processes mutation commands into stages, wraps a DELETE predicate with `isZeroOrNull` to retain non-matching or NULL rows, and builds conditional column expressions for UPDATE. `addStreamsForLaterStages` adds filtering and expression steps for the prepared stages. See [MutationsInterpreter.cpp, v25.8.1.5101-lts](https://github.com/ClickHouse/ClickHouse/blob/v25.8.1.5101-lts/src/Interpreters/MutationsInterpreter.cpp).

**Adopt:** compose ordered filtering and field-value transformation in a rewrite pipeline before writing surviving rows. Section 9.1.4 applies this decomposition to the shared batch executor.

**Adapt:** evaluate a restricted Milvus expression set over the Go compactor's record batches, preserving Milvus timestamp and schema semantics.

**Do not copy:** ClickHouse's mutation ordering, transaction behavior, expression language, or full execution framework. A separate Go evaluator and strict allowlist are Milvus-specific choices, not an industry-standard consequence of this precedent.

**Evidence boundary:** this is direct implementation evidence for mutation stages during rewriting. It neither proves Go/Segcore semantic equivalence nor establishes a performance benefit for Milvus; divergence could retain deleted rows or physically remove live rows.

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

### 2.7 Kudu: borrow base-plus-delta updates, not its tablet implementation

**Primary precedent for the update direction.** A Kudu Tablet contains multiple RowSets. For this comparison, a **DiskRowSet is analogous to a Milvus Sealed Segment**, not to a whole collection or Tablet. Kudu stores timestamped changes separately from base columns and applies visible deltas when reading. This is a structural analogy, not equivalent key/version semantics. See the [Kudu Tablet design](https://github.com/apache/kudu/blob/master/docs/design-docs/tablet.md).

| Kudu maintenance operation | What changes | Identity boundary |
| --- | --- | --- |
| Minor delta compaction | Merge delta files without rewriting base columns | Same RowSet and row IDs |
| Major delta compaction | Merge deltas into selected base columns and replace the relevant blocks | Same RowSet and row IDs |
| Merging compaction | Combine multiple DiskRowSets into a new DiskRowSet | Row IDs may change; still within the original Tablet |

These are distinct maintenance tasks, not creation of a new Tablet for each update. See [Kudu 1.18.1 background maintenance](https://kudu.apache.org/docs/background_tasks.html). `DiskRowSet::MajorCompactDeltaStoresWithColumnIds` publishes changed RowSet metadata and swaps its base/delta components; it does not overwrite old file contents in place. See [diskrowset.cc](https://github.com/apache/kudu/blob/master/src/kudu/tablet/diskrowset.cc). The two `master` references were reviewed on 2026-10-20 and are not immutable release snapshots.

**Adopt:** separate foreground deltas from base-data rewriting, with version-aware application and subsequent consolidation.

**Adapt:** use Milvus segment lifetimes, timestamp semantics, and committed VChannel mutation ordering. Section 9.1.7 now selects identity-preserving, layout-preserving column consolidation as the Delta Consolidation direction; publication, retained-reader safety, and interleaved-delete handling remain open implementation contracts.

**Naming distinction (2026-10-22):** use **Delta Consolidation** for the Milvus background column-maintenance process previously discussed as the Major tier; do not name it Major compaction. It borrows selected-column rewriting with stable row IDs. L0 dispatch, previously called Minor in this discussion, distributes deltas rather than merging a RowSet's delta files. Merging/Mix retains full-rewrite maintenance. Keep Kudu's own Minor/Major terminology when describing its implementation. This is an adapted responsibility split, not identical operations or protocols. See the fixed-release [Kudu delta-compaction design, 1.18.1](https://github.com/apache/kudu/blob/1.18.1/docs/design-docs/tablet.md#types-of-delta-compaction).

**Do not copy:** Kudu's key uniqueness, tablet replication, REDO/UNDO representation, or compaction protocol. This precedent does not prove lower Milvus latency or solve stale scalar/vector indexes after an update.

Kudu's [major delta compaction, 1.18.1](https://github.com/apache/kudu/blob/1.18.1/src/kudu/tablet/delta_compaction.cc) generates UNDO changes while merging updates into base columns and applies history-GC rules. That historical reconstruction capability is not adopted: Section 9.1.5 limits this design to serving-generation visibility and permits full rewrites to fold history without producing UNDO.

**Batch application and bypass precedent.** Kudu's [DeltaApplier, 1.18.1](https://github.com/apache/kudu/blob/1.18.1/src/kudu/tablet/delta_applier.cc) prepares base/delta batches, maintains row selection, and applies updates during column materialization; when `MayHaveDeltas()` is false, it materializes the base column directly. Adopt the proven-unaffected fast-path principle. Kudu's row-addressed deltas are not deferred Milvus predicates, so its separate delete/update application is not a template for reordering our dependent mutations. Section 9.1.4 defines the Milvus-specific time-eligibility rule.

### 2.8 Iceberg: separate delete applicability from physical file identity

For v2/v3 equality deletes, Iceberg determines applicable data files using partition scope and data sequence numbers. It distinguishes the logical age of data from the commit that created its physical file; file sequence alone is not a safe delete-applicability bound. See the [Iceberg specification](https://iceberg.apache.org/spec/#scan-planning). Its versioned [RewriteDataFiles API, 1.7.0](https://iceberg.apache.org/javadoc/1.7.0/org/apache/iceberg/actions/RewriteDataFiles.html#USE_STARTING_SEQUENCE_NUMBER) describes retaining the rewrite's starting sequence for output files to accommodate newer equality deletes.

**Adopt:** durable applicability metadata and a distinction between data age and replacement-file creation. **Adapt:** use Milvus VChannel/partition scope, row timestamps, and current storage metadata. **Do not copy:** Iceberg sequence rules, snapshot commit machinery, or equality-delete encoding as a complete arbitrary-predicate/update protocol. This is precedent for the separation, not proof that Milvus can safely retire an L0 file or skip a candidate segment. Section 9.4 records the selected tradeoff.

### 2.9 SQL Server and Oracle: mask stale base rows and supplement visible changes

**Primary precedents for Section 10.7's query-merge direction**, reviewed on 2026-10-20:

| Mature implementation | Documented mechanism | Difference from the Milvus proposal |
| --- | --- | --- |
| SQL Server Columnstore | Scans consult a delete bitmap to exclude old rows and include rows from the deltastore. An update to a compressed row marks the old row deleted and inserts the updated row into the deltastore. | Whole-row replacement in a row-oriented deltastore, not a versioned patch to selected fields at the same segment offset; a columnstore scan is not the same access path as a Milvus scalar-index lookup. |
| Oracle Database In-Memory | A transaction journal records modified row IDs and marks their IMCU versions stale from the modification SCN. Queries needing the newer row obtain it from the buffer cache; columnar data, journal, and buffer cache together provide read consistency. | A dual-format database with its own transaction/SCN machinery, not Milvus segment-local field deltas or its scalar-index execution framework. |

Sources: Microsoft's [SIGMOD 2013 paper, Sections 4 and 4.4](https://www.microsoft.com/en-us/research/wp-content/uploads/2013/06/Apollo3-Sigmod-2013-final.pdf), the [SQL Server Columnstore overview](https://learn.microsoft.com/en-us/sql/relational-databases/indexes/columnstore-indexes-overview?view=sql-server-ver17), and Oracle's [In-Memory Column Store Architecture — Transaction Journal](https://docs.oracle.com/en/database/oracle/oracle-database/26/inmem/in-memory-column-store-architecture.html).

**Adopt:** keep a reusable base representation, suppress stale base results, and include visible changed values in query evaluation.

**Adapt:** mask by field and segment-local row offset at the query snapshot, then evaluate the latest visible field version relative to the pinned base/index generation.

**Do not copy:** SQL Server's whole-row delete-plus-insert or Oracle's buffer-cache/SCN implementation. Neither source establishes the exact per-field MVCC formula or a performance bound for Milvus.

**Evidence boundary:** these are mature precedents for the read-combination principle. Section 10.7 is a Milvus-specific design adaptation, not an existing Milvus feature or an assertion that these products implement its exact scalar-index overlay.

### 2.10 Assignment expressions: borrow a small typed interface, not full SQL

Primary sources reviewed on **2026-10-21** show different capability boundaries, not an industry requirement to support every SQL expression:

| Reference | Observed contract | Implication for this design |
| --- | --- | --- |
| Kudu native write API / `RowOperationsPB` | UPDATE supplies column values, with separate column-presence and NULL bitmaps; this wire format does not carry an arbitrary assignment-expression AST | Delta-based updates do not inherently require a rich expression language. Borrow explicit field patches and the distinction between absent and NULL values, not Kudu's key or transaction semantics |
| ClickHouse update mutations | Store a selection predicate and a mapping from assigned columns to expressions | Borrow the selection/assignment separation and expression-shaped RHS, not the full function library or its execution engine |
| PostgreSQL 18 UPDATE | RHS expressions can read old values of the same row; UPDATE also supports subqueries and other-table inputs through FROM | Borrow the row-local assignment model. Do not make cross-table, subquery, or general SQL capability a Milvus feature-completion requirement |

Sources: [Kudu native Update API, 1.18.0](https://kudu.apache.org/releases/1.18.0/apidocs/org/apache/kudu/client/Update.html), [Kudu row-operation encoding](https://github.com/apache/kudu/blob/master/src/kudu/common/row_operations.proto), [ClickHouse mutation parsing, v25.8.1.5101-lts](https://github.com/ClickHouse/ClickHouse/blob/v25.8.1.5101-lts/src/Storages/MutationCommands.cpp), and [PostgreSQL 18 UPDATE](https://www.postgresql.org/docs/18/sql-update.html). The Kudu source link is moving `master`; it describes the native write protocol, not the capabilities of SQL engines layered over Kudu.

**Adopt:** explicit changed-column assignments, typed expression composition, and a clear separation from row selection. **Adapt:** make the admitted expression subset small enough to reproduce consistently during online execution, load/recovery, and Go compaction. **Do not adopt:** full SQL, arbitrary UDFs, or external computation as a requirement for this mutation path.

The selected scope in Section 10.8 is a Milvus product/design choice, not an industry-standard minimum. Reducing full-row replacement and redundant unchanged-vector/index work is the core benefit target; complex assignment expressions are not a prerequisite for that target. No measured Milvus gain follows from these precedents alone.

**Execution-layer precedent: separate operators, shared primitives.** In DuckDB v1.4.0, [PhysicalFilter](https://github.com/duckdb/duckdb/blob/v1.4.0/src/execution/operator/filter/physical_filter.cpp) calls `ExpressionExecutor::SelectExpression` to obtain a row selection, while [PhysicalProjection](https://github.com/duckdb/duckdb/blob/v1.4.0/src/execution/operator/projection/physical_projection.cpp) calls the same executor abstraction's `Execute` method to produce value columns. Adopt the separation between boolean selection and typed value production; adapt it to a small assignment module and Milvus's ordered, versioned row view. Do not infer that Milvus needs two complete execution frameworks, that its current filter executor already implements assignments, or that DuckDB's storage-update behavior must be copied. This is architectural precedent, not evidence of an existing Milvus update evaluator.

## 3. Current implementation

### 3.1 Delete application

The relevant source paths are:

- [SegmentInterface.h](../../internal/core/src/segcore/SegmentInterface.h): the current `Delete` entry accepts PKs and timestamps, not a selection bitmap.
- [ChunkedSegmentSealedImpl.cpp](../../internal/core/src/segcore/ChunkedSegmentSealedImpl.cpp): Sealed deletion invokes `DeletedRecord<true>::StreamPush` with PKs; its lookup callback resolves those PKs to segment offsets.
- [SegmentGrowingImpl.cpp](../../internal/core/src/segcore/SegmentGrowingImpl.cpp): Growing deletion uses its insert-record PK lookup and `DeletedRecord<false>`.
- [DeletedRecord.h](../../internal/core/src/segcore/DeletedRecord.h): `InternalPush` performs PK lookup through `search_pk_func_`, then updates the timestamp/offset skip list, deletion mask, counters, and resource accounting; `StreamPush` also maintains delete snapshots.
- [FilterBitsNode.cpp](../../internal/core/src/exec/operator/FilterBitsNode.cpp) and [MvccNode.cpp](../../internal/core/src/exec/operator/MvccNode.cpp): predicate evaluation and query visibility filtering already operate on segment-aligned bitmaps.
- [QueryContext.h](../../internal/core/src/exec/QueryContext.h): the query execution context exposes a const segment pointer; a mutation pipeline needs an explicit write-capable entry/context.
- [segment.go](../../internal/querynodev2/segments/segment.go): `LocalSegment.Delete` pins segment lifetime and serializes delete mutation with `deltaMut`.

The current interfaces do not already implement expression-delete application. This proposal must not be confused with an existing capability.

### 3.2 Upsert and partial-upsert baseline

Source baseline: local checkout at `fa995414c1` (the preceding predicate-delete documentation commit), inspected on 2026-10-20. The current partial-update design is [Partial Update Optimistic Lock](design_docs/20260601-partial-update-optimistic-lock-design.md).

| Operation | Observed implementation | Consequence for this proposal |
| --- | --- | --- |
| Full-row upsert | Builds insert messages and PK-delete messages; retrieval depends on AutoID/authorization and related requirements | Still replacement-row mutation, not an update delta |
| Partial upsert | Queries by PK, requests `output_fields = ["*"]`, merges old fields with supplied changes in Proxy, then builds insert and delete messages | Partial at the API boundary, but full-row materialization/reinsertion in the storage path |
| Update-only: modify existing rows, do nothing if absent | Requested behavior in this discussion; no separate row-update RPC was identified in the inspected API/implementation | Do not treat this as a verified existing execution path or assume partial upsert has these missing-row semantics |

See `retrieveByPKs`, `queryPreExecute`, and `upsertRetrieveOutputFields` in [task_upsert.go](../../internal/proxy/dml/task_upsert.go), and `appendUpsertAttempt` in [task_upsert_streaming.go](../../internal/proxy/dml/task_upsert_streaming.go). Partial upsert also carries validation and optimistic-concurrency/CAS behavior. Calling it **API-level syntactic sugar over replacement-row writes** describes its storage cost, not an absence of server-side semantics. Missing-PK behavior depends on the operation/payload; it is not universally a no-op.

The replacement row is appended to Growing data; this does **not** mean each request creates a new segment. The proposed delta path targets redundant full-row reads/writes and row-version churn within that actual baseline. Avoiding full-row reconstruction does not by itself eliminate PK resolution, validation, read-modify-write dependencies, or conflict handling.

## 4. Proposed predicate-delete pipeline

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

With update deltas, `predicate(i)` must evaluate the row's field values visible to this delete, not necessarily the immutable base values. Section 6 states this shared invariant; Sections 10.6–10.7 select versioned field values and snapshot-aware base-index/delta merging, while their concrete access interfaces remain open.

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

### Shared mutation ordering and timestamp visibility

**Agreed invariant:** online apply, load/recovery, and compaction must be observationally equivalent to the same VChannel's committed logical mutation order and timestamp visibility. This is not a new cross-VChannel atomicity guarantee, nor a requirement to execute every physical WAL record serially. Batching and parallelism are valid only if they preserve those results.

For example, for an existing row with `age = 17`:

```text
Committed update at ts=100: age := 20
Committed delete at ts=110: age < 18, with the earlier update visible
Expected result: the row survives, with age=20
```

Applying the predicate against the old base value would silently delete a live row. Replaying all deletes before all updates, or grouping mutations by field without preserving dependencies, is therefore invalid. Updates and predicate deletes will reside in separate durable file types; load and compaction merge their records by effective mutation TimeTick, not by file type or creation time. This ordering direction is agreed; the detailed merge/completeness protocol remains open; see Section 9.7.

This also qualifies index-based predicate execution: an index over unmodified base data alone cannot establish the correct match set after visible updates to that field. Scalar-index, raw-column, and cached-expression access must represent the same versioned row view. Sections 10.6–10.7 settle the field-version and query-merge direction, not its implementation. The existing predicate-load policy still avoids an extra eager-load requirement; it does not itself provide this update-aware overlay.

The goal is ordered, timestamp-correct mutation application across all three paths. It does not establish client-request idempotency, rollback, or historical field-value retention merely by referring to WAL replay.

#### Logical TimeTick order versus physical WAL order

The existing WAL allocates a message's TimeTick before its backend append. Concurrent writes can therefore appear physically as `B@101, A@100` even though their logical order is `A@100, B@101`. The consumer reorders them before delivery. In this design, "WAL order" means the relevant VChannel's committed logical mutation order, not physical MessageID order, Proxy arrival order, or file creation order. This does not imply that every insert/delta file is physically unordered.

**A confirmed watermark is a gap-free acknowledged prefix, not the largest timestamp observed.** At the PChannel WAL layer, an earlier allocated append still in flight prevents confirmation from advancing past it. For example, if confirmation has reached 99, completion of `B@101` alone cannot advance it while `A@100` is outstanding. Once A is also acknowledged, the prefix can advance through 101. The consumer uses the confirmation barrier to release buffered records in TimeTick order; transaction assembly additionally withholds uncommitted bodies. Sorting determines order; confirmation establishes that an earlier outstanding write is not missing from the deliverable prefix.

Do not confuse this append-confirmation watermark with QN apply progress, L0 persistence completion, or target-segment delta publication. Those are separate downstream milestones and must independently prove complete coverage. A TimeTick confirmation does not mean every allocated write succeeded, every transaction committed, or every consumer finished applying it.

The effective mutation timestamp is the existing WAL time, not a new clock: an ordinary message uses its TimeTick; an assembled internal transaction uses its commit TimeTick. In the current streaming insert path, each row's timestamp is explicitly set to this effective message TimeTick. Preserve the same meaning through mutation buffering and delta persistence; do not substitute Proxy receive time or flush time.

**Current-code cross-check:** the [TimeTick/transaction guide](../agent_guides/streaming-system/wal/timetick_and_txn.md), [TimeTick allocation](../../internal/streamingnode/server/wal/interceptors/timetick/timetick_interceptor.go), [AckManager prefix advancement](../../internal/streamingnode/server/wal/interceptors/timetick/ack/manager.go), and [scanner reordering/transaction assembly](../../internal/streamingnode/server/wal/adaptor/scanner_adaptor.go) establish these existing boundaries. [Insert message adaptation](../../pkg/streaming/util/message/adaptor/message.go) and [growing insert persistence](../../internal/streamingnode/server/wal/vchannel/segment/pack_writer.go) assign the effective TimeTick to row timestamps. Adopt these Milvus contracts rather than importing physical-log ordering or SQL transaction assumptions from another engine.

### Stable row identity and timestamp

- A row offset identifies a row in one segment layout; it is not a durable or cross-segment identifier.
- Pin segment lifetime and the relevant data/schema layout throughout evaluation and application. Reopen, replacement, release, or compaction must not silently reinterpret selected offsets.
- Use one fixed `delete_ts` and captured execution view. Growing inserts outside the selected prefix, or with `insert_ts >= delete_ts`, must not be deleted accidentally.
- Preserve the existing write serialization discipline, including interaction with PK deletes and loaded deltas; a read-only query pipeline alone does not provide it.

### DeletedRecord compatibility

- Preserve the current skip-list ordering, row-level duplicate suppression, deletion mask, and resource accounting.
- Preserve supported timestamp visibility and latest-snapshot queries within the serving segment's retained state. Do not replace timestamped records with only a timeless OR-ed bitmap. This does not require a replacement segment to reconstruct history folded by full rewrite; see Section 9.1.5.
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
8. For every allowlisted expression/type combination, compare Go compaction selection against Segcore and an independent oracle, including boundary values and NULL behavior where supported. Verify that an unsupported DELETE subexpression routes the whole request through the legacy PK path before predicate persistence/application; an unsupported UPDATE WHERE or SET must instead reject the request before append.
9. Validate the drop-field drain contract in Section 9.1.3 with delayed Streaming/L0 work, old growing data, unequal per-channel progress, mixed PK/predicate backlog, segment replacement, and restart between the two drain phases or before metadata publication. Cover an empty L0 inventory while deletes remain in Streaming and mismatched checkpoint/L0 observations. Physical column removal must not precede complete durable coverage and predicate evaluation.
10. For predicate-delta loading, verify index-only, raw-data, and mixed access paths against the same row/timestamp oracle. With suitable exact scalar indexes, check that `c > 18 && a == 18` does not read the corresponding raw columns for predicate evaluation; also cover cold indexes, lazy raw columns, NULL semantics, prefetch, and read/cancellation failures. Measure load/recover latency, storage reads, and peak memory against equivalent PK-delta replay rather than assuming a benefit.
11. For the joint design, compare online application, reload/recovery, and compaction with an independent ordered-mutation oracle. Cover update-before-predicate-delete, interleaved updates/deletes, historical snapshots, stale base indexes, separate delta files, and segment replacement. Test missing/deleted-row behavior after its API contract is selected; do not silently assume updates resurrect rows.
12. Evaluate update benefits against the actual partial-upsert replacement-row baseline, including foreground work, accumulated delta state, concurrent reads, compaction, and recovery. Section 10 defines candidate measurements; no update benchmark has been completed for this design.
13. Verify concurrent constant patches against the ordered-mutation oracle: disjoint-column patches retain both changes, overlapping-column patches resolve per column in committed order, and predicates over updated fields see preceding applicable mutations. Compare online execution, reload, and compaction, including intermediate timestamp-visible states rather than only the final row.
14. Validate coarse L0 dispatch while the collection is released: candidates come from current storage metadata, not loaded QNs; published segment deltas remain discoverable by offline compaction and later load. Inject failures between object writes, metadata publication, and L0 retirement.
15. Cover both L0-before-Mix and Mix-before-L0, queued L0 work, Streaming mutations arriving during compaction, multiple replacement generations, and QueryViews still serving retired storage segments. Verify complete mutation coverage without an ingress association table or descendant lookup.
16. Verify pruning against an ordered-mutation oracle, including absent statistics, stale base-column bounds after updates, and replacement segments containing old rows. Compare coarse fanout with the PK path at equal logical work; measure zero-match evaluation, metadata growth, load/recover latency, and online/background resource interference.
17. Validate asynchronous recovery's per-request gate: with complete relevant apply progress at 100, requests requiring 95 or 100 may proceed, while a request requiring 105 must wait until coverage reaches 105, even if later backlog remains. A completed record at 110 must not hide an unfinished mutation at 99. Cover multiple segments/VChannels, replacement/reload, apply failure, and mixed update/delete ordering.
18. Measure base-load completion, time to serve each consistency level, full catch-up time, and cold-recovery P99 separately under lazy loading, accumulated predicates, and concurrent searches. Verify that blocked Strong/Session/Bounded requests are not silently downgraded and that background work can make progress under load; no upgrade-timeout or availability improvement is assumed.
19. Verify that each reported deleted count and complete apply watermark describe one coherent segment state. Cover concurrent apply, delayed/out-of-order reports, replicas at different progress, release/reload, and compacted-away sources. Reports must neither double count replicas nor claim coverage of unfinished mutations.
20. Verify hit-independent scheduling with missing/stale stats and with an exact zero-deleted-row observation plus many outstanding predicates. QN application alone must not clear the durable backlog count; successful data compaction must retire only consumed records after output publication, including zero-match records. Test repeated delivery versus separate committed mutations using the same expression.
21. Verify Section 10.7's index/delta merge against a versioned full-scan oracle: updates entering and leaving a predicate's match set, repeated updates to one field, snapshots before/between/after updates, patches to unrelated fields, and updates assigning the base value again. Cover empty base-index results, stale min/max pruning, compound predicates, supported NULL semantics, and row deletion. Exercise concurrent publication and base/index replacement so masks, values, and progress cannot describe different generations. Measure updated-row scan cost as delta density and retained-version depth grow; this document does not claim those tests have run.
22. Validate the logical message and assignment boundaries in Sections 9.5 and 10.8: explicit delete/update discrimination, bound field IDs/types, shared scope, absent-versus-NULL patches, and phase-one rejection of non-literal RHS nodes before append. Before widening the allowlist, verify matching results and error behavior across online, load, and compaction evaluators, including earlier field updates and repeated delivery of non-idempotent arithmetic mutations. These are planned checks, not completed verification.
23. Validate the evaluator separation in Section 9.6.1: DELETE and UPDATE use identical WHERE eligibility and selection semantics, while SET eligibility is checked independently. Cover zero/many selected rows, multiple constant assignments, explicit NULL, unchanged omitted columns, and non-literal phase-one rejection. Future RHS expressions must resolve all inputs against the same pre-mutation row view across online, load, and compaction execution. These are planned checks, not executed tests.
24. Verify ordering under concurrent append: physically record `B@101` before `A@100`, keep confirmation behind the outstanding A, and confirm logical delivery of A before B. Distinguish append confirmation from QN application and durable L0/segment coverage; preserve the effective row/mutation timestamp, including existing internal transaction commit timestamps.
25. Persist `UPDATE@100`, `DELETE@110`, and `UPDATE@120` in separate update/predicate-delete files and compare online, reload, and compaction results with the ordered oracle. Vary file creation/flush order, split records across files, delay one input, and fail/retry publication. No intermediate update may be coalesced away, and a later record cannot justify advancing past missing earlier work.
26. Verify the single-mutation request contract: reject batches of independent native UPDATE/DELETE statements while accepting one predicate that matches many rows and one UPDATE containing several constant assignments. Preserve one native mutation message per target VChannel; do not infer collection-wide atomicity or user-controlled transactions from internal WAL transaction support.
27. Validate a common sync cut with highly uneven traffic: 100 update records, two PK deletes, and one predicate delete before T must all become durable even if the sparse buffers are below their individual thresholds. Cover a genuinely empty type, arrivals after T, overlapping file time ranges, and each trigger category. Measure small-file count, object requests, metadata growth, and retained-WAL pressure; matching-row counts are not buffer capacity estimates.
28. Fault-inject cross-type publication at every boundary: one file uploaded while another fails, partial metadata registration, all objects durable before publication, and crash/retry after publication before checkpoint or retained-input release. Run concurrent L0 scheduling and downstream data compaction: neither may treat `DELETE@110` as a complete usable prefix while `UPDATE@100` is unavailable. Verify consistent prefix/discovery observations, confirmed-empty types, failover, and safe cleanup without relying on object-listing order. These are required future tests, not completed verification.
29. Verify the PK-tombstone logical oracle with `DELETE pk=2@100` and `DELETE pk=2@150` in either materialization order: the final deleted-version set must agree, while retained timestamps preserve supported intermediate-snapshot semantics. Contrast this with a dependent update/predicate-delete sequence. For the proposed barrier, verify monotonic advancement per VChannel, no advancement across a missing type, isolation between VChannels, and no scheduler split that omits an applicable earlier delta despite a published higher cut.
30. Verify Section 9.7.3's range contract with barrier 100: publish complete `(100,120]`; reject partial overlap `(80,120]` and gap `(110,120]` without changing metadata or the barrier; acknowledge already-covered `(80,100]` without republishing. Cover invalid bounds, lifecycle identity and stale-owner fencing, concurrent publishers, and durable commit failures. A rejection must not silently clip a range or advance by `max()`.
31. Drop the initial and subsequent publication responses while SN remains running. Retrying the same range must eventually obtain `RangeApplied` once coverage is durable, without duplicate registration, premature input release, or cleanup based on an ambiguous timeout. Distinguish logical coverage from acceptance of arbitrary file paths supplied by a retry.
32. Crash SN after the file metadata and prefix commit but before ACK or local progress update. Recovery must read the committed per-VChannel prefix first and skip rematerializing that range. Contrast with an uploaded object whose metadata/prefix never committed: that range remains pending. Exercise several VChannels sharing one PChannel, failed prefix reads, MixCoord restart, and owner changes; no PChannel maximum or reset to zero may replace the individual authoritative prefixes. These are required future tests, not completed verification.
33. Publish SN ranges `(0,100]` and `(100,200]`, then dispatch `(0,50]`, `(50,150]`, and `(150,200]`. Verify splitting and combining publication batches without requiring matching file boundaries, consuming beyond the published prefix, losing a sparse delta type, or omitting applicable records at either endpoint. This is a required future test, not an implemented scheduler capability.
34. Delay an earlier dispatch range or one of its required targets while later physical work finishes. Later output must not become a usable downstream prefix across the missing work, and completed-dispatch progress must not advance prematurely. After durable publication to every required target, verify ordered consumption and complete coverage, including retry/restart and current-storage-view target selection. Physical task completion order is not the visibility contract; the detailed protocol remains to be designed.
35. Run the same interleaved PK-delete/predicate-delete/update program through Sort, Mix (including optimized merge-sort), full-rewrite bump-schema-version, and clustering compaction against an independent ordered-row oracle. Cover multiple input batches, matches appearing only in later batches, predicates reading earlier updates, and deleted rows followed by updates. Preparation may be reused, but one batch's matches or exhausted iterator must not suppress mutations for another batch. Verify that batch completion alone does not publish whole-segment completion.
36. Verify Section 9.1.4's fast path against execution without pruning. For deltas in `(100,120]`, rows written at `10..20` must still be evaluated even though the timestamp intervals do not overlap; rows all written at `130..140` may bypass those mutations. Cover mixed-age batches, timestamp equality under the applicable operation's rules, import/CDC effective timestamps, and missing time bounds. A bypass must preserve required TTL processing, sorting, clustering, and output publication. These are planned tests, not completed verification.
37. Keep an old segment serving while its full rewrite completes: queries pinned to the old generation must continue to resolve its base plus timestamp-visible deltas without mixing replacement-base state. Verify safe reader completion and handoff before reclaiming that state. After replacement, validate the new baseline and subsequent mutations; do not require reconstruction of intermediate history folded into that baseline or introduce UNDO files to satisfy the test. Existing retention/pinning obligations and pending deltas must remain protected.
38. Verify that same-channel L0 dispatch and each data-rewrite type cannot execute concurrently. A successful full rewrite must incorporate its applicable input deltas into the output base and must not copy consumed records into the output's pending delta set, including predicates evaluated with zero matches. Mutations still awaiting dispatch are not consumed by that rewrite and must remain available for subsequent L0 dispatch. Test QN load/handoff separately rather than treating it as an additional DataNode rewrite phase.
39. For Delta Consolidation, replace INT64 field `b` from a multi-file `[a,b,c]` group with a new single-column `[b]` group in one new manifest generation. Verify unchanged segment identity, row count/order/offsets, original file references for `a/c`, and unchanged vector data and ANN-index reuse. New-manifest reads must use the new `b`; pinned old-manifest reads must remain coherent, and old `b` bytes in the shared files must not be reported as reclaimed. Maintain the changed column's scalar index consistently. Verify that update-backlog thresholds schedule Delta Consolidation rather than introduce an extra Mix trigger, and that L0 dispatch still only dispatches deltas. Storage metadata tests alone do not validate this end-to-end flow.
40. Verify Delta Consolidation against an ordered mutation oracle with interleaved updates, PK deletes, and predicate deletes. Do not physically remove rows or lose outstanding delete effects. Include a predicate whose result would change if replayed against the post-consolidation base instead of its original mutation-time values; consuming update history must not corrupt its later replay. The residual-delete protocol must be designed before this test can pass.
41. Exercise sustained large-value updates with phase-one all-resident delta execution. Measure QN value-version residency and DN per-task delta memory independently of base batch size, check byte-based Delta Consolidation scheduling, and preserve existing memory-protection behavior. Uploading L0 files must not be treated as reclaiming QN versions; no spill-based capacity benefit is assumed. These are required future tests, not completed validation.

## 9. System-level integration checkpoints

The query-side Segcore change is only the local application mechanism. Compaction, scheduling statistics, target-QN segment loading, and L0 dispatch require integration before supported predicate deletes can replace expanded PK deletes end to end. Compaction evaluation now favors a restricted Go-layer evaluator; its exact supported set and execution contract remain open. The directions below separate scheduling hints (Section 9.2), target-QN loading (Section 9.3), and durable L0 dispatch (Section 9.4); agreement on one does not solve the others.

### 9.1 Applying predicate deletes during compaction

The current Go compaction path reads delta logs as `(pk, delete_ts)` pairs, builds a PK-to-timestamp map, and filters input rows with that map. It does not have the Segcore expression framework and therefore cannot execute a persisted predicate delete directly. See [compaction/common.go](../../internal/compaction/common.go) and [mix_compactor.go](../../internal/datanode/compactor/mix_compactor.go).

[milvus-storage PR #556](https://github.com/milvus-io/milvus-storage/pull/556) explored a storage-side solution. Its manifest-aware masked reader recognizes predicate delta logs, evaluates SQL predicates with Arrow compute, and can return either a compaction-oriented keep mask or per-row delete timestamps. This would let a Go compactor consume already-filtered batches without embedding an expression engine in Go.

That PR is a closed, unmerged prototype rather than an available dependency. It is historical context, not the current preferred execution location. **The stateless-Segcore extraction approach has been abandoned.**

#### 9.1.1 Preferred direction: a restricted Go-layer predicate evaluator

The current preference is a lightweight evaluator in the Go compaction layer, without moving expression execution into milvus-storage or extracting Segcore's execution framework. This is an architectural direction, not an implemented engine or a claim that the engineering cost is already established.

Mix compaction already reads `storage.Record` batches with Arrow columns, filters PK deletes and TTL through `EntityFilter`, and writes surviving rows. See [mix_compactor.go](../../internal/datanode/compactor/mix_compactor.go), [merge_sort.go](../../internal/datanode/compactor/merge_sort.go), and [entity_filter.go](../../internal/compaction/entity_filter.go). With updates included, the agreed extension is the shared ordered batch executor in Section 9.1.4:

```text
Read base batch -> ordered PK-delete / predicate-delete / update execution
  -> updated values and surviving rows -> rewrite-specific processing -> write
```

This follows the **ClickHouse mutation-stage precedent in Section 2.5**, not its concrete engine. Existing TTL policy remains required; it is not another WAL delta type and is not skipped by a delta fast path. Reusing Milvus's typed expression representation is a candidate way to avoid a second expression language; it does not by itself guarantee identical evaluation semantics. The evaluator need not reproduce ANN execution or query scheduling.

#### 9.1.2 Agreed safety boundary: strict supported set and legacy fallback

- **Default deny for the new path.** Predicate delete supports only an explicitly allowlisted, deliberately restricted expression set whose semantics are stable and verified across query-side application and compaction. The exact initial set is not selected yet.
- **Validate the complete typed expression.** Eligibility covers operators, operand types, NULL behavior, conversions, and composition, not merely an operator-name list. One unsupported node or combination makes the whole expression ineligible; do not partially apply a supported subset.
- **Preserve the existing route.** An expression accepted by today's delete API but unsupported by the new predicate path is evaluated through the existing query path to retrieve matching PKs, followed by ordinary PK deletion. Existing validation and operational limits still apply; fallback does not promise that every large delete will succeed.
- **Choose before side effects.** This is request routing before predicate persistence/application, not a recovery mechanism after a predicate delete has committed or partly executed. A compactor encountering an unsupported persisted predicate must not silently ignore or approximate it; its error/recovery contract remains to be specified.
- **Keep semantic parity as the admission gate.** A broader Segcore expression capability does not automatically expand predicate-delete support. Extending the allowlist requires cross-evaluator validation first.

Timestamp semantics, evaluator errors, version compatibility, and the detailed L0 dispatch protocol remain open integration work. Section 9.4 selects coarse dispatch without expression evaluation in L0. Section 9.1.3 records the agreed drop-field safety requirement and preferred drain granularity; its progress and recovery protocol remains to be designed. A narrow expression set reduces the semantic surface; it does not remove those obligations. No detailed operator set, evaluator ABI, or routing implementation is specified here.

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

#### 9.1.4 Agreed shared batch DeltaExecutor and safe bypass

Discussion checkpoint: **2026-10-22**. This settles the execution model, not a concrete class, ABI, or implemented engine. Evolve the filtering-only responsibility of `EntityFilter` into a shared ordered delta-execution layer: selection alone cannot represent updates to fields used by later predicates.

**All four data-rewrite families participate.** Current [Sort](../../internal/datanode/compactor/sort_compaction.go), [Mix](../../internal/datanode/compactor/mix_compactor.go) and its [merge-sort path](../../internal/datanode/compactor/merge_sort.go), [full-rewrite bump-schema-version](../../internal/datanode/compactor/bump_schema_version_compactor.go), and [clustering](../../internal/datanode/compactor/clustering_compactor.go) already consume PK deltas through `EntityFilter`. Sort cannot be excluded on the assumption that L0 targets are necessarily sorted: the current [L0 target selection](../../internal/datacoord/compaction_task_l0.go) does not require that invariant. The new shared execution contract covers these rewrite paths; it does not require a version-only or additive schema operation to become a full rewrite.

**Prepare one logical sequence; apply it to each eligible batch.** Merge applicable PK-delete, predicate-delete, and update records from the complete persisted input prefix by their effective mutation TimeTicks, for example:

```text
[Update@100, PKDelete@105, PredicateDelete@110, Update@120]

Read base storage.Record batch (Arrow columns)
  -> apply the applicable ordered delta sequence
  -> updated field values + surviving-row selection
  -> rewrite-specific sort / merge / clustering / schema projection
  -> write output
```

This is not three passes grouped by mutation type. Delta reads, ordered-program preparation, and expression parsing can be reused across batches within their applicable segment/schema context. Do not re-read the same delta files for every batch or require a separate fully materialized merged file.

**Phase-one capacity decision (2026-10-22): all applicable deltas are memory-resident; base batching does not bound delta residency.** In the base-batch-at-a-time model, every base batch must revisit the complete applicable ordered mutation program. Load and retain that program, including required assignment payloads, until all base batches in its execution scope have been processed. Applying the program to the first batch does not make its delta prefix disposable: later batches may match the same predicates. Reading/decoding delta files in chunks is not evidence that the retained execution state is also bounded to one chunk. Disk cache and spill are outside phase one.

The capacity baseline is **resident applicable delta state + base/merge working set + batch working/output buffers + preparation/decoding overhead**, not merely one Arrow batch. Batch working buffers hold updated columns and selection state; they are not a persistent, multiversion QN overlay. Scope the resident delta set to the segment(s) whose executors are simultaneously active, not all collection history. Current [Mix](../../internal/datanode/compactor/mix_compactor.go) constructs an `EntityFilter` before its base-reader loop; [MergeSort](../../internal/datanode/compactor/merge_sort.go) retains filters for all input segments during the merge. [ComposeDeleteFromDeltalogs](../../internal/compaction/common.go) reads PK deltas and builds a resident PK-to-latest-delete-timestamp map. This is compacted PK-delete state, not necessarily the original encoded file bytes; the same reduction cannot be assumed safe for an interleaved update/predicate program.

Kudu's [DeltaApplier](https://github.com/apache/kudu/blob/1.18.1/src/kudu/tablet/delta_applier.cc) can seek a delta iterator to a row ordinal and prepare it for a row batch. That row-addressed model is a material alternative, not evidence that our unevaluated predicates can be partitioned by base batch. A bounded-memory delta path would require an explicit additional design, such as externally backed replay or a different execution organization; no disk-cache/spill solution or resulting memory reduction is assumed here. Concrete representation, memory admission, and oversize handling remain open.

The executor maintains batch-local **current field values, live-row selection, and mutation time/scope eligibility**. Later predicates read preceding updates; a deleted row is not resurrected by a later update. Keep the field-value view consistent with Section 6 rather than evaluating every predicate against unchanged base columns. Intermediate values may use replacement columns or temporary batch-local state; do not require physical mutation of borrowed Arrow input arrays or the QN's multiversion overlay. Unchanged columns may be reused with correct ownership. Offline output construction does not require using the QN's `DeletedRecord` as its batch state.

For a row initially containing `age=17`, `UPDATE age=23@100 -> DELETE WHERE age<18@110 -> UPDATE age=25@120` retains the row with `age=25`. The delete must see 23, not the original 17 or the future 25. The same program may also match rows in the next batch: finishing one batch does not globally consume its predicates or permit reusing that batch's match bitmap for another batch.

**Safe temporal bypass.** Skip a batch's delta execution only when it is established that no row is eligible for any mutation in the selected sequence. A delta timestamp interval describes when mutations occurred, not an insertion-time window containing all rows they may affect:

- With mutation times in `(100,120]`, a batch whose rows were all effectively written at `130..140` may bypass those mutations: all rows are newer than every mutation.
- A batch whose rows were written at `10..20` must still be evaluated. Although its row timestamps do not intersect `(100,120]`, those old rows may be the mutations' targets.
- A mixed-age batch requires row-level eligibility or another sound exclusion proof. Use logical effective write/commit timestamps, not physical file creation time; the existing [EntityFilter](../../internal/compaction/entity_filter.go) accounts for import/CDC commit timestamps. Missing or untrustworthy bounds cannot justify skipping execution. Preserve each operation's timestamp-equality rule rather than deriving it from interval notation.

Bypass skips only the proven-inapplicable delta work, **not required TTL handling, sorting, clustering, schema processing, or output publication**. For a batch with eligible rows, a predicate can still match zero rows; that outcome is not known from timestamp eligibility alone. Sections 2.5 and 2.7 provide the ordered-transform and proven-unaffected precedents, not a vendor implementation of this exact Milvus rule.

**Batch execution is not whole-rewrite completion.** Applying through T to one batch does not prove that a segment is materialized through T. Complete all required input batches, make the output durable, and publish the corresponding metadata before reporting rewrite completion. Sort/clustering can buffer, reorder, or repartition across batches; this shared executor does not require one input batch to be immediately written as one output batch. Physical field removal still honors Section 9.1.3 and occurs only after the required predicates can be evaluated with their input fields available.

#### 9.1.5 Agreed version-retention boundary: serving visibility, not time travel

Discussion checkpoint: **2026-10-22**. Multiversion execution does not imply indefinite historical storage. The agreed boundary is:

- **Old segment while serving:** preserve its supported timestamp-visible state, resolved from the original base plus the corresponding visible update/delete deltas. Readers must use a coherent, pinned generation rather than mix an old overlay with a replacement base. Finishing the background rewrite alone does not permit invalidating that serving state or reclaiming resources still used by readers.
- **New segment after full rewrite:** write the final surviving values for the mutation prefix incorporated by this rewrite as a new baseline. The replacement does not retain or reconstruct intermediate history folded into that baseline. For example, base `age=17`, `UPDATE age=23@100`, and `UPDATE age=25@120` may produce a replacement base containing `age=25`, without carrying the old 17/23 versions.
- **No new UNDO or long-term time-travel requirement:** do not add reverse deltas or a post-rewrite historical-read window for this feature. Source-state reclamation still follows safe handoff, reader lifetime, and existing retention/pinning rules. This boundary does not authorize losing pending mutations that the rewrite has not consumed.

**Industry principle and applicability.** PostgreSQL [VACUUM](https://www.postgresql.org/docs/17/routine-vacuuming.html#VACUUM-FOR-SPACE-RECOVERY) preserves row versions while they may still be visible to other transactions and reclaims obsolete versions afterward. Adopt the separation of reader-consistency protection from long-term historical retention; adapt it to Milvus segment generations and handoff, not PostgreSQL's transaction implementation. Kudu's UNDO-based reconstruction in Section 2.7 is a material alternative, deliberately not selected here.

**Current-code evidence.** The existing [Mix rewrite](../../internal/datanode/compactor/mix_compactor.go) skips rows rejected by `EntityFilter` and writes survivors; the [output writer](../../internal/datanode/compactor/segment_writer.go) publishes the rewritten data rather than deleted-row history. The [Sort compaction test](../../internal/datanode/compactor/sort_compaction_test.go) likewise expects 995 output rows after deleting five of 1,000, with empty output deltalogs. This supports the history-folding boundary, not a claim that update-aware handoff is already implemented.

The policy is settled. Concrete pinning, handoff, and reclamation mechanics remain implementation work; the feature must not silently reinterpret an already-admitted reader's state during replacement.

#### 9.1.6 Agreed compaction boundary: exclusion and consumed-delta ownership

Discussion checkpoint: **2026-10-22**. DataNode compaction is an offline transformation of a fixed input base and its selected, complete delta input; it is not a QN segment-load or streaming-catch-up process.

- **Reuse execution exclusion.** The current [compaction scheduler](../../internal/datacoord/compaction_inspector.go) prevents L0 compaction from executing concurrently with Mix, Sort, full-rewrite bump-schema-version, or Clustering compaction on the same channel. This is stronger than exclusion on an individual target segment: L0 may append deltas to any eligible segment on that channel. Preserve this boundary when extending L0 to dispatch update and predicate deltas. Existing [scheduler tests](../../internal/datacoord/compaction_inspector_test.go) cover same-channel Bump/L0 conflicts in both directions.
- **Fold consumed deltas into the replacement base.** Execute the applicable ordered input mutations, write surviving rows with their final values, and publish the rewritten output. Fully consumed PK deletes, predicate deletes, and updates are not copied into that output as pending deltas. A predicate fully evaluated with zero matches is also consumed. The current rewrite/output-writer evidence in Section 9.1.5 supplies the PK-delete precedent; extending it to new delta types remains implementation work.
- **Exclusion is not global drainage.** New WAL mutations may still arrive, and undispatched deltas may remain in Streaming/L0. They are outside this rewrite's consumed input, must not be discarded, and remain subject to subsequent dispatch against current storage metadata. Neither rewrite completion nor scheduler exclusion claims that all channel mutations have been consumed.
- **Keep load/handoff in its own discussion.** QN loading of the replacement, streaming-tail catch-up, reader admission, and serving-generation handoff belong to Section 9.3 and the corresponding later-stage design. Do not infer a compaction-side replay problem by assuming that the output carries already-consumed input deltas. No additional `base-consumed-through` field is settled by this discussion; any such proposal needs evidence from the actual load/handoff path.

The existing cross-batch catalog-publication risk in Section 12.1 remains separate: execution exclusion does not by itself provide crash-atomic metadata publication. The consumed-delta rule above describes **full rewrites**; update-only Delta Consolidation has the narrower ownership boundary below.

#### 9.1.7 Agreed maintenance separation: L0 dispatch, Delta Consolidation, and full rewrite

Discussion checkpoint: **2026-10-22**. **Delta Consolidation** is the agreed name for a distinct background column-maintenance process, previously discussed as the Major tier. It is not segment-merging compaction. The identity-preserving schema-reconciliation paths provide an analogy even though their existing implementation is named bump-schema-version compaction. This is a design-level responsibility split, not a rename of existing task enums/APIs or a claim that the new operation is implemented.

| Process | Responsibility | Segment/layout and index boundary |
| --- | --- | --- |
| L0 dispatch | Append/distribute L0 PK-delete, predicate-delete, and update deltas to the appropriate segments under the complete, ordered dispatch contract | No business-expression evaluation, base rewrite, row movement, or index rebuild; preserve existing PK Bloom-filter filtering |
| Delta Consolidation | Consume update deltas into selected columns of the existing segment | Preserve segment identity, row count, row order, and offsets; maintain affected scalar indexes and reuse unchanged columns and indexes |
| Merging | Perform normally scheduled Mix compaction and its full data rewrite | Produce replacement segments, fold applicable ordered updates and deletes, physically remove deleted rows, and build required output indexes |

Other established full-rewrite operations, including Sort, Clustering, and the full-rewrite schema-bump branch, retain Section 9.1.4's shared delta-consumption contract. This separation does not remove those operations or require a new update to trigger them.

**Single-column rewrite plus manifest switch, not S3 object overwrite.** Preserve the segment identity and write replacement data for the selected column, then publish a new manifest generation that redirects that column to its new group. Even if INT64 field `b` shares a physical group with `a/c`, the selected direction does not require rewriting that entire group:

```text
Manifest M:
  group-0: columns=[a,b,c], files=[F0_1, F0_2, ...]

Manifest M+1:
  group-0: columns=[a,c],   files=[F0_1, F0_2, ...]  // unchanged files
  group-1: columns=[b],     files=[F1_1, F1_2, ...]  // rewritten b only
```

A column group is metadata describing columns and a list of files, not one file. Each replacement column must retain the complete row alignment: the same row count, order, and segment-local offset-to-row mapping. It is not a sparse list containing only changed cells. Do not sort, merge segments, or physically remove deleted rows during this operation. Predicate inputs may require reading other columns, but reading them does not require rewriting them.

**Reuse boundary.** Preserve unchanged column files and their indexes. Phase one excludes vector-column updates, so unchanged vector data and ANN indexes must continue to be reused rather than rebuilt for an INT64 or other scalar patch. The changed column's old scalar index is not valid for its replacement values and must be maintained consistently with the new base. Coherent serving publication remains required; a manifest switch alone does not implement QN handoff or cache/index refresh.

**Logical invisibility is not space reclamation.** In M+1, normal reads of `b` resolve to the new group; the old `b` in F0 is no longer logically exposed through that manifest. Its physical bytes remain in the unchanged shared files, which are still needed for `a/c` and cannot be deleted to reclaim only `b`. A retained old manifest can still address the old column, and pinned readers must remain coherent. Reclaiming the embedded old bytes requires a later rewrite/replacement of those shared files plus safe reclamation; it is not a benefit delivered by this manifest switch. This does not add an UNDO or long-term time-travel requirement.

**Verified storage primitives, not an implemented Milvus feature.** The current [Milvus dependency pin](../../internal/core/thirdparty/milvus-storage/CMakeLists.txt) resolves to milvus-storage `e6e1ab0817ab85a140f37ba7c920a5e6ebba0d71`. Its [transaction implementation](https://github.com/milvus-io/milvus-storage/blob/e6e1ab0817ab85a140f37ba7c920a5e6ebba0d71/cpp/src/transaction/transaction.cpp#L92) supports removing `b` from the old group's column mapping and adding a new `[b]` group in the same manifest commit. This storage `DropColumn` operation is not a collection-schema DROP FIELD. It removes indexes attached to `b`; replacement-group row counts are checked, but preserving row order is the writer's responsibility. The [column-group reader](https://github.com/milvus-io/milvus-storage/blob/e6e1ab0817ab85a140f37ba7c920a5e6ebba0d71/cpp/src/format/column_group_reader.cpp#L796) projects the columns exposed by each group, so the original shared files need not be rewritten merely to change that mapping.

**Explicit capability boundary.** milvus-storage already provides the primitives to write only replacement `b` data and switch its column mapping through `DropColumn("b")` plus `AddColumnGroup([b], replacement_files)` in one manifest transaction, while leaving the original files for `a/c` unchanged. It does **not** provide an offset-addressed cell-update API such as `update(column, offset_start, offset_end, values)`, nor does it automatically evaluate update deltas or select affected files. Replacing a file means writing a new immutable file and publishing new references, not modifying the original file in place. The caller must produce the replacement values and preserve row alignment.

The existing [DropColumnThenAddSameName test](https://github.com/milvus-io/milvus-storage/blob/e6e1ab0817ab85a140f37ba7c920a5e6ebba0d71/cpp/test/api_transaction_test.cpp#L1228) checks manifest structure, not complete Milvus read/index/handoff correctness. The current Go [ManifestUpdates wrapper](../../internal/storagev2/packed/manifest_commit.go) does not expose the column-drop/replacement combination. **Milvus must still implement the Delta Consolidation workflow:** consume applicable deltas, write replacement column data, connect the Go-side manifest operations, and publish coherent column/index state. End-to-end reuse and retained-reader safety still require verification; the replacement flow has not been exercised end to end in this discussion.

**Local precedent, not an existing update implementation.** The current [schema-bump compactor](../../internal/datanode/compactor/bump_schema_version_compactor.go) has version-only and additive reconciliation paths that retain the source segment identity. Its additive writer creates a separate group for each output field and can add multiple fields in one operation; Delta Consolidation instead replaces the mapping for an existing column. Its dropped-field branch performs a full replacement rewrite, so the analogy is specifically to identity-preserving maintenance, not every schema-bump operation. The current [Mix completion](../../internal/datacoord/compaction_task_mix.go) publishes new segments and queues index construction, which explains the amplification avoided by keeping update consolidation separate.

**Agreed tiered scheduling policy.** Keep the decision to schedule maintenance separate from the delta types consumed during execution:

| Signal | Scheduling or scope decision | Required boundary |
| --- | --- | --- |
| Update-delta byte backlog and retained value-version memory | Determine when to run Delta Consolidation | Do not trigger or accelerate Mix because updates accumulate |
| Scalar column-group fragmentation and obsolete column bytes retained in shared files | Determine the column rewrite and regrouping scope of Delta Consolidation | Preserve segment identity and row count/order/offsets; reuse unchanged vector data and ANN indexes |
| Delete-driven reclamation and existing segment-organization conditions, such as small-segment merging | Determine when to run Mix/full rewrite | Once scheduled, consume all applicable PK-delete, predicate-delete, and update deltas in TimeTick order |

Under the agreed UPDATE semantics, updates change cell values without changing the logical row count; deletes change row visibility and can justify physical row reclamation. Update backlog therefore controls consolidation timing, while CG fragmentation controls its rewrite scope, not an escalation to Mix. Regrouping columns consolidated in separate rounds may require rewriting columns with no new updates in the current round; it is not a free metadata-only merge. It remains layout-preserving column maintenance rather than segment merging.

Small updates can accumulate until consolidation is worthwhile; large-value backlog can justify earlier consolidation. Exact thresholds, memory budgets, and grouping heuristics remain open. A normally triggered Mix consumes updates because they are part of its input state, **not because they triggered or accelerated that task**. Do not use high update-maintenance cost as an alternative automatic Mix trigger. This also preserves the earlier decision not to turn predicate-record backlog alone into an unconditional Mix trigger; its remaining policy use is open.

**Delete boundary.** Delta Consolidation is update consolidation, not physical delete compaction: neither PK nor predicate deletes remove rows or shift offsets in this tier. Physical deletion remains a full-rewrite responsibility. This does not postpone online deletion visibility until compaction; the query-side delete path continues to apply committed deletes.

**Ordered-mutation safety remains mandatory.** An update-only physical output is not permission to ignore interleaved logical deletes or to discard their durable effects. A pending predicate may depend on a field version that Delta Consolidation would otherwise fold away. Delta Consolidation must preserve the meaning of outstanding deletes and later predicates, and must not resurrect deleted rows or claim those deletes have been fully consumed. How to reconcile that requirement with column-only consolidation is an explicit remaining design item; no conversion format, selective folding rule, or new progress protocol is selected here. In particular, do not replay a retained historical predicate against a newly folded value and assume the result is unchanged.

**Industry applicability.** Section 2.7's Kudu precedent supports stable-row-ID column consolidation distinct from RowSet merging. Our L0 dispatch is not Kudu's delta-file merge, and our deltas may contain unevaluated predicates rather than row-addressed changes. Adopt the separation and layout-preservation principle; adapt it to milvus-storage column groups and versioned manifests. Do not copy Kudu's UNDO mechanism or assume its protocol solves Milvus publication and replay. Safe publication, old-reader protection, reuse/GC ownership, exclusion with other maintenance, and the mixed-delete boundary remain unimplemented.

### 9.2 Providing delete-impact statistics to MixCoord

Decision status: **collection boundary settled on 2026-10-19; progress-bound observations and predicate-backlog signals clarified on 2026-10-20; maintenance-tier separation clarified on 2026-10-22**. Collect cumulative applied-delete counts from loaded sealed segments through a dedicated stats RPC with independent polling, as a best-effort enhancement. Missing observations must not block normally eligible compaction. Outstanding predicate counts describe burden; Section 9.1.7 now rules out introducing an extra Mix trigger purely to drain delta backlog. Detailed wire contracts, thresholds, and remaining policy use of the signal remain deferred.

Current single-compaction scheduling uses `Statistics.DeleteNumRows / Segment.NumOfRows`, together with delta-log count and size. `DeleteNumRows` is the sum of `EntriesNum` across delta binlogs, so it is an approximate signal even for PK deletes rather than a count of unique physical rows. See [data_coord.proto](../../pkg/proto/data_coord.proto) and [compaction_trigger.go](../../internal/datacoord/compaction_trigger.go).

For a predicate delta log, one `(expr, delete_ts)` entry may match zero rows or most of the segment. Recording `EntriesNum = 1` therefore makes the current ratio unusable: MixCoord cannot distinguish a negligible delete from an 80% delete without evaluating the predicate or receiving a materialized per-segment impact statistic.

Compaction-time evaluation alone cannot provide an advance estimate of reclaimed rows, because the result arrives after the scheduling decision. Exact hit counts are nevertheless **not a scheduling prerequisite**. Use two complementary observations: applied-delete counts indicate potential row reclamation, while outstanding predicate counts indicate maintenance/replay burden regardless of matches. Neither is the authoritative source of which mutations must be applied, and neither implies permission to override Section 9.1.7's maintenance-tier boundaries.

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
- **Bind the count to complete apply progress.** Conceptually report `(segment identity/incarnation, applied_through, cumulative_deleted_rows)`. For a still-valid, non-compacted-away Sealed segment, the count is exact for that applied state, not an estimate or a timeless current value. Read/publish the count and progress coherently; attaching an independently sampled watermark could claim a state the count does not describe. `applied_through` denotes a complete relevant committed prefix, not the largest message timestamp seen. The exact wire/progress encoding remains open.
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

Retaining the existing loaded-data prerequisite for general predicate deletes means this statistics source does **not** further restrict their current deletion capability. It does not imply that a collection remains loaded after a delete: release, restart, or loading only some partitions can leave statistics unavailable or stale. Collection is therefore best-effort, never a precondition for compaction. Unknown is not zero, and this direction does not claim current exact counts for unloaded segments or offline PK deletes.

Under continued mutations, a cached observation can quickly cease to represent current apply progress. It remains valid for its original progress while its segment identity is valid, but its scheduling freshness is limited. Persisting loaded-time observations to S3 preserves history, not freshness; the added persistence/lifecycle work is not justified for the selected best-effort policy. Retain bounded in-memory observations where useful and invalidate retired identities. Predicate backlog still describes burden when observations are missing, without requiring a durable stats cache or an extra Mix trigger.

Counts are segment-local: statistics for a dropped or compacted-away source segment cease to describe an active scheduling candidate. A replacement segment is observed from its own applied state, not assigned the source segment's deleted count. Ordinary cache invalidation remains necessary; the per-delete file lifecycle is what disappears.

This is distinct from durable mutation coverage in Section 9.4. `{segment_x: deleted_300}` is a scheduling observation, not an executable delete record. If compaction removes those rows, the output must not inherit the old count as outstanding deletion work; if the mutation is still unapplied, discarding its only durable representation would lose correctness regardless of the statistics policy.

#### 9.2.3 Alternatives considered and set aside

The following records the discussion history, not additional requirements for the selected direction:

| Alternative | Why it is not selected for the current direction |
| --- | --- |
| Count each expression before committing it, then persist per-segment increments in etcd | Needs overlap handling, event deduplication, and association with a successful delete; synchronous persistence couples scheduling metadata to delete latency, while asynchronous persistence needs loss recovery |
| Save prepared count sets to S3 and publish them after WAL commit | Avoids a large metastore payload, but introduces prepared-object retention, commit-to-file association, publication recovery, and safe orphan cleanup |
| Persist loaded-time cumulative observations to S3 | Preserves an as-of count but does not keep it current through subsequent mutations or segment replacement; freshness limitations do not justify a new persistence lifecycle for optional scheduling hints |
| Use a durable broadcast/job workflow to coordinate that publication | Offers useful recovery mechanisms, but adds job-like coordination to ordinary predicate deletes solely to make scheduling statistics durable |

For the prepared-count-set S3 alternative, the discussion required channel-independent commit/publication, stable VChannel identity rather than a Delegator or replica identity, and only a small stats reference in WAL rather than a large `{segment -> count}` table. A timeout alone could not authorize deletion of a prepared file whose delete might still commit. These requirements explain its complexity; they are not carried into the selected reporting approach.

The user-provided scale target is **1 million collections and 100 million segments**. Avoid retaining one count set per delete or assuming a dense coordinator-resident table for every segment. The current preference is bounded state for the observed loaded working set; its memory and reporting costs still need measurement. Earlier compressed-set size estimates were illustrative, not a validated capacity model.

#### 9.2.4 Industry precedent and applicability

PostgreSQL exposes estimated dead-row counts, allows cumulative statistics to lag, and resets counters after an unclean shutdown. Its documentation warns that resetting counters used by autovacuum can delay maintenance and cause bloat. See [PostgreSQL cumulative statistics](https://www.postgresql.org/docs/current/monitoring-stats.html).

- **Adopt:** maintenance scheduling can use derived statistics without making those statistics the source of deletion truth.
- **Adapt:** use Milvus's applied, offset-deduplicated segment count, collected from loaded query-side state; it is exact for the observed applied state rather than a raw predicate estimate.
- **Do not copy:** PostgreSQL's storage, persistence, or vacuum protocols. Its behavior does not establish that loaded-only collection or Milvus's report-selection rules are an industry standard.

TiKV's Region heartbeat carries both region metadata and runtime statistics, while its Region epoch represents peer-configuration and split/merge changes. This demonstrates that sharing transport does not require sharing change semantics. See [RegionHeartbeatRequest](https://github.com/pingcap/kvproto/blob/master/proto/pdpb.proto) and [RegionEpoch](https://github.com/pingcap/kvproto/blob/master/proto/metapb.proto). Adopt that distinction, not its Raft protocol or a requirement to combine messages. The dedicated RPC selected here is a Milvus-specific architectural choice, not a claim that different refresh frequencies inherently require separate RPC methods.

RocksDB leveled compaction triggers on L0 file count and uses backlog-based compaction scores; it does not require an exact deleted-row ratio to justify maintenance. See the [official leveled-compaction design](https://github.com/facebook/rocksdb/wiki/Leveled-Compaction), inspected on 2026-10-20.

- **Adopt:** accumulated maintenance work can trigger compaction independently of estimated row reclamation.
- **Adapt:** count outstanding predicate mutation records attached to a Milvus segment, rather than RocksDB SST files.
- **Do not copy:** RocksDB thresholds, levels, or scoring formula. Predicate count is a coarse burden signal, not a measurement of evaluation cost, and this analogy does not equate Milvus L0 with RocksDB L0.

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

#### 9.2.6 Outstanding predicate count: retained signal, revised scheduling boundary

Use the number of predicate mutation records still requiring durable incorporation by data compaction as a hit-independent burden signal, including for unloaded collections and missing/stale QN observations. The earlier proposal used this as a fallback compaction trigger. Under Section 9.1.7's newer tier split, it must not automatically introduce or accelerate Mix/full rewrite; the specific remaining policy use is open. Update-byte backlog is routed to Delta Consolidation, not substituted into a delete ratio or a new Mix trigger.

- Count logical outstanding records, not physical files, replica copies, or distinct expression strings. Separate committed mutations with the same expression remain separate records; duplicate delivery of one mutation must not inflate the count.
- Online QN application does not retire this storage backlog: the segment's durable base/delta state still requires those records during recovery. Reduce the backlog only when durable compaction output and metadata establish that the corresponding records have been consumed, not when a worker merely finishes evaluation.
- The signal also applies when a valid QN observation reports zero deleted rows. Processing 100 predicates that match nothing can still eliminate 100 future replay steps; it is not evidence that 100 rows were deleted.
- Treat applied-delete count and predicate backlog as complementary observations, not mutually exclusive branches. A zero applied-delete count does not mean zero replay work; it also does not by itself authorize a new Mix trigger.
- This is a deliberately simple initial policy, not a claim that expression count is proportional to CPU/I/O cost or the only possible metric. Thresholds, prioritization against other compaction triggers, resource budgets, and anti-starvation behavior require validation.

#### 9.2.7 Deferred details; no detailed design yet

Remaining work is to define the RPC name and request/response fields, coherent count/progress sampling, batching and refresh parameters, collector ownership inside MixCoord, source/progress validation across replicas and reloads, and cache bounds. Predicate count remains a burden measurement; its durable accounting and policy use must be reconciled with Section 9.1.7's normal-Mix scheduling boundary. Statistics loss may delay precise reclamation prioritization, but must not block normally eligible maintenance, cause missed delete application, or change query visibility.

This checkpoint settles the **statistics source, progress-bound observation model, collection-interface boundary, and hit-independent burden signal**, not the final scheduling policy, compaction-side expression evaluator in Section 9.1, or durable expression replay. No code implementation is authorized or implied by settling this part of the design.

### 9.3 Target-QN load phase: access paths and progress-gated asynchronous recovery

Decision status: **access-path policy settled on 2026-10-19; asynchronous recovery and per-request visibility direction recorded on 2026-10-20; not implemented**.

**Scope.** This section covers the target QN loading segment data/index resources and applying the segment's own delta logs. The asynchronous extension also states the read-admission contract that this progress must support. Delegator L0 forwarding, streaming-tail catch-up, and publication coordination remain required integration boundaries; their detailed protocols are not redesigned here.

**Agreed execution policy.** The load/recovery path recognizes persisted predicate records and invokes the existing Segcore expression execution framework, followed by delete-specific timestamp/visibility filtering and `ApplyDeletesMap`. Preserve each record's original `delete_ts` and load/replay bookkeeping. Do not expand matches into PKs or require a separate eager load of all fields referenced by the predicate before evaluating it.

- **Use suitable scalar indexes.** When the existing expression path can answer the predicate exactly through scalar indexes, including required NULL semantics, use their result bitmaps without reading the corresponding raw columns for predicate evaluation. For example, `c > 18 && a == 18` can combine the two indexes' result bitmaps with AND when both operations are supported.
- **Otherwise access required data on demand.** Reuse existing raw-column access, lazy materialization, chunk access, and prefetch behavior. Mixed index/raw execution is valid. Do not change a field's lazy-load configuration solely because it participates in a predicate delta.
- **Keep dependencies explicit.** This is independence from preloading raw column contents, not independence from the segment's schema, stable row layout, index/data availability, or MVCC timestamp information. Indexes themselves may require I/O, and candidate-only indexes may still require raw-data rechecks.
- **Separate base load from read readiness for either lazy-load setting.** The agreed recovery architecture uses background predicate application with both `lazy-load=false` and `lazy-load=true`; it is not a lazy-load-only workaround. Start evaluation when its required schema/layout/data or index access is ready, and overlap remaining work where dependencies allow. Base loading may return before full predicate catch-up, but queries may proceed only when their consistency requirements are satisfied by a valid readable state. This is not a first-query-triggered replay mechanism or permission to advertise unapplied mutations as visible. Rollout/configuration details remain open; neither eager data loading nor asynchronous apply replaces Section 9.1.3's durable drain requirement.

**Existing implementation basis.** [segment_loader.go](../../internal/querynodev2/segments/segment_loader.go) loads segment resources and then its delta logs. [UnaryExpr.cpp](../../internal/core/src/exec/expression/UnaryExpr.cpp) selects scalar-index or raw-data evaluation, while [ConjunctExpr.cpp](../../internal/core/src/exec/expression/ConjunctExpr.cpp) combines child results. [Expr.h](../../internal/core/src/exec/expression/Expr.h) gates raw-data prefetch on the selected execution path. [ChunkedColumnGroup.h](../../internal/core/src/mmap/ChunkedColumnGroup.h) provides lazy reader/cache initialization and chunk access. These are reusable capabilities, not an already implemented predicate-delta loader.

**Industry precedent.** PostgreSQL's [multiple-index bitmap combination](https://www.postgresql.org/docs/current/indexes-bitmap-scans.html) supports the mature principle of combining per-index row selections. Adopt that principle and adapt it to Milvus's segment-local bitmaps and timestamped deletion state; do not copy PostgreSQL's heap-access or transaction workflow, or infer Milvus performance from it.

**Latency and memory boundary.** Avoiding an additional forced eager-load stage neither disables existing prefetch nor guarantees bounded peak memory: the reused raw-data path may prefetch many chunks. Synchronous predicate application places cold-column reads on base-load latency; asynchronous application moves them to background catch-up and potentially the waiting queries' critical path. Compare expression evaluation plus delete bookkeeping against PK-delta reading, decoding, PK-to-offset lookup, and the same bookkeeping. Index-only execution can avoid predicate-column reads, but no load-time speedup or memory reduction has been measured. Resource budgeting and exact loader/CGo wiring remain implementation work; the access-path policy above is settled.

**Joint-design qualification.** When update deltas are present, an otherwise suitable scalar index must also provide the correct update-visible selection. Section 6 governs ordered delta application and field-value visibility; Sections 10.6–10.7 specify the agreed versioned-value and query-merge direction. Reusing the existing expression framework does not, by itself, implement either capability.

#### 9.3.1 Primary precedent: PostgreSQL hot standby

PostgreSQL 18 documents three distinct stages:

1. Before recovery reaches a consistent, snapshot-readable state, the standby does not accept queries.
2. Once that state exists, read-only queries and WAL replay can proceed concurrently; the standby can still lag behind the primary.
3. Ordinary standby queries use the available snapshot. They do **not** automatically wait for full catch-up, and the first query has no special catch-up requirement. Replaying a transaction's commit makes its changes visible to new snapshots.

Sources: [Hot Standby — User's Overview](https://www.postgresql.org/docs/18/hot-standby.html#HOT-STANDBY-USERS) and [Administrator's Overview](https://www.postgresql.org/docs/18/hot-standby.html#HOT-STANDBY-ADMIN).

**Adopt:** distinguish a consistent readable state from being fully caught up; allow recovery and reads to coexist.

**Adapt:** Milvus queries carry freshness requirements through `GuaranteeTs`; the proposed gate checks relevant apply coverage against each request.

**Do not copy or misattribute:** PG's physical recovery, transaction snapshots, or readiness conditions are not Milvus's segment-loading protocol. A per-query `GuaranteeTs` wait is our adaptation, not PG's ordinary hot-standby behavior. This precedent does not justify exposing an arbitrary partially reconstructed segment.

#### 9.3.2 Agreed direction: per-request apply-progress visibility

Maintain complete apply progress, and evaluate admission against each query's minimum visibility requirement. Conceptually:

```text
valid readable state exists
AND complete relevant apply progress >= query.GuaranteeTs
    -> the freshness gate is satisfied
otherwise
    -> wait for the required progress, or use the defined routing/error policy
```

With relevant progress at 100, requests requiring 95 or 100 satisfy this gate; a request requiring 105 waits until progress reaches 105. It does not need to wait for work after 105 or for the backlog to become empty. This rule applies to every request, not only the first query after recovery.

The progress value is a **complete applicable prefix**, not the maximum timestamp of a processed predicate. It must cover all relevant mutations and dependencies, including PK deletes and updates when present. A query spanning multiple segments/VChannels cannot use only the fastest participant's progress. Exact progress ownership, aggregation, and integration with the existing serviceable-time machinery remain open.

`GuaranteeTs` is a minimum visibility requirement, not an exact historical snapshot timestamp. Satisfying this gate does not promise arbitrary time travel or remove other schema, MVCC, and resource-readiness checks. The loader must establish a valid readable state; assigning a low progress value alone cannot manufacture one.

Current [search admission and `waitTSafe`](../../internal/querynodev2/delegator/delegator.go) already compare serviceable progress with request guarantees, and [Proxy search preparation](../../internal/proxy/dql/task_search.go) derives those guarantees from consistency settings. These are integration precedents, not proof that the current timestamp accounts for deferred predicate application; it must not advance past incomplete required work in the new mode.

#### 9.3.3 Consistency and recovery boundaries

The proposed mode separates base-load completion, request-specific read readiness, and full mutation catch-up. It does not promise that every query becomes available when base load finishes.

- An explicitly permitted weak read may see stale data, including rows deleted before the restart reappearing until replay applies their deletion. A Strong/Session request, or a Bounded request whose requirement exceeds progress, must not silently receive that weaker result.
- Background application must preserve Section 6's committed VChannel mutation order. Delaying `UPDATE age = 20` and then evaluating a later `DELETE WHERE age < 18` against the old value can permanently diverge; this is not acceptable temporary staleness.
- Async execution does not remove cold I/O, CPU, memory pressure, or query interference. Backlog progress and failure must be observable; calling the stale period "brief" requires measured catch-up capacity.
- This read-side progress is not a storage durability checkpoint and cannot substitute for the Streaming/L0-to-target drain required before physical field removal.

**Still open:** configuration/defaults and rollout, establishment of the initial readable state, progress publication during partial application, binding progress to the correct segment/view generation, crash/retry behavior, routing and timeout policy, and upgrade/readiness integration. Shorter base-load latency alone does not establish shorter strong-read recovery, absence of rolling-upgrade timeouts, or 99.99% service availability.

### 9.4 Predicate L0 dispatch: coarse scope without ingress segment associations

Discussion checkpoint: **2026-10-22**. The responsibility boundary, conservative-routing tradeoff, independent dispatch batching, and complete/ordered dispatch requirements are agreed. Payloads, publication/recovery details, and performance remain unvalidated.

#### 9.4.1 The missing problem and the existing reuse boundary

Removing Proxy's matching-PK query also removes the PK list used for L0 Bloom-filter dispatch. A predicate such as `pk > 5` cannot use a PK membership Bloom filter to discover all matching segments. The same issue applies to `UPDATE ... SET age = 18 WHERE pk > 5`: constant assignment simplifies the patch, not target discovery.

The current implementation, inspected at `fa995414c1`, separates:

1. **Candidate selection:** `selectFlushedSegment` / `BuildCompactionRequest` in [compaction_task_l0.go](../../internal/datacoord/compaction_task_l0.go) select targets from current storage metadata using collection, channel, partition, lifecycle, and data-time bounds.
2. **PK filtering:** `applyBFInParallel` in [l0_compactor.go](../../internal/datanode/compactor/l0_compactor.go) tests deletion PKs against the candidates' Bloom filters. Even this is an approximate candidate test, not proof of exact row matches.

Reuse the first responsibility; do not pretend arbitrary predicate evaluation can be replaced by the second. Likewise, `pk > 5` does not identify one PK-hashed VChannel merely because it references the PK. Safe narrower routing for special predicates is an optimization, not a requirement for the general path.

The existing [compaction inspector](../../internal/datacoord/compaction_inspector.go) excludes concurrent L0 and Mix/Sort/Bump execution on the same channel, with corresponding exclusion for Clustering; same-channel behavior is covered by [scheduler tests](../../internal/datacoord/compaction_inspector_test.go). This is **execution exclusion, not a guarantee that all pending L0 or Streaming mutations have drained before a sealed rewrite**. Reusing it does not by itself prove the new predicate/update protocol. Section 9.1.6 settles the rewrite's consumed-delta ownership and separates this stage from QN load/handoff.

#### 9.4.2 Rejected: request-time segment association files plus lineage repair

The rejected approach determines segment associations when a predicate enters, persists them to S3, and later uses that table as authoritative dispatch information. It is not selected for the following reasons:

- **The initial list can already be stale.** Storage compaction publishes replacement SegmentMeta before query-side handoff completes. A request may still see `X, Y` while storage already exposes `Z`; no assumption limits the discrepancy to one replacement generation. See [mix task publication](../../internal/datacoord/compaction_task_mix.go), [SegmentMeta replacement](../../internal/datacoord/meta.go), and [query target readiness](../../internal/querycoordv2/observers/target_observer.go).
- **Persisting a snapshot does not maintain it.** After `X + Y -> Z`, a file naming X does not automatically express which outstanding mutations Z still needs. A file uploaded successfully to S3 may therefore be durably obsolete.
- **Repair becomes a separate distributed protocol.** Following descendants, handling merges/splits and multiple generations, coordinating replacement publication, retries, recovery, and retention would be required by that approach. Moving the table from etcd to S3 changes storage cost, not this consistency burden.
- **Counts do not repair applicability.** Per-segment deleted counts are scheduling hints, not mutation payloads or mappings of affected rows; Section 9.2's reconstructible stats must not become the source of dispatch truth.

**Decision:** reject this approach under the project's engineering-complexity constraints, not as a claim that all lineage-aware systems are theoretically impossible. Do not add a side association table and then compensate for its staleness with an ancestry-tracking service.

#### 9.4.3 Selected responsibility split

```text
Committed predicate mutation: scope + logical timestamp + predicate [+ constant patch]
  -> durable channel/partition-scoped L0
  -> choose candidates from current storage metadata at L0 dispatch
  -> publish predicate deltas as part of the candidates' durable segment state
  -> evaluate in online/load execution or background data compaction
```

This is a lifecycle model, not a wire format. Section 9.5.2 separately defines one native mutation message per target VChannel.

- **Before dispatch:** retain the scoped predicate in durable L0, without freezing the request's QueryView segment list.
- **During dispatch:** select the current candidate segments and publish their predicate deltas under a validated scheduling/publication boundary. Prefer extending existing L0/compaction coordination over repairing stale ingress associations.
- **After dispatch:** the predicate is an input of the segment's normal delta lifecycle. Subsequent data compaction must evaluate applicable deltas in mutation order and incorporate their effects into its output. The selected direction does not add a second mechanism for migrating an unapplied ingress association to descendants.
- **If a rewrite completes first:** later L0 dispatch selects current replacement segments within the predicate's scope. It does not resolve the original QueryView IDs through a chain of successors.

L0 must remain a **lightweight distribution stage**: no expression execution, no reading business columns to determine matches, and no dependency on QN availability or a loaded collection. Online/load evaluation belongs to the query execution path; offline data compaction reads persisted data independently of query loading. The concrete update evaluator and representation remain open.

Release after mutation acceptance must not make L0 wait for reload. Outstanding predicates can remain in segment-owned durable deltas until offline compaction or a later load evaluates them. This does not change the separately retained admission prerequisite for accepting general predicate deletes in Section 1.1.

**Durability is still mandatory.** WAL commitment alone is not completed L0 dispatch. Predicate content and the metadata that makes it discoverable for the current targets must be durably published before dispatch is reported complete or the source is retired. The agreed high-level organization is a new segment-owned predicate delta file type batching multiple records, not a separate metastore association entry for every predicate/segment pair. This does not require exactly one physical file for the segment's entire lifetime; encoding, batching/rolling, and publication/cleanup remain open. The no-side-table decision removes ingress lineage repair, not ordinary segment metadata, completion accounting, or garbage-collection obligations.

**Complete input prefix is mandatory.** L0 delta compaction must acquire all PK-delete, predicate-delete, and update deltas in its selected interval within the VChannel's published prefix; it must not independently schedule whatever type's files happen to be visible first. Section 9.7.2 defines the required atomic visibility boundary; Section 9.4.5 separates it from dispatch batching and completion. L0 remains lightweight dispatch, not expression evaluation. Target-segment publication must also preserve completeness for later load/data compaction; atomic input publication alone does not prove the downstream handoff correct.

#### 9.4.4 Explicit tradeoffs and rejected alternatives

| Choice | Benefit | Cost or reason not selected |
| --- | --- | --- |
| Query matching PKs before WAL append | Retains the existing PK-based routing inputs | Restores the PK materialization/transfer the native predicate path is intended to avoid; remains the legacy fallback |
| Evaluate predicates inside L0 compaction | Can determine actual matches before emitting target deltas | Rejected: turns lightweight dispatch into field reads and expression execution |
| Ask loaded QNs to resolve L0 targets | Can use already loaded data/indexes | Rejected as a dispatch dependency: collection release or QN unavailability would stall progress |
| Persist request-time segment associations and repair them across compaction | Attempts narrower targeting | Rejected: stale-at-creation QueryViews, repeated replacement, and lineage/publication complexity |
| Coarse channel/partition/data-time dispatch | Avoids the ingress association table, ancestry repair, and loaded-QN dependency | Selected: more candidate segments receive predicates, including segments with zero matching rows |

**Accepted tradeoff:** give up exact matched-segment dispatch in exchange for a simpler lifecycle. Persisting the association does not itself make dispatch selective; without evaluation or trustworthy exclusion evidence, all candidates in scope must remain eligible. This is scoped fanout, not indiscriminate delivery to every collection or to data inserted after the mutation.

Metadata pruning is optional. For example, complete valid PK bounds might exclude a segment from `pk > 5`; missing bounds cannot justify exclusion. For mutable fields, base statistics alone are unsafe: a preceding update may change `age = 17` to `20`, so old `max(age) = 17` cannot exclude that segment from a later `age > 18` predicate. Replacement-segment creation time is likewise not the logical insertion time of its rows.

The cost moves from matched-PK materialization toward candidate dispatch and repeated evaluation. If mutation `m` has `C_m` candidate segments, record delivery can scale with `sum(C_m)`, but batching records into segment-owned files avoids implying one metastore key per delivery. The remaining metadata is normal segment delta-file discovery/publication and completion/retention information, not an ingress matched-segment table. File batching reduces metadata overhead, not the number of candidate evaluations.

Extra expression execution during background data compaction is an accepted tradeoff, including zero-match work; it is not on a foreground request's latency path, although it still consumes shared resources. Cold load/recovery is the more latency-sensitive concern: many predicates across many fields may trigger substantial index/raw-data reads, even on zero-match segments. Section 9.3 addresses that concern by decoupling base loading and ordered apply with a per-query progress gate, not by claiming the I/O disappears. Fanout, file/metadata growth, and recovery costs still need validation against Section 9.2.3's scale target.

#### 9.4.5 Agreed: independent dispatch batches with complete, ordered coverage

**Publication and dispatch have different progress and batching responsibilities.** For one VChannel, let `P` denote the complete, durably published L0 delta prefix certified by `delta_prefix_barrier[v]`, and let `C` denote the prefix completely and durably dispatched to all applicable target segments. They satisfy `C <= P`; neither is QN apply progress. Here `C` is dispatch completion, not SN's previous publication cut used as a request bound in Section 9.7.

SN publication boundaries do not prescribe MixCoord dispatch boundaries. Within the available published prefix, MixCoord may split a publication batch or combine records from several publication batches:

```text
SN publication:     (0,100]           (100,200]
MixCoord dispatch:  (0,50]   (50,150]   (150,200]
```

These are logical intervals, not a requirement to split or rewrite physical input files at those timestamps. Dispatch through 150 requires publication through at least 150; publication through 200 does not require dispatching the entire second SN batch at once. The dispatch-completion boundary need not coincide with any SN publication boundary.

**Three agreed dispatch requirements:**

1. **Contiguous, complete input.** A selected next interval `(C,T]`, with `C < T <= P`, contains every applicable PK-delete, predicate-delete, and update record in that interval. A type with no applicable records is empty, not unfinished. Count, size, partition, or file batching cannot create a missing dependency or silently omit part of the selected interval.
2. **Ordered downstream visibility.** A later interval must not become usable by downstream load/data compaction across missing applicable work in an earlier interval. Physical preparation and tasks need not all run serially; out-of-order physical completion is not permission for out-of-order logical visibility. Advance `C` only when the complete preceding dispatch work and its durable target metadata are finished, not when one worker or one subset finishes.
3. **Complete target coverage.** Deliver the interval's applicable deltas to every required target selected under the current-storage-view and scope rules. Do not declare VChannel dispatch complete while a required target is missing. This preserves conservative fanout, not precise match discovery or ingress QueryView associations.

L0 remains lightweight instruction distribution without expression evaluation. These requirements specify the scheduler's input and downstream visibility contract, not a new requirement to mirror SN sync tasks, use one task for the whole VChannel, or serialize every physical write. Completion-state representation, task decomposition, atomic visibility mechanics, retry/recovery, and target changes during dispatch remain implementation-design work.

**Current-code boundary.** The PK-only [L0 view](../../internal/datacoord/compaction_l0_view.go) uses count/size thresholds and input slice order in ordinary `Trigger`; `ForceTrigger` and `ForceTriggerAll` sort available candidates by `dmlPos`. The [L0 policy](../../internal/datacoord/compaction_policy_l0.go) groups by partition/channel and filters against the earliest growing-segment start position. Neither candidate sorting nor this eligibility filter certifies that all earlier deltas have arrived or that dispatch completes in prefix order. The complete-prefix contract is new, not a guarantee already supplied by PK L0 scheduling. Section 9.7.2 explains why fixed-PK timestamped deletions differ from order-dependent update/predicate evaluation.

**Industry precedent and applicability.** Kafka's [consumer isolation](https://kafka.apache.org/41/configuration/consumer-configs/#isolation.level) limits ordered `read_committed` delivery to the last stable offset, while [max.poll.records](https://kafka.apache.org/41/configuration/consumer-configs/#max.poll.records) independently bounds each poll's returned records and does not change underlying fetch batching. Adopt the separation of a safe visibility frontier from consumption batch size; adapt it to Milvus's published and dispatched delta prefixes. Do not import Kafka's transaction/offset protocol or treat it as proof of atomic fanout to multiple target segments. The three dispatch requirements above are this design's agreed contract; their implementation is not established by the analogy.

#### 9.4.6 What is settled and what is not

**Settled:** no ingress QueryView-derived target table or dedicated lineage-repair mechanism; coarse scope-based dispatch against current storage metadata; no expression execution or loaded-QN dependency in L0; segment-owned predicate delta files batching multiple records; publication and dispatch batches may differ; dispatch preserves contiguous complete input, ordered downstream visibility, and all applicable targets; published and dispatched progress are distinct; actual evaluation in online/load and background data-compaction paths; acceptance of possible zero-match work.

**Still open:** exact predicate-file encoding and segment metadata layout, batching/rolling and cleanup, retry/restart and partial-publication behavior, the implementation of the agreed ordering/completion frontiers across PK deletes/predicate deletes/updates, coverage of old growing data and replacement outputs, and cost limits. Existing task exclusion and target selection are reuse candidates, not an end-to-end proof. The drop-field drain in Section 9.1.3 still needs its complete durable-prefix guarantee; coarse dispatch does not weaken it.

**Separate legacy risk:** Section 12.1 tracks multi-batch catalog publication of an L0 task result. Retain task-result publication as the intended logical completion boundary, but do not assume the current implementation provides crash-atomic publication for an arbitrarily large target set. Resolving that implementation risk is separate from the agreed coarse-routing and ordering decisions above.

### 9.5 Shared logical WAL message and expression representation

Discussion checkpoint: **2026-10-21**. Share the logical envelope, header identity, and mutation target between predicate delete and update. Update adds assignments. This settles the responsibility split, not message enum values, protobuf field numbers, or a requirement to use one concrete protobuf type.

```text
Properties — reuse the existing WAL envelope
  message_type: PREDICATE_DELETE | UPDATE       // conceptual names
  message_version: ...
  vchannel: ...
  timetick: ...                                // WAL-assigned
  txn_context: ...                             // existing internal envelope only; not a user transaction API

Header — common identity and schema-validation contract
  collection_id: ...
  schema_version: ...

Body — common mutation target
  partition_scope: ALL | PARTITION_IDS(...)
  predicate: BoundTypedExpr<Boolean>

Body — UPDATE only
  assignments:
    - field_id: ...
      value_expr: BoundTypedExpr<TargetCompatibleType>
```

- **Reuse infrastructure metadata, not a second clock.** Retain the existing WAL properties rather than duplicating them in the body. Preserve the effective committed mutation timestamp through L0, loading, and compaction, including existing transaction timestamp semantics where applicable. Section 6's committed logical order is authoritative; physical append arrival order is not a replacement for it.
- **Share scope and selection, distinguish the operation explicitly.** `partition_scope + predicate` is the common logical target. An update is not inferred from the presence of assignments, and an empty assignment list must not mean delete. Schema version binds field interpretation; it is not a substitute for payload/expression semantic compatibility.
- **Persist a bound, typed logical expression.** Resolve field names to IDs and types and bind literal/template values before persistence. Do not persist only raw expression text or a segment/index-specific physical execution plan. A restricted subset of [plan.proto](../../pkg/proto/plan.proto) is a candidate representation; reusing its nodes does not authorize every expression accepted by the query engine.
- **Keep business content in the body.** Predicate literals, assignment values, and partition-ID lists belong in the payload, not an unbounded specialized header. Do not carry matched PK lists, segment IDs/offsets, per-segment counts, or a prepared stats table. A PK literal inside a predicate is legitimate input; a query-expanded list of matches is not the proposed payload.
- **Do not fabricate affected-row counts.** Predicate admission does not know the number of matching rows. The existing PK-delete header's `rows` field must not be populated with an expression count or with zero as if no rows were deleted. Public result/count semantics remain a separate open contract.
- **Gate capabilities before append.** Typed syntax and protobuf decodability do not prove that all required online/load/compaction consumers implement the same semantics. Unsupported predicate deletes retain the pre-append legacy PK route; an UPDATE with an unsupported WHERE or SET is rejected before append, without an automatic upsert or insert-plus-delete fallback. Section 9.6.2 records the routing boundary. Section 9.5.1 settles the Proxy/StreamingNode responsibility principle; detailed checks and cross-version admission remain open. Committed unknown nodes must not be skipped.

**Current-code cross-check:** the [message model guide](../agent_guides/streaming-system/message/message.md), [properties.go](../../pkg/streaming/util/message/properties.go), and [messages.proto](../../pkg/proto/messages.proto) distinguish shared properties, small infrastructure headers, and business payloads. Existing Insert headers carry schema versions, while the PK Delete header carries collection ID and row count; a common new mutation schema header is proposed, not already implemented. [Transaction assembly](../../pkg/streaming/util/message/builder.go) applies the commit TimeTick to transaction bodies. These are reuse precedents, not evidence of existing predicate-delete/update message support.

#### 9.5.1 Agreed validation responsibility principle

Discussion checkpoint: **2026-10-21**. **Proxy validates that the request is legal under version V; StreamingNode validates that version V remains admissible at the mutation's commit boundary.**

Here, V denotes the relevant schema and expression-capability versions, not a decision to merge them into one version number or introduce a new wire field. Static validation means that its result is stable for the bound request and those fixed versions: syntax, field/type binding, assignment restrictions, and supported-expression eligibility. It does not mean that the live schema or enabled capability set can never change.

Proxy may also precheck mutable conditions to reject early, but such checks cannot replace authoritative write-side admission. For example, a request valid under a schema containing field `c` must not be admitted after the applicable drop-field boundary merely because Proxy checked it earlier. This does not require StreamingNode to reparse or execute the predicate/assignments.

The existing [Insert append handler](../../internal/streamingnode/server/wal/interceptors/shard/shard_interceptor.go) and [schema-version check](../../internal/streamingnode/server/wal/interceptors/shard/shards/shard_manager_collection.go) provide a local precedent for write-side version validation. Their legacy compatibility exceptions are not automatically inherited by the proposed mutation types. This checkpoint settles the responsibility rule only; concrete checks, version negotiation, retry behavior, and synchronization are deliberately deferred.

#### 9.5.2 Agreed request unit and internal-transaction boundary

SDK -> Proxy -> StreamingNode accepts **one logical native UPDATE or predicate DELETE per request**, producing one corresponding message per target VChannel. A batch of independent UPDATE/DELETE statements is not supported. Independent requests may still arrive concurrently; the restriction concerns request composition, not request concurrency. This restriction does not prevent one WHERE from matching many rows or one UPDATE from carrying multiple assignments; those assignments remain in the same mutation payload. Buffering multiple independently committed requests into one persistence file is not a batch mutation API.

Matching millions of rows does not itself enlarge an expression-plus-patch message into millions of row records. Do not introduce transaction splitting based on the number of matched rows, or a transaction-internal operation ordinal for the new single-mutation path. Concrete payload-size limits remain implementation work; this agreement does not promise unbounded expressions or assignment values.

Existing internal transactions serve a different purpose: a single-VChannel insert split into multiple messages can commit as one visible group, and existing upsert combines insert/delete messages through the same infrastructure. Atomic committed visibility does not mean simultaneous physical writes or cross-VChannel atomicity. In particular, the current producer sends transaction bodies concurrently; it is not a user-specified sequence of SQL statements.

**User-controlled DML transactions are a non-goal**, not an assumed future extension of this design. Retain internal transactions needed by existing write correctness without exposing BEGIN/COMMIT or mixed UPDATE/DELETE transaction batches. The references to transaction timestamps elsewhere describe infrastructure compatibility, not an additional public capability.

**Current-code cross-check:** [VChannel grouping](../../internal/distributed/streaming/util.go), [producer autocommit/transaction selection and concurrent body append](../../internal/distributed/streaming/internal/producer/producer_task.go), [insert packing](../../internal/proxy/dml/task_insert_streaming.go), and [upsert append](../../internal/proxy/dml/task_upsert_streaming.go). The new native request restriction is a design decision, not a claim that these message types are already implemented.

### 9.6 Assignment representation and phase-one admission

Use `field_id + value_expr` rather than a constant-only field that must later be replaced. A constant is the simplest expression node; phase one restricts admission, not the long-term shape of the assignment.

```text
Phase 1: SET age = 23
  Assignment(field_id=102, value_expr=Literal(Int64(23)))

Later core: SET age = age + 1
  Assignment(field_id=102,
             value_expr=Add(Column(field_id=102, type=INT64), Literal(Int64(1))))
```

The notation is conceptual, not an existing protobuf constructor. `WHERE` must produce a boolean selection, whereas `value_expr` must produce a value compatible with its target field; they may share a typed node vocabulary without sharing the same admitted operator/type set. [plan.proto](../../pkg/proto/plan.proto) already contains `ValueExpr`, `ColumnExpr`, and `BinaryArithExpr`, but neither their presence nor `CallExpr` supplies a ready update evaluator or a durable semantic-version contract. Explicit NULL encoding still needs design; it must not be inferred from an omitted assignment or a protobuf default.

Section 10.8 defines the agreed capability boundary. Binary compatibility, exact type/conversion rules, and cross-version feature admission remain detailed work; the expression-shaped representation does not itself enable phase-two calculations.

#### 9.6.1 Agreed execution split: WHERE selection, assignment values, mutation apply

Discussion checkpoint: **2026-10-21**. Use separate responsibilities and interfaces, not a requirement for two independent general-purpose execution frameworks:

```text
Shared WHERE evaluator -> selected segment-local offsets
                           |-> DELETE -> delete application
                           `-> UPDATE -> assignment values -> field-delta application
```

- **WHERE evaluation is shared by DELETE and UPDATE.** Use the same supported predicate/type combinations, boolean/NULL semantics, and snapshot-correct row selection. In the Go compaction design, both operations reuse one restricted predicate evaluator rather than maintaining divergent WHERE implementations.
- **Assignment evaluation belongs only to UPDATE.** For selected rows, produce typed values for explicitly assigned fields. Do not force value production through a boolean-selection interface or put field values into `DeletedRecord`.
- **Phase one needs only constant filling.** Each selected row receives the same typed literal patch, including explicit NULL where allowed. Computing RHS values does not require reading old values; WHERE evaluation and versioned patch application may still require data/state access. Do not introduce a general assignment executor just to broadcast constants over selected offsets.
- **Extend with a restricted value evaluator later.** Same-row references and arithmetic can add typed value production, such as `age = age + 1`, without changing the selection/application split. Read all RHS inputs from the applicable pre-mutation row view, including earlier committed updates, not partially applied assignments from the same SET. Section 10.8 remains the capability ceiling, not full SQL.
- **Reuse infrastructure without merging capability profiles.** Types, literal representation, versioned field access, and suitable expression primitives may be reused. `WHERE_supported_set` and `SET_supported_set` remain independently validated; an operation supported in one context is not automatically supported in the other. Logical semantics must agree across Segcore and Go compaction even where implementation code cannot be shared.

Section 2.10 records the DuckDB Filter/Projection precedent and its limits. Concrete operators, batching, memory ownership, and publication mechanics remain implementation-design work; the names above describe responsibilities, not existing classes.

#### 9.6.2 Pre-append support-routing contract

| Valid request under the existing API/schema rules | Routing decision before append |
| --- | --- |
| DELETE WHERE outside the native predicate subset | Use the existing query-PK then PK-delete path, subject to its existing limits |
| UPDATE WHERE outside the shared predicate subset | Reject; do not silently convert to upsert or insert-plus-delete |
| UPDATE WHERE supported but SET outside the assignment subset | Reject |
| All expressions supported for the requested operation | Append the native mutation without querying row data |

Validate the entire bound expression and its compositions, not only its top-level operator. This table concerns unsupported capabilities, not transient append failures or runtime expression errors. Do not apply a supported subset, change routes after partial commitment, or treat an append timeout as proof that no mutation committed. Section 9.5.1 supplies the agreed validation responsibility principle; the detailed check allocation and compatibility protocol remain open.

### 9.7 Agreed persistence-buffer and separate-delta ordering

Discussion checkpoint: **2026-10-21**. This settles responsibilities and record-level ordering, not a concrete buffer class, binary file format, storage path, or manifest protocol. It adapts the existing ordered WAL consumption and persistence ownership; Section 2's base-plus-delta precedents do not supply this Milvus-specific protocol.

- **Buffer commands, not a collapsed final state.** Retain predicate, scope, and effective mutation TimeTick for a predicate delete; retain those plus assignments for an update. Preserve collection/schema/expression interpretation from the admitted message. Do not regenerate timestamps when flushing.
- **Do not execute in the persistence buffer.** Buffering neither evaluates expressions nor expands matching PKs/offsets. Online forwarding and load/recovery catch-up belong to the query-side consumer/pipeline, separately from persistence buffering and L0 materialization. This does not prohibit the online pipeline from applying updates to serving Growing data.
- **Keep intermediate mutations.** `UPDATE age=23@100 -> DELETE WHERE age<24@110 -> UPDATE age=25@120` cannot be reduced to the last assignment before evaluating the delete. Physical batching/compression must preserve the ordered commands and timestamp visibility.
- **Use separate update and predicate-delete file types.** Each type can batch multiple records. These are durable mutation instructions, not an ingress-time segment association table or already-evaluated offset results. After L0 dispatch, the files belong to the candidate target segments under Section 9.4's scope/publication rules.
- **Keep buffers and files ordered, then merge across types.** The existing WAL consumer supplies reordered committed messages; type-based demultiplexing preserves ordered subsequences, so the new buffers need not independently sort physical WAL arrivals. Write each update/predicate-delete file in effective TimeTick order and retain each record's timestamp. Readers merge applicable records across types into the logical order used online. File creation time, listing order, and file-level min/max timestamps are not substitutes; physical encoding and merge mechanics remain open.

Illustrative file names, not specified storage keys:

```text
update.delta:           UPDATE@100, UPDATE@120
predicate_delete.delta: DELETE@110

Application order:     UPDATE@100 -> DELETE@110 -> UPDATE@120
```

**Existing buffer precedent:** the [insert buffer](../../internal/streamingnode/server/wal/vchannel/segment/insert_buffer.go) appends retained messages supplied by the ordered consumer; it does not independently sort physical WAL records. The [WAL L0 materializer](../../internal/streamingnode/server/wal/vchannel/l0materializer/wal.go) separately owns retained delete inputs until output and registration complete; its [design](design_docs/wal/l0_materializer.md) describes that handoff. Reuse these ownership and timestamp principles; neither component currently implements the proposed native update/predicate buffers or file types.

**Still open:** record encoding, batching/rolling limits, concrete durable publication/discovery, retained-input integration, and recovery/GC mechanics. Sections 9.7.3–9.7.4 settle the publication interface, result semantics, and recovery sequence, not their implementation. A later available delta record is not proof that all earlier mutations are available. This discussion does not turn append-confirmation watermarks into downstream persistence checkpoints.

#### 9.7.1 Shared sync triggers and the common-cut tradeoff

Reuse the existing PK-delete sync trigger categories: buffered-byte capacity, oldest-pending age, explicit Flush/lifecycle boundaries, and recovery/pressure-driven persistence requests. Account for actual predicate/assignment/PK payloads, not unknown matching-row counts. Exact limits and tuning remain open; sharing the policy does not mean all three buffers must independently reach their thresholds.

The agreed direction is a common sync cut T within one VChannel's consumed, complete logical prefix. If the previous published cut is C, include every pending delta in `(C, T]` across **PK deletes, predicate deletes, and updates**. Records after T belong to a later batch. Files may be built and uploaded separately or concurrently; publication must still satisfy Section 9.7.2.

```text
Pending deltas in (C, T]:
  update:           100 records -> sync all 100
  PK delete:          2 records -> sync both, even below the byte threshold
  predicate delete:   1 record  -> sync it, even below the byte threshold
```

A type with no records in the interval requires no empty file, but completion must establish that it is empty, not merely not yet flushed. Do not defer a sparse type's earlier records while declaring the common cut complete.

**Cost:** coordinated cuts can produce small files for low-volume types, increasing object requests, metadata, and future file handling. A slow or failing type can also hold back publication and retained-WAL release for the others. The correctness requirement is a complete usable prefix; common-cut sync is the selected direction, not proof that every correct implementation must use identical physical flush schedules. Do not relax prefix completeness to improve file sizes.

#### 9.7.2 Atomic visibility of the cross-type delta prefix

**Agreed invariant:** L0 delta compaction consumes a complete, durably published delta prefix, not a collection of independently visible files. Given dispatch completion C and an authoritative published prefix P, it may select its own next cut T with `C < T <= P`; its input must contain all applicable records of all three types in `(C, T]` for that VChannel. T need not equal an SN publication cut, but it must lie within coverage certified by P, not merely below the largest timestamp found in existing files. Readers must distinguish confirmed absence of a type from unfinished persistence. Section 9.4.5 defines the independently batched dispatch contract.

**Agreed logical barrier:** `delta_prefix_barrier[v] = P` is a timestamp in the committed mutation TimeTick domain, scoped to one VChannel. It certifies a continuous, complete published delta prefix through P across all three types. "Continuous" means no missing committed delta, not a record at every integer timestamp. It is neither a per-type maximum nor a collection-wide maximum over channels.

Advance this publication barrier atomically and monotonically from its previous published value B to P only after every delta in `(B, P]` has durable output and complete discoverable metadata. Consumers must acquire a coherent view of the barrier and its covered inputs; an atomic integer update by itself does not establish that relationship. Partition-scoped dispatch may select applicable records, but type, size, or file-count limits must not cut away an applicable earlier mutation or part of the selected complete interval. The barrier certifies eligibility for L0 dispatch, not completion of dispatch or application on target segments.

Example within one VChannel, initially published through 80:

```text
@100  UPDATE SET age=23 WHERE pk=2
@105  PK DELETE pk=9
@110  PREDICATE DELETE WHERE age<18
```

| Persistence/publication state | `delta_prefix_barrier[v]` | L0 scheduling of this new interval |
| --- | --- | --- |
| PK-delete and predicate-delete files are ready; the update file is not | 80 | Not eligible |
| All three files and their metadata are durable; publication has not completed | 80 | Not eligible |
| The complete interval is atomically published | 110 | Eligible as complete applicable input, not an arbitrary subset of its files |

For a pre-existing row with `pk=2, age=17`, the correct ordered result retains the row with age 23. Exposing the predicate delete without the earlier update could instead cause permanent deletion. L0 still only dispatches instructions; the order-dependent evaluation happens later in the consuming execution path.

**Separate publication from recovery progress.** The write-side WAL checkpoint protects recovery/truncation; the delta-prefix barrier constrains the scheduling-side input view. They are different contracts, even if a future implementation can reuse some plumbing. **Holding back the WAL checkpoint alone is insufficient if a consumer can already pick up the partial file set.** Neither that checkpoint nor the largest uploaded file timestamp automatically supplies the proposed scheduling barrier.

This is **logical publication atomicity**, not simultaneous uploads, an S3 multi-object atomic-write requirement, or a new user DML transaction. All covered objects must be durable and their metadata complete before advertising P as consumable. A failed type prevents advancing the common completion boundary; uploaded partial output must not be mistaken for a ready prefix. The rule is per VChannel, not a new cross-channel transaction guarantee.

**Why the PK-only path does not require the same batch-wide publication contract.** A PK delete has an already-fixed target PK and its original delete timestamp; it does not evaluate a predicate against field values produced by earlier mutations. In particular, `DELETE pk=2@100` and `DELETE pk=2@150` have the same final logical deletion result whichever is materialized first, once both are accounted for under the row-version timestamp rules. At a snapshot after both deletes, their deleted-version sets combine by union; the later tombstone covers the earlier one's scope for that same PK. This is not a claim that a later delete replaces pending deletes for other PKs.

Preserve the original timestamps and supported snapshot semantics: a reader between 100 and 150 must not see the future delete, so final-state equivalence is not permission to discard required history. This logical property also does not authorize reordering the online stream or bypassing apply-interface preconditions. Unfinished deletes must remain recoverable, and checkpoint advancement and GC must not lose them. Once updates can change predicate inputs, the analogous operations are no longer generally commutative, so independent file availability is insufficient.

**Current-code cross-check, not a new guarantee.** In the current local PK-only path, a [materialize batch](../../internal/streamingnode/server/wal/vchannel/l0materializer/materializer.go) can split into several L0 groups, each running a separate SyncTask. The pure-delete [writer](../../internal/flushcommon/syncmgr/pack_writer.go) emits one delta file per such task. [SaveBinlogPaths](../../internal/datacoord/services.go) registers each L0 separately and makes it a compaction candidate; the [outer materializer](../../internal/streamingnode/server/wal/vchannel/l0materializer/wal.go) waits for all groups before completing retained inputs. The [compaction filter](../../internal/compaction/entity_filter.go) uses PK identity and effective row timestamp versus delete timestamp, rather than file arrival order. These are reuse boundaries, not evidence that cross-type atomic visibility already exists.

**Industry precedent and limits.** [RocksDB atomic flush](https://github.com/facebook/rocksdb/wiki/Atomic-flush) separates multi-column-family file generation from all-or-nothing MANIFEST publication; its documentation also explains that WAL recovery can preserve consistency without enabling atomic flush. Adopt the distinction between physical I/O and logical publication. Adapt it to Milvus's complete delta-prefix boundary, where file-consuming compaction must not assume that missing inputs will be repaired by replaying the SN WAL. Do not copy RocksDB's MANIFEST implementation or claim its guarantee automatically applies to distributed object storage.

**Priority before implementation: explicitly design and review the atomic publication mechanics.** Sections 9.7.3–9.7.4 settle the external contract and recovery sequence. The following remain unresolved, rather than implicitly satisfied by three writers returning success:

- The durable representation and atomic publication mechanism behind the agreed per-VChannel timestamp barrier, including how it binds the exact durable file/metadata set and confirms empty types.
- Consistent consumer acquisition of that publication, including concurrent sync, L0 scheduling, and preservation of completeness when publishing target-segment deltas for load/data compaction.
- The implementation of the agreed crash/retry flow, ownership fencing across failover, integration with retained-WAL release, and safe reclamation of unpublished files without losing committed mutations.
- Bounds and measurements for small-file amplification, metadata/publication cost, and stalls caused by one slow delta type.

No concrete manifest, ready-marker, catalog transaction, or wire format is selected here. The invariant, common-cut direction, and publication/recovery contracts below are agreed; their implementation remains a **correctness-critical open design**, not an existing guarantee.

#### 9.7.3 Agreed interface: publish a complete VChannel delta range

**Use a new VChannel-level `PublishDeltaPrefix` interface, not an extension of `SaveBinlogPaths`.** The current [request](../../pkg/proto/data_coord.proto) registers file metadata for one `segmentID`; its repeated checkpoint/start-position fields do not make it a multi-segment delta-range publication. Current L0 output is represented by `SegmentInfo` with `Level=L0`, as created by [CreateL0Operator](../../internal/datacoord/meta.go). Keep that storage representation. The new publication unit may encompass several L0 outputs across partitions and delta types without redefining what an L0 segment is.

Responsibilities:

- **SN's VChannel materializer** chooses a consumed, complete cut T, writes every required delta output in `(C,T]`, and submits the complete output metadata, including confirmation of empty types.
- **MixCoord's DataCoord module** owns and durably persists the authoritative `delta_prefix_barrier[v]` and its committed output associations. It conditionally publishes a new range as one logical state transition.
- **L0 scheduling** acquires only the committed complete-prefix view. Staged file metadata is not eligible merely because it exists or its timestamps are below an advertised barrier; consumers use the published associations.

**Scope:** WAL is PChannel-level; delta-prefix state is VChannel-level. A PChannel can contain VChannels at different published cuts. Neither publication nor recovery substitutes the maximum cut across them. SN reassignment or restart must not reset a VChannel's logical coverage; publisher ownership and channel lifecycle identity must be checked separately, with detailed fencing mechanics still open.

Conceptual request, not a finalized protobuf:

```text
PublishDeltaPrefix(
    vchannel_identity,
    range_start_exclusive = C,
    range_end_inclusive = T,
    complete_delta_outputs,
    publisher_ownership
)
```

The RPC publishes **a range `(C,T]`**, not just an end timestamp or an independently visible file list. Let B be MixCoord's current barrier. For a valid request in the same VChannel lifecycle and from an authorized publisher:

| Range condition | Result and state transition |
| --- | --- |
| `C == B` and `T > B`, with all required output durable and complete | Atomically publish the output associations and advance B to T; return `RangeApplied` |
| `T <= B`, with the requested range entirely inside the established prefix | Return `RangeApplied`; do not register the submitted files again or change the prefix |
| `C < B < T` | Return `RangeMismatch`: partial overlap; publish nothing |
| `B < C < T` | Return `RangeMismatch`: gap; publish nothing |

Invalid bounds such as `C >= T`, invalid lifecycle identity, or stale ownership must not be accepted through the already-covered branch. These validations precede the range decision; concrete error codes remain implementation work. A mismatch response should include the current barrier so SN can reconcile instead of blindly retrying an incompatible range.

For example, with existing prefix `[0,100]`, `(100,120]` is the only illustrated new extension; `(80,120]` overlaps and `(110,120]` leaves a gap. Do not silently clip, union, or use `max(B,T)` to accept either. The check against B and publication must form one atomic conditional transition, not an unlocked check followed by a later update. Physical uploads may precede that transition, but partial metadata must remain unschedulable.

**`RangeApplied` acknowledges logical coverage, not a specific attempt's files.** Here “Applied” means the complete range has been durably published; it does not mean QN application, target-segment dispatch, or compaction has finished. Within the established VChannel prefix, `T <= B` proves coverage and lets SN finish this range's persistence responsibility. No historical request-response cache or per-attempt file-identity comparison is required solely to provide that answer. This does not weaken the initial publication's obligation to bind actual durable inputs, nor authorize GC based only on a range response.

**Distinguish a safe state transition from a usable retry result.** Rejecting every repeated request as a mismatch prevents duplicate state changes but leaves SN unable to distinguish completed work from a genuine range conflict. A lost ACK therefore remains an unknown outcome until retry/readback confirms coverage. Retry the same request after transport uncertainty; once covered, return `RangeApplied` even if earlier responses were lost. A genuine `RangeMismatch` requires reconciliation, not an assumption that the earlier attempt succeeded or permission to delete its files.

**Industry precedent and adaptation.** [Iceberg reliability](https://iceberg.apache.org/docs/latest/reliability/) separates immutable file preparation from atomic metadata publication and revalidates assumptions on retry. [AWS's idempotent-API guidance](https://aws.amazon.com/builders-library/making-retries-safe-with-idempotent-APIs/) distinguishes avoiding duplicate effects from giving callers an unambiguous completion result. Adopt those principles; adapt the result to Milvus's durable range coverage. Do not copy Iceberg's snapshot format or require AWS-style historical request-token receipts: this contract acknowledges a complete logical prefix rather than the identity of a resource-creation attempt. The two-result range protocol is this design's choice, not an existing Milvus API or an industry-standard wire contract.

#### 9.7.4 Agreed SN recovery and lost-response flow

**Recovery reads published state before resuming delta materialization.** The current [WAL recovery path](../../internal/streamingnode/server/wal/recovery/recovery_storage_impl.go) already restores persisted component metadata before bounded replay. The following adds a new MixCoord prefix dependency; it is not a claim that the current code already implements it.

1. Establish valid WAL ownership and identify the relevant VChannels. Read each VChannel's durable published prefix and the associated published metadata from MixCoord before scheduling replacement delta output. A failed lookup is not an empty prefix and must not default to zero.
2. Initialize each delta materializer's covered boundary from that authoritative state. Skip rematerialization/publication of already-covered delta ranges, including a commit whose ACK or local progress update was lost. Ownership changes do not reset the boundary.
3. Use normal WAL recovery to reconstruct outstanding work beyond that boundary and publish subsequent complete, contiguous ranges. This skip applies only to the covered delta-persistence responsibility: it does not skip WAL replay required by inserts, metadata, online state, or other consumers, and does not replace the PChannel recovery/truncation checkpoint.
4. During a live SN's publication attempt, a lost response leaves the outcome unknown. Retry the same range and output description until a definitive result or ownership/lifecycle change is observed. `RangeApplied` completes this range's persistence handoff; `RangeMismatch` requires reconciling with authoritative state. Retained-input and global checkpoint updates must still satisfy their other consumers' obligations.

```text
Publish file metadata A and prefix=100 atomically for (80,100]
  -> commit succeeds, ACK is lost
  -> SN crashes
  -> recovery reads prefix=100 and the committed metadata from MixCoord
  -> skip materializing/publishing (80,100]; continue beyond 100
```

Do not use “A was committed, then recovery blindly regenerates B for the same range” as the expected flow; it violates the recovery-first lookup above. If only object A was uploaded and the metadata/prefix commit never completed, MixCoord still reports 80: that range remains pending and must be completed. Conversely, a live SN that only lost the ACK can resend its unchanged request and receive `RangeApplied` without any crash or regenerated file.

File retention and orphan cleanup remain separate detailed-design work. A timeout or range mismatch is not proof that a referenced object is uncommitted garbage. The committed prefix and associations must survive MixCoord restart; SN recovery cannot infer completeness from object-store listing or local memory.

## 10. Update deltas: benefits and agreed semantics

Discussion checkpoint: **2026-10-21**. Logical in-place updates through deltas, joint mutation-order correctness, versioned field values, snapshot-aware base-index/delta query merging, and a bounded long-term assignment scope are agreed directions. The benefit analysis below is a set of hypotheses and evaluation questions, not measured results or an approved detailed update design. Prior large-delete tests motivate predicate deletion; they do not establish update-delta performance.

### 10.1 Why pursue this alongside predicate deletes?

The strongest candidate workload is a **small field patch to a wide existing row, particularly one containing a large unchanged vector**. Today's partial upsert reconstructs and reinserts the full row. A changed-fields delta could make the foreground payload reflect the patch instead of the entire row, while retaining base data for unchanged fields. Full-row replacement, vector updates, and workloads dominated by read-modify-write/conflict handling may offer different benefits.

| Potential benefit | Current cost it targets | Boundary and evidence needed |
| --- | --- | --- |
| Less full-row retrieval and Proxy materialization | Fetching old fields and rebuilding a complete insert payload | A delta path may still need target lookup, expression inputs, validation, or CAS; measure retrieved bytes, CPU, allocations, and RPC latency |
| Less foreground write/transfer amplification | Sending unchanged fields through insert messages and persisting replacement-row data | Measure WAL/network bytes and durable bytes including delta identity/version metadata; a smaller payload alone does not prove lower total cost |
| Less redundant work on unchanged vectors and indexes | Replacement rows and unnecessary full rewrites reintroduce unchanged content into the insert/index lifecycle | Delta Consolidation must preserve unchanged vectors/indexes while folding scalar updates; measure this separately from normally triggered full rewrites |
| Less row-version and delete/insert churn | One logical change produces a replacement row plus deletion of the old version | Delta chains and retained versions also consume memory/storage; compare steady-state footprint and maintenance work |
| Less foreground interference with ANN/search | Retrieval, insertion, deletion, and subsequent maintenance compete for resources | Read-time delta application or index correction can move cost onto queries; compare latency distributions, errors, throughput, and component CPU/memory |

These are workload-dependent opportunities. **Update delta is not an elimination of compaction**: it changes the foreground representation and may defer/concentrate work into delta application or consolidation. More delta history can increase query work, memory, and load/recover time. Kudu provides a lifecycle precedent, not a speedup estimate.

Section 9.1.7 now makes the maintenance distinction explicit: update accumulation should be consumed by layout-preserving Delta Consolidation rather than forcing an entire segment through Mix and ANN-index reconstruction. Full rewrites still perform their normal work when otherwise scheduled; their cost has not disappeared.

### 10.2 Shared lifecycle versus feature-specific work

| Area | Shared requirement | Update-specific question still open |
| --- | --- | --- |
| WAL and segment routing | Preserve committed order, applicability, and timestamp semantics; use Section 9.4's coarse predicate dispatch | Physical row targeting, changed-field encoding, concurrency checks, and missing-row behavior |
| Segment application and queries | Apply to the intended row version and expose consistent snapshots; use Sections 10.6–10.7's versioned-value and scalar query-merge semantics | Physical version storage and scalar overlay/index/cache implementation; vector updates are outside phase one |
| Durable deltas and load/recovery | Reconstruct the same state without omitted or reordered effects | Delta representation, update/delete merge, replay cost, history retention |
| Compaction and schema changes | Preserve mutation dependencies through rewriting and field removal | Delta Consolidation's column publication and residual-delete contract, full-rewrite replacement, and extension of the delete drain to updates |
| Scheduling | Update backlog determines consolidation timing; CG fragmentation determines rewrite/regrouping scope; deletes and existing segment-organization conditions schedule Mix | Consolidation thresholds, grouping heuristics, and remaining predicate-burden policy; updates do not trigger or accelerate Mix, but scheduled Mix consumes all applicable delta types |

This is the reason for a joint document, not a decision to force both mutations through one `DeletedRecord`, evaluator, file format, or scheduling metric. Existing delete decisions remain scoped to deletes unless explicitly extended.

### 10.3 Candidate benefit-validation scope

Before selecting a detailed update design, compare it with current partial upsert at equal logical work and required consistency. Vary changed-field fraction, row/vector width, indexed versus unindexed fields, repeated updates to hot rows, and cold versus warm reads. Include both foreground-only measurements and a sustained phase that exposes accumulated deltas and maintenance costs.

Measure client latency/throughput and concurrent ANN behavior; Proxy/query/streaming/compaction CPU and memory; retrieval, WAL, network, and object-storage bytes; index work; retained data/delta size; and load/recover latency after updates accumulate. Validate mixed update/predicate-delete correctness before treating any performance result as a benefit. No numerical target, benchmark configuration, or claimed gain is settled here.

The benefit target and initial update workload still require evaluation before selecting concrete update operators, overlay structures, index-correction implementation, or a durable format. Sections 10.4–10.8 settle the logical selection/assignment, ordering, field-version, scalar query-merge semantics, and bounded assignment scope; they do not establish an implementation or performance benefit.

### 10.4 Agreed initial update semantics: row selection x constant column patch

Discussion checkpoint: **2026-10-20**. An update has two independent dimensions:

1. **Row selection:** a supported `WHERE` condition may match zero, one, or many visible rows. PK equality is one predicate form; without an enforced uniqueness contract, it must not be assumed to identify only one row.
2. **Column assignment:** one `SET` patch may assign constants to one or more supported columns. Every matched row receives the same patch; columns absent from the patch retain their values.

```sql
UPDATE t
SET age = 20, status = 'active'
WHERE score < 60;
```

The logical request is a predicate (`score < 60`) plus a field patch (`{age: 20, status: "active"}`). It need not be expanded into one client request per row or column. This is a request-semantics decision, not a requirement for one WAL record, an on-disk encoding, or statement-wide atomicity.

**Initial assignment scope is constant values only.** Exclude `score = score + 10`, `a = b`, `version = version + 1`, and other assignments that depend on old field values. This restriction applies to `SET`, not to field access required to evaluate the supported `WHERE` predicate. Exact supported fields, types, and predicate combinations remain to be selected.

The **2026-10-21** scope clarification includes explicit NULL assignment to supported nullable fields. A field absent from the patch remains unchanged; an explicitly assigned NULL clears its value. Section 9.6 records the extensible `value_expr` representation, and Section 10.8 distinguishes the constant-only first phase from the small long-term core.

**Industry basis:** [PostgreSQL UPDATE](https://www.postgresql.org/docs/18/sql-update.html) separates selection of matching rows from assignment to explicitly named columns. Adopt that separation; adapt it to a restricted constant-assignment model. Do not infer PostgreSQL key uniqueness, locking, transaction guarantees, or general expression support for Milvus.

This agreement does not define UPSERT conflict/missing-key behavior or a batch API carrying different externally supplied patches for different keys. Those are separate semantics to discuss, not implicit consequences of multi-row selection.

### 10.5 Agreed concurrent-patch and predicate-order semantics

Discussion checkpoint: **2026-10-20**. The following are agreed semantic requirements, not an implemented concurrency protocol:

1. **Independent patches to different columns retain both changes.** A patch changes only its explicitly assigned columns, without copying unrelated old values into a replacement row. For an existing target row, with no intervening insert/delete or other mutation affecting applicability, the following two updates produce `age = 23, gender = 'male'` in either order:

   ```sql
   UPDATE t SET age = 23 WHERE pk = 2;
   UPDATE t SET gender = 'male' WHERE pk = 2;
   ```

2. **Assignments to the same row and column resolve in committed logical mutation order.** Between `SET age = 23` and `SET age = 24` on the same applicable row, the later committed mutation determines the final value of `age`; it does not overwrite unrelated columns. "Later" refers to the relevant VChannel's committed logical order, not Proxy arrival order or physical WAL append order. Intermediate reads must still respect their timestamp visibility.
3. **Predicate evaluation and patch application obey the same mutation order.** A `WHERE` predicate must see earlier applicable updates to its referenced fields. For example, after an applicable `SET age = 23`, a later `SET gender = 'male' WHERE age = 18` must not match that row based on its old base value. Online execution, load/recovery, and compaction must reconstruct the same outcome, as required by Section 6.

Disjoint assignment columns alone do not imply that two arbitrary updates commute: their predicates may depend on fields modified by each other. The first example is independent because both predicates use the unchanged PK and the target-row set is stable.

**Industry basis:** adopt the [SQL UPDATE rule that unassigned columns retain their values](https://www.postgresql.org/docs/18/sql-update.html). Adapt it to Milvus's field deltas and per-VChannel committed ordering; do not infer PostgreSQL row locks, isolation levels, or cross-channel statement atomicity. This agreement does not settle the apply/publication mechanism or removal of existing partial-upsert CAS.

### 10.6 Agreed versioned field-delta semantics

Record which field of a segment-local row changed, its assigned value, and the mutation's logical timestamp/version, rather than overwriting the only retained value. This is an execution-state model, not a prescribed WAL or durable-file format; offsets remain local to a pinned segment layout.

```text
Base: offset=3, pk=2, age=17, gender='female'
ts=100: age := 23
ts=110: gender := 'male'
ts=120: age := 25
```

| Supported read snapshot T | Visible age | Visible gender |
| --- | --- | --- |
| 95 | 17 | female |
| 105 | 23 | female |
| 115 | 23 | male |
| 125 | 25 | male |

For each row and field, resolve the latest applicable version visible at `T`, falling back to the compatible base value when no such update exists. An update to another field does not replace this field's value. The mutation timestamp supplies the visibility/version boundary under Section 6's committed-order contract; it is not a new client idempotency key.

**QN multiversion residency is a key resource risk, especially for large scalar values.** Budget retained value payloads and per-row/version metadata across loaded segments and query replicas, not just the latest value or mutation count. Repeated replacements can retain multiple large values while earlier versions remain required; the immutable base need not be copied for every update. Sharing immutable assignment payloads may reduce duplication but does not eliminate version bookkeeping, and no sharing, eviction, or spill implementation is assumed. Evaluate retained bytes, version depth, and query/apply interference explicitly. This serving-side cost is distinct from full-rewrite output/index construction; Section 9.1.4 separately accounts for the compactor's resident delta program, while Section 9.1.5 bounds the historical-retention requirement.

**Retention is about mutation-time values, not whether PKs repeat.** Updates establish values for successive visibility intervals, and subsequent predicates may depend on those intermediate values. Within the retained serving state, do not reduce the history to the latest assignment as if it were a monotone PK tombstone. Phase one retains the required delta state in memory. Existing [memory-based write protection](../../internal/rootcoord/quota_center.go) is a reuse point, not proof that the new UPDATE path is already covered. In the user-described pooled Cloud compaction architecture, do not assume a direct RocksDB-style local compaction-backlog-to-frontend feedback loop. Use update-byte statistics to schedule Delta Consolidation independently of that protection; thresholds and safe capacity limits remain open. Persistence and query application are separate WAL consumers: syncing an L0 object does not itself release QN versions, and Delta Consolidation completion alone does not authorize reclaiming state still required by readers.

A later delete at `ts=115` evaluating `age < 18` must see `age=23`, not the base value 17 or the future value 25. The same field-resolution semantics must be used by queries and subsequent mutations, and reconstructed by load/recovery and compaction.

`T` denotes a supported execution snapshot, not the `GuaranteeTs` freshness lower bound. These examples describe the old segment's retained serving state, not a promise that a replacement base can reconstruct the same history. Section 9.1.5 settles the boundary: preserve multiversion visibility while the old generation serves readers, allow full rewrite to fold consumed history into the new baseline, and introduce neither UNDO nor long-term time travel. Concrete version storage, safe handoff/reclamation, equal-timestamp ordering, and publication remain detailed-design work.

### 10.7 Agreed scalar filtering: shared bitmap correction for base indexes and raw chunks

**Decision:** reuse the immutable base scalar index or raw chunks without rewriting them for each foreground patch. Both access paths produce a base match bitmap and use the same snapshot-visible update correction. Section 2.9 provides mature precedents for suppressing stale base results and supplementing changes, with explicit implementation differences.

For a single-field predicate `P(age)`, fix a compatible base/index generation, segment layout, supported snapshot `T`, and evaluated row-offset range:

```text
I    = offsets matching P through base scalar-index lookup OR raw-chunk evaluation
U(T) = offsets with a visible age update not incorporated into this base/index
D(T) = offsets in U(T) whose latest visible age value satisfies P

Matches(T) = (I - U(T)) union D(T)
           = (I AND NOT U(T)) OR D(T)   // positive-match bitmap notation
```

Apply the row's insert/delete visibility rules as well. The equation expresses logical positive-match sets; it does not change the existing executor bitmap polarity. If an index produces only candidates, its required exact recheck still applies. Preserve expression validity/NULL semantics alongside match bits: an update to or from NULL must replace the old validity state as well, especially before composing NOT or other boolean operators.

**QN raw-chunk execution, not the compactor's read/modify/write batch model:** the raw path evaluates the predicate against the segment's base chunks to obtain `I`; the index path obtains `I` through index lookup. Both then mask visible updated offsets and supplement matches from their latest visible values. This does not require copying a whole raw chunk, constructing a fully patched column before filtering, or changing shared base memory. Internal execution batching and lazy access remain implementation details of the existing access paths. Restrict `I`, `U(T)`, and `D(T)` to the same evaluated offset range and translate chunk-local positions to the segment's row offsets consistently. The memory structure must support finding updates in that range without requiring every chunk evaluation to rescan all segment updates.

For example, base `age` values at offsets `[0, 1, 2]` are `[17, 23, 23]`; updates set offset 0 to 23 at `ts=100` and offset 1 to 25 at `ts=105`. For `age = 23` at `T=110`, either base access path produces `I={1,2}`, while `U(T)={0,1}` and `D(T)={0}`; the corrected result is `{0,2}`. At `T=95`, neither update is visible, so the original result `{1,2}` remains valid. Raw data is analogous to a flat scan for this filtering contract, not a claim that its physical access cost is identical to an index.

**Snapshot masking and supplementation rules:**

- Subtract by **row offset for this field**, not by equal values, PK alone, or the delta's own match set. Every offset in `I` whose indexed field has a visible unmerged update is removed before supplementation, including updates whose new values fail the predicate.
- `U(T)` is snapshot-specific: an update after `T` must not hide the old match. A patch to `gender` alone does not mask the base `age` index entry.
- Resolve each updated row to **one latest visible value** before evaluating the predicate. Do not union matches from every historical delta version. An assignment that restores the base value still follows the same rule: mask the old entry, then re-add it if the visible value matches.
- Supplement independently of base hits from either access path. An empty base result cannot suppress the delta branch, and base-only min/max or other pruning must not discard updated rows that could match.
- For compound predicates, combine corrected field selections with the proper boolean/NULL semantics, or evaluate against a coherent versioned row view. A stale base result from one operand must not prune candidates needed by another corrected operand.

For the `age` history in Section 10.6, the same offset demonstrates both removal and addition:

| Snapshot and predicate | Base hit at offset 3 | After masking | Delta supplementation | Final hit |
| --- | --- | --- | --- | --- |
| `T=95`, `age=17` | Yes | Retained; no visible update | None | Yes |
| `T=105`, `age=17` | Yes | Removed | 23 does not match | No |
| `T=105`, `age=23` | No | Empty | Latest visible value 23 matches | Yes |
| `T=125`, `age=23` | No | Empty | Latest visible value is 25; historical 23 is ignored | No |

**Initial execution direction:** a separate delta index is not required. Scan the updated rows for the referenced field and offset range, resolve their latest visible values, and evaluate the predicate; retain the existing index or raw-chunk access path for base evaluation. This is not a requirement to scan every historical delta record on each query. The version lookup/iteration structure is deferred. As update density or retained history grows, evaluate delta indexing or consolidation; scanning deltas is not assumed to remain cheap indefinitely.

**Coherent publication requirement:** a read must use a mutually consistent base/index generation, updated-offset set, resolved values, and complete apply progress. Otherwise it can observe a new mask without its replacement value, or an advanced progress marker while matches are still stale. Pin the relevant state across evaluation; exact synchronization and publication granularity remain open. Rebase/consolidation may retire overlay entries only when the replacement base/index and retained-reader contract safely cover them.

**Scope and current-code boundary:** this settles scalar-predicate result semantics, not a C++ data structure, delta-index format, vector-index update algorithm, or cache invalidation protocol. The current [UnaryExpr.cpp](../../internal/core/src/exec/expression/UnaryExpr.cpp) routes evaluation through scalar-index/raw-data paths and performs chunk pruning; [Expr.h](../../internal/core/src/exec/expression/Expr.h) contains chunk/scan and index evaluation paths and access-path-dependent prefetch. These are integration points to make update-aware, not evidence that this merge already exists. Projection, expression caches, and later mutation predicates must agree with the same field versions; correcting only returned output values cannot prevent missed filter matches. The shared bitmap rule covers filtering, not the separate version-resolved field-output interface.

#### 10.7.1 Expression integration and update-record responsibilities

Discussion checkpoint: **2026-10-22**. **Using the old base is valid; using only the old base is not.** Keep base access as the main evaluation path and incorporate visible updates before its result is used as an authoritative selection. Do not interpret the Kudu precedent below as a requirement to abandon base indexes or raw chunks.

**Mutation WHERE uses the same version-aware evaluation.** An UPDATE or predicate DELETE must evaluate its WHERE against the state produced by earlier applicable mutations in committed logical order, not against base data alone or its own not-yet-applied assignments. For example, base `age=17`, followed by `UPDATE@100 SET age=23`, must not match `DELETE@110 WHERE age<18`. After WHERE produces the selected offsets, the update application side records field assignments and their timestamp; the update-version store need not retain or interpret the original WHERE AST. Query-time evaluation still needs the store's version values, not just a final bitmap.

**Integrate by responsibility, not by leaf/non-leaf classification.** Field readers, index-backed predicates, comparisons, and any other admitted value-dependent operators must use the correct versions. Pure logical AND/OR/NOT operators combine already-corrected child results; they need not independently read UpdatesRecord. For `age > 15 AND status != 'active'`, correct the age branch and status branch at the same snapshot, then combine them. Correcting only the final base-derived result is insufficient: it cannot discover rows made newly eligible by updates without evaluating their changed values.

Candidate pruning and short-circuiting must also consume corrected results. A base-only empty bitmap, min/max exclusion, or stale child result must not suppress the update branch or terminate a compound expression before its new matches can be considered. The current [ConjunctExpr.cpp](../../internal/core/src/exec/expression/ConjunctExpr.cpp) builds active-row masks and can skip remaining expressions; [LogicalBinaryExpr.cpp](../../internal/core/src/exec/expression/LogicalBinaryExpr.cpp) combines child outputs. These are existing responsibilities to integrate with, not already update-aware implementations.

**Input windows are not result ranges.** A node may evaluate offsets `[4096,8192)` and return sparse matches such as `{4100,5000,8000}`. The input window bounds which updates to inspect for the referenced field; it does not assert that matches are contiguous. A node receiving an already-valid discrete selection instead needs batched offset access. Never restrict supplementation to the old base's hits, which would omit newly matching updated rows. The current [scan window in Expr.h](../../internal/core/src/exec/expression/Expr.h) and [discrete value gathering in ColumnExpr.cpp](../../internal/core/src/exec/expression/ColumnExpr.cpp) illustrate the two access shapes.

**Industry reference and boundary.** Kudu's [DeltaApplier, release 1.18.1](https://github.com/apache/kudu/blob/1.18.1/src/kudu/tablet/delta_applier.cc) seeks deltas by row ordinal and applies updates by column; when deltas may exist, it disables decoder-level evaluation over uncorrected values. Adopt the prevention of stale-value filtering and the separation of value/version access from expression execution. Adapt this to Milvus's bitmap-correction path; do not copy Kudu's materialize-then-apply execution model or claim that it implements this exact bitmap equation.

**NULL requires a separate deep dive.** The need to distinguish an explicit NULL update from no visible update is established, but validity storage, operator behavior, index/raw parity, and compound three-valued evaluation are not settled by this section. Prior NULL references are correctness requirements, not a completed implementation design.

### 10.8 Agreed assignment capability boundary: simple deterministic row-local values

Discussion checkpoint: **2026-10-21**. The long-term target is **simple, deterministic, side-effect-free row-local assignment**, not full SQL or arbitrary user computation. Section 2.10 records the mature precedents and their limits. This is the agreed scope, not a claim that non-constant updates already exist.

| Capability | Example | Scope decision |
| --- | --- | --- |
| Constant assignment, including explicit NULL for nullable fields | `status = 'active'`, `tag = NULL` | Phase-one core |
| Same-row field reference/copy | `backup_score = score` | Long-term core extension; not phase one |
| Basic numeric arithmetic | `count = count + 1`, `score = score * 0.9` | Long-term core extension; precise operator/type combinations require validation |
| A few deterministic built-ins | `coalesce(score, 0)`, `lower(name)` | Add individually only for demonstrated needs; examples are not a committed function library |

**Non-goals for feature completion:** CASE and elaborate conditional programs, complex JSON/array transformations, arbitrary UDFs, subqueries, aggregates, joins/cross-table computation, and external/model execution. Preserve room for deliberate future extensions without requiring these capabilities to finish delta-based updates. Do not persist calls whose result depends on random state, the evaluator's current clock, or an external service as replayable assignment expressions. Complex backfill computation can remain outside this evaluator and supply values through a separately defined write workflow.

**Phase-one cell granularity (2026-10-22):** support whole-cell constant replacement for eligible scalar fields, including JSON, strings, and scalar arrays; exclude JSON-path updates, array-element updates, and append operations. Storage-specific partial-cell optimizations, such as [MySQL's conditional binary-JSON partial updates](https://dev.mysql.com/doc/refman/8.4/en/json.html#json-partial-updates), are deliberately deferred and are not assumed benefits of this delta design.

**Semantic and admission constraints:**

- Read referenced fields from the applicable row state immediately before the mutation in committed logical order, including earlier visible field deltas. Use the compatible base when no visible update replaces it; never substitute stale base values for visible updates or read values from later updates. The same pre-update-row model should govern all RHS expressions within one SET; exact evaluation/publication mechanics remain deferred.
- Keep field-update eligibility separate from expression expressiveness. Phase one excludes vector-column updates, which is the premise for unchanged-vector/ANN reuse during Delta Consolidation. PKs, partition keys, and clustering keys must not be updated in a way that changes data identity or distribution. Ordinary scalar patches remain the target; field/type admission details and function-field restrictions must be enforced independently of RHS expressiveness.
- Require consistent results and failure semantics across online execution, load/recovery, and Go compaction, including NULL, conversions, overflow, and division by zero. Restrict expression size/depth and function cost through explicit limits; numerical limits and error/publication policy are not settled here.
- Do not widen the allowlist because the wire format can represent a node. Each admitted combination needs a stable, version-compatible meaning and cross-evaluator verification. Replaying `age = age + 1` twice over an already-updated state would change the result: application/progress accounting must prevent that duplicate effect. This is distinct from client retries creating separate committed requests, and no new exactly-once protocol is claimed.

**Rationale:** richer expressions increase the semantic surface that every execution and recovery path must preserve. The primary benefit target remains avoiding full-row reconstruction/reinsertion and redundant work on unchanged vectors and indexes; it does not require a general SQL assignment engine. Phase one remains constant-only even though the message shape can express later extensions.

### 10.9 UpdatesRecord: query-driven access and append-growth constraints

Discussion checkpoint: **2026-10-22**. Access requirements and growth constraints are agreed; a concrete C++ container layout is not. `UpdatesRecord` names the proposed segment-local field-version state, not an existing implementation or a requirement to merge it with `DeletedRecord`. Preserve the compatible serving generation, row layout, and complete apply-progress contract from Sections 10.6–10.7.

**Derive storage from the access pattern:**

1. Locate the referenced field without scanning unrelated fields.
2. Enumerate updated offsets in an evaluation window, or gather versions for a discrete offset selection, without rescanning all segment updates for every chunk.
3. For each selected cell, resolve its latest applicable version visible at execution snapshot `T`; if none exists, retain the compatible base value and do not mask its match.
4. Expose resolved values to expression evaluation or field output. The store is not responsible for interpreting the complete WHERE, and a final result bitmap alone is not enough to resolve changed matches.

The following is a candidate logical organization, **not three mandatory nested hash tables**:

```text
UpdatesRecord (one segment / compatible base generation)
  field: age
    offset 3 -> [(ts=100, value=23), (ts=120, value=25)]
    offset 8 -> [(ts=110, value=30)]
  field: status
    offset 3 -> [(ts=105, value="active")]
```

At `T=105`, age offset 8 has no visible update and must not enter the snapshot's update mask. An ever-updated-offset bitmap is only a candidate directory, not the mask for every snapshot. Use row offsets rather than PKs for this already-resolved state. A global TimeTick-ordered mutation list is appropriate for execution order, but not necessarily a suitable sole index for per-field query access.

**Field lookup.** A hash map is a reasonable candidate, not a requirement. A field-ID-to-slot mapping followed by a compact array is another representation; array slot access is direct addressing, not modulo hashing. The offset directory must support the access patterns above; ordered structures, block directories, and bitmap-assisted lookup remain candidates rather than settled choices.

**Timestamp lookup.** Within a cell, maintain an ordered version sequence. A snapshot needs a predecessor lookup, not all versions in a time interval: find the first version with `ts > T` and select its predecessor, if present. For `[(100,23),(120,25),(150,30)]` at `T=130`, use 25. [C++ upper_bound](https://eel.is/c++draft/upper.bound) provides the standard search contract; ordered random-access storage supports logarithmic lookup. Append already-ordered applied versions rather than sorting on every query. Equal-timestamp ordering, complete publication, and the supported-snapshot boundary still follow the shared mutation contract and require implementation work.

**Frequent append is a primary design constraint.** Logical ordering and binary search do not require one physically contiguous, repeatedly reallocated vector. Evaluate segmented/chunked version sequences that append into a tail block and allocate a new block when necessary, without relocating the full historical sequence. A block directory can support ordinal access or timestamp-based block lookup, followed by an in-block search. This is a candidate growth strategy, not a selected container, block size, or concurrency protocol.

**Mature and local references:**

- Kudu's [DeltaKey](https://github.com/apache/kudu/blob/1.18.1/src/kudu/tablet/delta_key.h) organizes REDO keys by row ID and then ascending timestamp. Adopt row-local version ordering; the outer field partition here is a Milvus-oriented adaptation, not Kudu's exact data structure. Do not import its UNDO retention model.
- [oneTBB concurrent_vector](https://uxlfoundation.github.io/oneTBB/main/tbb_userguide/concurrent_vector_ug.html) preserves existing element locations while growing, at additional overhead relative to vector. Its documented construction/access synchronization requirement shows that stable locations alone are not a publication protocol. Borrow the non-relocating-growth principle, not a decision to instantiate one concurrent_vector per cell.
- Local [ConcurrentVector.h](../../internal/core/src/segcore/ConcurrentVector.h) provides chunk-based ordinal access, and [Utils.cpp](../../internal/core/src/segcore/Utils.cpp) implements `upper_bound` over `ConcurrentVector<Timestamp>`. These demonstrate that noncontiguous storage can support binary search; they do not establish suitability or safe reuse for many sparse per-cell histories.

**Costs and boundaries to evaluate before selecting a structure:** avoid a large allocated block/container per cell with only one or two versions; account for directories, allocations, fragmentation, and payload ownership as well as version bytes. Large immutable values may be referenced rather than repeatedly copied, but sharing and lifecycle details remain open. Chunking can avoid moving old value records while the directory itself still grows; do not promise zero copying of all metadata. Readers must see only fully published versions, with safe concurrent directory growth and retained-state ownership. Neither ordinary vector nor a concurrent-growth container solves these requirements by itself. Phase one remains memory-resident, without assuming spill or premature history reclamation.

This section settles the query-driven contract and the need to avoid repeated whole-history movement, while leaving the concrete field/offset containers, short-history representation, segmented-growth design, NULL handling, and concurrent publication for focused follow-up design.

## 11. Deferred integration and open contracts

This document does not fully specify WAL payloads, deterministic replay, L0 persistence and catch-up during segment load, release/reload behavior, CDC, compaction execution, or replica-wide visibility. Section 9.1.3 settles the drop-field safety requirement, both mandatory drain phases, and the all-VChannel release condition, while preferring unified drainage and reuse of existing checkpoint/L0 progress. Section 9.3 settles target-QN predicate-delta loading's access-path policy and records the asynchronous-recovery direction: separate base load from background apply, with per-request `GuaranteeTs` admission against complete relevant progress. Its PG precedent and applicability limits are explicit; the progress protocol remains open. Section 9.4 settles coarse predicate dispatch and rejects ingress association tables with lineage repair; its durable protocol and cost remain open. These remain required integration work, not benefits already delivered by local bitmap application. Support for accepting general predicate deletes against unloaded data is deferred; this is distinct from allowing already-accepted mutations to progress through L0 and offline compaction after release.

Before coding the predicate-delete part, finalize:

- **Critical: implement the cross-type atomic publication contract in Sections 9.7.2–9.7.4.** The dedicated VChannel interface, contiguous range rule, coverage-based `RangeApplied`/`RangeMismatch` results, and recovery-first prefix lookup are agreed. Finalize durable metadata/transaction mechanics, owner fencing, coherent consumer discovery, downstream handoff, and safe retention/cleanup. Checkpoint gating without consumer visibility gating is insufficient. This work must not be reduced to adding independent writers or broadening `SaveBinlogPaths`.
- The concrete message types and protobuf/semantic versioning behind Sections 9.5–9.6's shared logical contract, schema validation, capability admission, CDC handling, and public affected-row-count semantics. Preserve a common scope/predicate contract and expression-shaped assignments without treating the illustrative message names or AST notation as finalized wire definitions.
- The buffer/file encoding and durable handoff behind Section 9.7: retain ordered mutation commands and original effective TimeTicks in separate update/predicate-delete file types, and establish complete-prefix merge/recovery without introducing user-controlled transaction batches.
- The delete plan/context and write-capable segment handle, including the CGo boundary.
- Bitmap ownership, whole-segment versus batch-local input, and the applied-row-count definition.
- Mutation serialization, snapshot publication granularity, and cancellation/partial-application behavior.
- The QN integration and memory layout behind Sections 10.7.1 and 10.9: version-aware mutation WHERE, correction before logical pruning, window/discrete-offset access, predecessor lookup, and append growth without repeated whole-history relocation. Select concrete containers only after evaluating sparse-cell overhead and concurrent publication. Deep-dive explicit NULL, validity handling, and compound-expression semantics separately; do not treat the agreed bitmap equation as a complete NULL design.
- The restricted Go compaction evaluator's exact supported expression/type set, parity tests, and schema/version contract; preserve the agreed legacy fallback for expressions outside that set.
- The shared batch executor behind Section 9.1.4: concrete integration across all four rewrites, Arrow ownership and batch working buffers, reusable ordered-program iteration, memory limits, and durable completion bookkeeping. Phase one loads the complete applicable delta program in memory independently of base-batch size, without disk cache or spill. Preserve safe time-eligibility bypass and existing TTL policy; the agreed execution model does not establish these implementation details or their performance.
- Section 9.1.7's Delta Consolidation implementation: replace selected column data and affected scalar indexes while retaining segment identity/layout and unchanged ANN indexes; define coherent publication, remaining-delta ownership, old-reader protection, and exclusion with L0 dispatch/full rewrites. Resolve interleaved-delete semantics before consuming update history. Do not claim physical delete reclamation or full cross-type completion merely because the selected update columns were folded. Update-backlog timing thresholds, CG-fragmentation-driven rewrite/regrouping heuristics, and safe all-resident capacity remain open; updates must not trigger or accelerate Mix, even though normally scheduled Mix consumes all applicable deltas.
- Serving-generation pinning, handoff, and safe reclamation under Section 9.1.5's settled retention boundary. Do not turn this implementation work into an UNDO or post-rewrite time-travel requirement; protect admitted readers and mutations not incorporated by the rewrite.
- The per-VChannel completion proof for both Streaming -> L0 and L0 -> target segment delta-log publication, including coherent checkpoint/L0 observations, old growing data, replacement segments, retry/restart, and gating every rewrite that can remove the field; preserve the two-phase requirement and unified-drain preference in Section 9.1.3.
- The dedicated stats RPC's detailed contract, coherent count/apply-progress sampling, source/freshness validation, and cache invalidation for Section 9.2's best-effort cumulative observations; define predicate-backlog accounting and policy use without requiring stats availability, introducing a durable stats cache, or overriding Section 9.1.7's normal-Mix scheduling boundary.
- The predicate-delta loader/CGo wiring, replay bookkeeping, failure handling, and resource budgeting; preserve Section 9.3's access-path policy and per-request progress gate. Define the initial readable state, complete apply-progress publication/aggregation, and recovery-mode/readiness controls without silently weakening requested consistency. Delegator L0/streaming catch-up must participate in the completeness proof, but its detailed protocol remains separate work.
- The durable coarse-dispatch protocol and fanout measurements in Section 9.4. Section 9.4.5 settles independent publication/dispatch batching and complete-input, ordered-visibility, and target-coverage requirements, not their implementation. Define completion-state persistence, task decomposition, publication/recovery, and safe target changes while preserving lightweight L0 and release-independent progress, without reintroducing a request-time association table or lineage-repair subsystem.
- The separately tracked legacy L0 catalog-publication risk in Section 12.1: establish the required downstream visibility and recovery guarantees when one task updates more metadata KVs than a single batch can commit. A single task result or manifest-backed storage is not proof of cross-batch atomicity.

For update deltas, Section 10.4 settles multi-row predicate selection and multi-column constant assignment, while excluding old-value-dependent assignments from phase one. Section 10.5 settles the observable results of independent and overlapping patches and their predicate-order dependency. Sections 10.6–10.7 settle versioned field values and scalar query merging: mask snapshot-visible updated offsets in the base result, then supplement matches from each row's latest visible field value, initially without requiring a separate delta index. Section 10.8 bounds the long-term assignment core to literals, same-row references, and basic arithmetic, with a few deterministic functions only when justified; it does not require full SQL. Sections 9.5–9.6 record shared message responsibilities, the Proxy/StreamingNode validation responsibility principle, single-mutation request scope, extensible typed RHS representation, separate selection/value/application responsibilities, and pre-append support routing, not a completed wire or detailed validation protocol. Section 9.7 settles the buffer responsibility and separate-file ordering direction, not durable format/publication details. The benefit target, exact supported fields/types/predicates, and UPSERT semantics still require discussion. Detailed physical row targeting, value-version storage/GC, overlay/index/cache implementation, coherent publication, concurrency enforcement, durable replay, load, and compaction mechanisms remain open. The shared ordering invariant in Section 6 applies to all of them; it is a requirement, not an implemented guarantee.

No implementation or performance validation accompanies this draft. This revision records agreed query semantics, logical message responsibilities, single-mutation request scope, TimeTick-preserving buffer/file ordering, VChannel range-publication and SN recovery contracts, shared ordered batch delta execution and safe bypass, all-resident phase-one delta execution, separate L0 dispatch, Delta Consolidation, and full-rewrite responsibilities, bounded assignment capability, and industry applicability boundaries; it does not authorize implementation or mark the complete update design as settled.

## 12. Tracked legacy implementation risks

### 12.1 L0 task-result publication across catalog KV batches

Discussion checkpoint: **2026-10-22**. This is an existing implementation boundary relevant to predicate/update deltas, recorded separately from the main data-flow design. Code inspection establishes a reachable multi-batch write path; no production incident or crash-induced data loss has been reproduced in this investigation.

**Publication and encoding.** After DataNode returns an L0 compaction result, MixCoord's DataCoord module updates the affected target L1/L2 segments and retires the input L0 segments. The result is not serialized as one catalog KV. In the manifest-backed V3 path, each changed target contributes one SegmentInfo KV containing its manifest reference, statistics, and other segment metadata; each retired input L0 contributes its updated SegmentInfo KV. Multiple changes to one segment still produce one SegmentInfo KV. Delta-file references live in the manifest, not one catalog KV per file. Legacy non-manifest segments can additionally require per-segment/field FieldBinlog-list KVs, not necessarily one KV per file.

**Batch unit and trigger.** `SaveByBatch` partitions the encoded KV map, not the worker result by bytes or the delta records by type. The current default `metastore.maxEtcdTxnNum` is 64 KV writes per batch. With all affected segments manifest-backed and no additional compatibility KVs:

```text
KV count = changed target segments + retired input L0 segments
200 changed targets + 1 retired input L0 = 201 KVs
SaveByBatch: 64 + 64 + 64 + 9 -> four MultiSave transactions
```

The 200-target example is illustrative, not an observed test result. Input L0 count/size selection thresholds do not cap target fanout. `dataNode.compaction.levelZeroMaxBatchSize` controls the worker's execution batches (default `-1`, no fixed configured count cap; memory-based batching still applies); their outputs are combined into one task result. The current L0 target-selection path has no independent target-count cap. PK filtering can reduce actual changed targets but does not guarantee a bound; coarse predicate/update dispatch may associate even one instruction with every scope/time-eligible target, including zero-match segments. The deciding quantity is the encoded KV count, not deleted rows, predicate count, file count, or an assumed threefold multiplier for three delta types.

**Failure boundary.** Each `MultiSave` is a transaction, but the loop across batches is not one transaction. Crossing the limit is not itself an error. Failure or crash after an earlier batch commits can leave partially persisted target/input metadata. Normal in-memory SegmentInfo publication waits for the complete catalog call to succeed; that does not prove all-or-nothing state after restart. Neither a successful set of object writes nor one task-result callback establishes the complete downstream prefix required by Section 9.4.5. Task-state persistence is a separate subsequent write, not an atomic wrapper around the KV batches.

**Disposition.** Keep L0 lightweight and preserve the agreed task-level logical publication boundary; park the concrete remediation as implementation work. Before relying on it, fault-inject beyond the configured KV limit, between batch commits, and before task-state persistence. Verify recovered target discovery, input-L0 retention/retirement, retry behavior, and completed-dispatch progress `C`; partial publication must not be mistaken for complete coverage or expose an incomplete usable mutation prefix. No remediation or new commit protocol is selected here.

**Source evidence:** [L0 result adoption and input retirement](../../internal/datacoord/compaction_task_l0.go), [batch manifest publication](../../internal/datacoord/segment_manifest_commit.go), [SegmentInfo publication](../../internal/datacoord/meta.go), [catalog encoding and SaveByBatch](../../internal/metastore/kv/datacoord/kv_catalog.go), [KV encoding helpers](../../internal/metastore/kv/datacoord/util.go), [batch splitting](../../pkg/util/etcd/etcd_util.go), and [etcd MultiSave](../../internal/kv/etcd/etcd_kv.go). These are code-path findings, not end-to-end fault-injection results.
