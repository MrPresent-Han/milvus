# Predicate Deletes and Delta-Based Updates

Date: 2026-10-16

Discussion checkpoint: 2026-10-20

Status: Draft — predicate-delete data flow, progress-bound best-effort statistics, predicate-backlog scheduling, asynchronous recovery direction, constant-assignment update semantics, and snapshot-aware base-index/delta query merging agreed; detailed protocols remain open; not implemented

Scope: Joint design for predicate deletes and logical in-place updates through deltas, including segment application, ordering, persistence, load/recovery, compaction, and schema safety. Predicate-delete decisions are more developed; update benefits and detailed mechanisms still require evaluation.

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
- **Preserve field-level patch semantics under concurrency.** Independent patches to different columns retain both changes; assignments to the same row and column resolve in committed logical mutation order. Predicate evaluation and patch application must observe earlier applicable mutations consistently across online execution, load/recovery, and compaction; see Section 10.5.
- **Keep versioned field values and correct stale-index results.** Preserve the base scalar index, remove old matches for rows with snapshot-visible updates to the indexed field, and supplement matches evaluated against their latest visible values. Initially, scan the updated-field values without requiring a separate delta index; see Sections 10.6–10.7 and the industry comparison in Section 2.9.
- **Evaluate benefits before detailed update design.** Section 10 records opportunities and tradeoffs, not measured improvements or a settled implementation. Exact field/type and predicate support, physical row targeting, index-overlay implementation, concurrency control, and update-delta representation remain open.

Sections 4–5 and 9 retain the established predicate-delete directions. Their choices are not automatically extended to updates: for example, a cumulative deleted-row count is not an update-compaction cost model, and the settled delete drain is not yet a complete update/schema protocol.

### 1.3 Shared predicate-delta dispatch direction

- **Use coarse scope, not an ingress-time matched-segment table.** Persist predicate mutations with their VChannel/partition scope and logical timestamp; choose candidate segments from current storage metadata when L0 dispatch runs, not from the QueryView seen by the original request.
- **Keep L0 lightweight.** L0 dispatch does not execute predicates, read business columns to discover matches, or depend on the collection remaining loaded. Actual evaluation belongs to online/load execution and background data compaction.
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
             -> durable predicate L0 + metadata
             -> lightweight scope-based dispatch using current storage metadata
             -> segment-owned predicate delta files + published metadata
                  |-> DataNode data compaction: evaluate and rewrite
                  `-> QN load/recover: base loading + background ordered apply
                        -> per-query admission by complete apply progress
```

1. **Admission and WAL append.** Proxy validates the typed expression, schema, and operation scope, then submits a supported predicate directly to WAL without retrieving PKs or collecting segment statistics. Unsupported expressions retain the legacy PK path. The initial general-predicate admission prerequisite remains loaded data; this is not a requirement that the collection stay loaded afterward. Commitment and ordering are per relevant VChannel, not a new collection-wide atomicity guarantee.
2. **Online consumption and application.** The query-side consumer retains predicates in the Delegator's delete buffer for catch-up and forwards applicable records to serving segments in the relevant QueryView, including local Growing data on SN and Sealed data on target QNs. Segment-local execution evaluates the predicate and applies selected offsets. Online forwarding is scoped, not a fanout to unrelated collections or partitions.
3. **Durable L0 production.** The persistence consumer materializes committed predicate deletes into L0 output and publishes its metadata. This and the online delete buffer are separate responsibilities: the current `deleteBuffer.Put` path must not be described as the owner that flushes L0 files. Extending the existing WAL L0 materializer to predicate records remains implementation work.
4. **L0-to-segment dispatch.** Predicate L0 dispatch selects candidate Sealed segments by current storage metadata, channel/partition scope, and safe logical data-time bounds. “Candidate” does not mean the expression is known to match. Persist a segment-owned predicate delta file type that can batch multiple records, with discoverable segment metadata; do not create a request-time segment association table or evaluate expressions in L0. Publication must complete before the corresponding L0 input can be retired. This storage operation is not a second online broadcast to all already-loaded segments.
5. **Scheduling and offline compaction.** MixCoord does not need to know a predicate's match count to schedule work. Accurate applied-delete counts, when available, are progress-bound best-effort hints; outstanding predicate counts provide a hit-independent scheduling signal. DataNode reads durable inputs and evaluates predicates without a loaded-QN dependency. Executing 100 predicates and deleting zero rows is an accepted outcome: incorporating those records into durable output can still remove future replay work.
6. **Load and recovery.** For either lazy-load setting, separate base loading from ordered predicate application; overlap work where initialized resources permit. Base loading may return without waiting for full predicate catch-up, but it does not make every query immediately serviceable. Each query must satisfy the complete relevant apply-progress gate in Section 9.3. Suitable scalar indexes avoid raw-column reads; otherwise reuse on-demand access without an extra forced eager-load stage.

**Shared safety boundary.** Online application, recovery, and compaction must preserve the same committed logical mutation order, including updates that change later predicate matches. WAL commitment, QN apply progress, durable L0 completion, and durable target-segment coverage are different milestones. Stats and read-side progress cannot replace the two durable drain phases required before physical field removal. Base-load completion alone does not guarantee Strong-read readiness, a rolling-upgrade deadline, or availability targets.

**Current-code cross-check.** [Delegator delete consumption](../../internal/querynodev2/delegator/delegator_data.go) currently buffers PK deletes and forwards them, while the separate [WAL L0 materializer](../../internal/streamingnode/server/wal/vchannel/l0materializer/wal.go) retains delete handles and produces durable L0 work. The current [segment loader](../../internal/querynodev2/segments/segment_loader.go) loads resources followed by delta logs. These are reuse boundaries, not evidence that the proposed predicate flow or asynchronous query gate already exists.

## 2. Motivation and industry precedent

The existing expression-delete path retrieves matching PKs before applying PK-based deletes. At very high match counts, PK materialization, result-size limits, transfer, and repeated PK-to-offset lookup can become substantial costs.

### 2.1 Shared convention and comparison scope

The mature shared pattern is **evaluate a condition -> identify local rows -> mutate through a storage-owned entry**. An explicit SQL mutation operator is one realization; a segment-local query-delete loop is another.

The delete comparisons below inspect fixed release tags. Section 2.7 separately identifies Kudu's versioned operational documentation and moving-source references; Section 2.9 compares production base-plus-change query mechanisms. These establish implementation precedent, not comparative performance or proof of Milvus correctness.

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

### 2.7 Kudu: borrow base-plus-delta updates, not its tablet implementation

**Primary precedent for the update direction.** A Kudu Tablet contains multiple RowSets. For this comparison, a **DiskRowSet is analogous to a Milvus Sealed Segment**, not to a whole collection or Tablet. Kudu stores timestamped changes separately from base columns and applies visible deltas when reading. This is a structural analogy, not equivalent key/version semantics. See the [Kudu Tablet design](https://github.com/apache/kudu/blob/master/docs/design-docs/tablet.md).

| Kudu maintenance operation | What changes | Identity boundary |
| --- | --- | --- |
| Minor delta compaction | Merge delta files without rewriting base columns | Same RowSet and row IDs |
| Major delta compaction | Merge deltas into selected base columns and replace the relevant blocks | Same RowSet and row IDs |
| Merging compaction | Combine multiple DiskRowSets into a new DiskRowSet | Row IDs may change; still within the original Tablet |

These are distinct maintenance tasks, not creation of a new Tablet for each update. See [Kudu 1.18.1 background maintenance](https://kudu.apache.org/docs/background_tasks.html). `DiskRowSet::MajorCompactDeltaStoresWithColumnIds` publishes changed RowSet metadata and swaps its base/delta components; it does not overwrite old file contents in place. See [diskrowset.cc](https://github.com/apache/kudu/blob/master/src/kudu/tablet/diskrowset.cc). The two `master` references were reviewed on 2026-10-20 and are not immutable release snapshots.

**Adopt:** separate foreground deltas from base-data rewriting, with version-aware application and subsequent consolidation.

**Adapt:** use Milvus segment lifetimes, timestamp semantics, and committed VChannel mutation ordering. Whether Milvus can preserve segment identity or rewrite only selected columns during consolidation remains open.

**Do not copy:** Kudu's key uniqueness, tablet replication, REDO/UNDO representation, or compaction protocol. This precedent does not prove lower Milvus latency or solve stale scalar/vector indexes after an update.

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

Applying the predicate against the old base value would silently delete a live row. Replaying all deletes before all updates, or grouping mutations by field without preserving dependencies, is therefore invalid. Loading and compaction must preserve the same semantics even if updates and deletes reside in separate durable files. Their merge/progress protocol remains open.

This also qualifies index-based predicate execution: an index over unmodified base data alone cannot establish the correct match set after visible updates to that field. Scalar-index, raw-column, and cached-expression access must represent the same versioned row view. Sections 10.6–10.7 settle the field-version and query-merge direction, not its implementation. The existing predicate-load policy still avoids an extra eager-load requirement; it does not itself provide this update-aware overlay.

The goal is ordered, timestamp-correct mutation application across all three paths. It does not establish client-request idempotency, rollback, or historical field-value retention merely by referring to WAL replay.

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

## 9. System-level integration checkpoints

The query-side Segcore change is only the local application mechanism. Compaction, scheduling statistics, target-QN segment loading, and L0 dispatch require integration before supported predicate deletes can replace expanded PK deletes end to end. Compaction evaluation now favors a restricted Go-layer evaluator; its exact supported set and execution contract remain open. The directions below separate scheduling hints (Section 9.2), target-QN loading (Section 9.3), and durable L0 dispatch (Section 9.4); agreement on one does not solve the others.

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

### 9.2 Providing delete-impact statistics to MixCoord

Decision status: **collection boundary settled on 2026-10-19; progress-bound observations and predicate-backlog scheduling clarified on 2026-10-20**. Collect cumulative applied-delete counts from loaded sealed segments through a dedicated stats RPC with independent polling, as a best-effort enhancement. Compaction must remain schedulable without those observations, using outstanding predicate counts. Detailed wire contracts, thresholds, and implementation remain deferred.

Current single-compaction scheduling uses `Statistics.DeleteNumRows / Segment.NumOfRows`, together with delta-log count and size. `DeleteNumRows` is the sum of `EntriesNum` across delta binlogs, so it is an approximate signal even for PK deletes rather than a count of unique physical rows. See [data_coord.proto](../../pkg/proto/data_coord.proto) and [compaction_trigger.go](../../internal/datacoord/compaction_trigger.go).

For a predicate delta log, one `(expr, delete_ts)` entry may match zero rows or most of the segment. Recording `EntriesNum = 1` therefore makes the current ratio unusable: MixCoord cannot distinguish a negligible delete from an 80% delete without evaluating the predicate or receiving a materialized per-segment impact statistic.

Compaction-time evaluation alone cannot provide an advance estimate of reclaimed rows, because the result arrives after the scheduling decision. Exact hit counts are nevertheless **not a scheduling prerequisite**. Use two complementary signals: observed applied-delete counts indicate potential row reclamation, while outstanding predicate counts indicate maintenance/replay burden regardless of matches. Neither is the authoritative source of which mutations must be applied.

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

Under continued mutations, a cached observation can quickly cease to represent current apply progress. It remains valid for its original progress while its segment identity is valid, but its scheduling freshness is limited. Persisting loaded-time observations to S3 preserves history, not freshness; the added persistence/lifecycle work is not justified for the selected best-effort policy. Retain bounded in-memory observations where useful, invalidate retired identities, and fall back to predicate backlog rather than requiring a durable stats cache.

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

#### 9.2.6 Agreed fallback: outstanding predicate count

Use the number of predicate mutation records still requiring durable incorporation by data compaction as the initial hit-independent scheduling signal. This lets unloaded collections and segments with missing/stale QN observations receive maintenance without evaluating expressions just to schedule it.

- Count logical outstanding records, not physical files, replica copies, or distinct expression strings. Separate committed mutations with the same expression remain separate records; duplicate delivery of one mutation must not inflate the count.
- Online QN application does not retire this storage backlog: the segment's durable base/delta state still requires those records during recovery. Reduce the backlog only when durable compaction output and metadata establish that the corresponding records have been consumed, not when a worker merely finishes evaluation.
- The signal also applies when a valid QN observation reports zero deleted rows. Processing 100 predicates that match nothing can still eliminate 100 future replay steps; it is not evidence that 100 rows were deleted.
- Treat applied-delete count and predicate backlog as complementary signals, not mutually exclusive branches. Available stats may improve reclamation priority, but cannot suppress maintenance solely because their count is zero.
- This is a deliberately simple initial policy, not a claim that expression count is proportional to CPU/I/O cost or the only possible metric. Thresholds, prioritization against other compaction triggers, resource budgets, and anti-starvation behavior require validation.

#### 9.2.7 Deferred details; no detailed design yet

Remaining work is to define the RPC name and request/response fields, coherent count/progress sampling, batching and refresh parameters, collector ownership inside MixCoord, source/progress validation across replicas and reloads, and cache bounds. The predicate-count fallback direction is settled; its durable accounting representation, thresholds, and combined prioritization are not. Statistics loss may delay precise reclamation prioritization, but must not block backlog-based scheduling, cause missed delete application, or change query visibility.

This checkpoint settles the **statistics source, progress-bound observation model, collection-interface boundary, and hit-independent fallback signal**, not the compaction-side expression evaluator in Section 9.1 or durable expression replay. No code implementation is authorized or implied by settling this part of the design.

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

Discussion checkpoint: **2026-10-20**. The responsibility boundary and conservative-routing tradeoff are agreed. Payloads, publication/recovery details, and performance remain unvalidated.

#### 9.4.1 The missing problem and the existing reuse boundary

Removing Proxy's matching-PK query also removes the PK list used for L0 Bloom-filter dispatch. A predicate such as `pk > 5` cannot use a PK membership Bloom filter to discover all matching segments. The same issue applies to `UPDATE ... SET age = 18 WHERE pk > 5`: constant assignment simplifies the patch, not target discovery.

The current implementation, inspected at `fa995414c1`, separates:

1. **Candidate selection:** `selectFlushedSegment` / `BuildCompactionRequest` in [compaction_task_l0.go](../../internal/datacoord/compaction_task_l0.go) select targets from current storage metadata using collection, channel, partition, lifecycle, and data-time bounds.
2. **PK filtering:** `applyBFInParallel` in [l0_compactor.go](../../internal/datanode/compactor/l0_compactor.go) tests deletion PKs against the candidates' Bloom filters. Even this is an approximate candidate test, not proof of exact row matches.

Reuse the first responsibility; do not pretend arbitrary predicate evaluation can be replaced by the second. Likewise, `pk > 5` does not identify one PK-hashed VChannel merely because it references the PK. Safe narrower routing for special predicates is an optimization, not a requirement for the general path.

The existing [compaction inspector](../../internal/datacoord/compaction_inspector.go) excludes concurrent L0 and Mix/Sort/Bump execution on the same channel, with corresponding exclusion for Clustering; same-channel behavior is covered by [scheduler tests](../../internal/datacoord/compaction_inspector_test.go). This is **execution exclusion, not a guarantee that all pending L0 or Streaming mutations have drained before a sealed rewrite**. Reusing it does not by itself prove the new predicate/update protocol.

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

This is a lifecycle model, not a wire format or a claim that one logical mutation occupies one WAL record.

- **Before dispatch:** retain the scoped predicate in durable L0, without freezing the request's QueryView segment list.
- **During dispatch:** select the current candidate segments and publish their predicate deltas under a validated scheduling/publication boundary. Prefer extending existing L0/compaction coordination over repairing stale ingress associations.
- **After dispatch:** the predicate is an input of the segment's normal delta lifecycle. Subsequent data compaction must evaluate applicable deltas in mutation order and incorporate their effects into its output. The selected direction does not add a second mechanism for migrating an unapplied ingress association to descendants.
- **If a rewrite completes first:** later L0 dispatch selects current replacement segments within the predicate's scope. It does not resolve the original QueryView IDs through a chain of successors.

L0 must remain a **lightweight distribution stage**: no expression execution, no reading business columns to determine matches, and no dependency on QN availability or a loaded collection. Online/load evaluation belongs to the query execution path; offline data compaction reads persisted data independently of query loading. The concrete update evaluator and representation remain open.

Release after mutation acceptance must not make L0 wait for reload. Outstanding predicates can remain in segment-owned durable deltas until offline compaction or a later load evaluates them. This does not change the separately retained admission prerequisite for accepting general predicate deletes in Section 1.1.

**Durability is still mandatory.** WAL commitment alone is not completed L0 dispatch. Predicate content and the metadata that makes it discoverable for the current targets must be durably published before dispatch is reported complete or the source is retired. The agreed high-level organization is a new segment-owned predicate delta file type batching multiple records, not a separate metastore association entry for every predicate/segment pair. This does not require exactly one physical file for the segment's entire lifetime; encoding, batching/rolling, and publication/cleanup remain open. The no-side-table decision removes ingress lineage repair, not ordinary segment metadata, completion accounting, or garbage-collection obligations.

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

#### 9.4.5 What is settled and what is not

**Settled:** no ingress QueryView-derived target table or dedicated lineage-repair mechanism; coarse scope-based dispatch against current storage metadata; no expression execution or loaded-QN dependency in L0; segment-owned predicate delta files batching multiple records; actual evaluation in online/load and background data-compaction paths; acceptance of possible zero-match work.

**Still open:** exact predicate-file encoding and segment metadata layout, batching/rolling and cleanup, retry/restart and partial-publication behavior, ordering/completion frontiers across PK deletes/predicate deletes/updates, coverage of old growing data and replacement outputs, and cost limits. Existing task exclusion and target selection are reuse candidates, not an end-to-end proof. The drop-field drain in Section 9.1.3 still needs its complete durable-prefix guarantee; coarse dispatch does not weaken it.

## 10. Update deltas: benefits and agreed query semantics

Discussion checkpoint: **2026-10-20**. Logical in-place updates through deltas, joint mutation-order correctness, versioned field values, and snapshot-aware base-index/delta query merging are agreed directions. The benefit analysis below is a set of hypotheses and evaluation questions, not measured results or an approved detailed update design. Prior large-delete tests motivate predicate deletion; they do not establish update-delta performance.

### 10.1 Why pursue this alongside predicate deletes?

The strongest candidate workload is a **small field patch to a wide existing row, particularly one containing a large unchanged vector**. Today's partial upsert reconstructs and reinserts the full row. A changed-fields delta could make the foreground payload reflect the patch instead of the entire row, while retaining base data for unchanged fields. Full-row replacement, vector updates, and workloads dominated by read-modify-write/conflict handling may offer different benefits.

| Potential benefit | Current cost it targets | Boundary and evidence needed |
| --- | --- | --- |
| Less full-row retrieval and Proxy materialization | Fetching old fields and rebuilding a complete insert payload | A delta path may still need target lookup, expression inputs, validation, or CAS; measure retrieved bytes, CPU, allocations, and RPC latency |
| Less foreground write/transfer amplification | Sending unchanged fields through insert messages and persisting replacement-row data | Measure WAL/network bytes and durable bytes including delta identity/version metadata; a smaller payload alone does not prove lower total cost |
| Less redundant work on unchanged vectors and indexes | Replacement rows reintroduce unchanged vector/scalar content into the insert/index lifecycle | Conditional on retaining a usable base index and correct updated-field access; measure index work instead of assuming it disappears |
| Less row-version and delete/insert churn | One logical change produces a replacement row plus deletion of the old version | Delta chains and retained versions also consume memory/storage; compare steady-state footprint and maintenance work |
| Less foreground interference with ANN/search | Retrieval, insertion, deletion, and subsequent maintenance compete for resources | Read-time delta application or index correction can move cost onto queries; compare latency distributions, errors, throughput, and component CPU/memory |

These are workload-dependent opportunities. **Update delta is not an elimination of compaction**: it changes the foreground representation and may defer/concentrate work into delta application or consolidation. More delta history can increase query work, memory, and load/recover time. Kudu provides a lifecycle precedent, not a speedup estimate.

### 10.2 Shared lifecycle versus feature-specific work

| Area | Shared requirement | Update-specific question still open |
| --- | --- | --- |
| WAL and segment routing | Preserve committed order, applicability, and timestamp semantics; use Section 9.4's coarse predicate dispatch | Physical row targeting, changed-field encoding, concurrency checks, and missing-row behavior |
| Segment application and queries | Apply to the intended row version and expose consistent snapshots; use Sections 10.6–10.7's versioned-value and scalar query-merge semantics | Physical version storage, overlay/index/cache implementation, scalar/vector support |
| Durable deltas and load/recovery | Reconstruct the same state without omitted or reordered effects | Delta representation, update/delete merge, replay cost, history retention |
| Compaction and schema changes | Preserve mutation dependencies through rewriting and field removal | Update consolidation, completion across segment replacement without an ingress lineage table, and extension of the delete drain to updates |
| Scheduling | Use hints for prioritization, not as mutation truth | Update density/bytes/read amplification cannot be inferred from deleted-row count alone |

This is the reason for a joint document, not a decision to force both mutations through one `DeletedRecord`, evaluator, file format, or scheduling metric. Existing delete decisions remain scoped to deletes unless explicitly extended.

### 10.3 Candidate benefit-validation scope

Before selecting a detailed update design, compare it with current partial upsert at equal logical work and required consistency. Vary changed-field fraction, row/vector width, indexed versus unindexed fields, repeated updates to hot rows, and cold versus warm reads. Include both foreground-only measurements and a sustained phase that exposes accumulated deltas and maintenance costs.

Measure client latency/throughput and concurrent ANN behavior; Proxy/query/streaming/compaction CPU and memory; retrieval, WAL, network, and object-storage bytes; index work; retained data/delta size; and load/recover latency after updates accumulate. Validate mixed update/predicate-delete correctness before treating any performance result as a benefit. No numerical target, benchmark configuration, or claimed gain is settled here.

The benefit target and initial update workload still require evaluation before selecting concrete update operators, overlay structures, index-correction implementation, or a durable format. Sections 10.4–10.7 settle the logical selection/assignment, ordering, field-version, and scalar query-merge semantics; they do not establish an implementation or performance benefit.

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

A later delete at `ts=115` evaluating `age < 18` must see `age=23`, not the base value 17 or the future value 25. The same field-resolution semantics must be used by queries and subsequent mutations, and reconstructed by load/recovery and compaction.

`T` denotes a supported execution snapshot, not the `GuaranteeTs` freshness lower bound. Examples assume the necessary base/version history is retained; this agreement does not promise arbitrary historical reads. Version storage, retention/GC, equal-timestamp ordering, and publication remain detailed-design work.

### 10.7 Agreed scalar query merge: base index plus snapshot-visible deltas

**Decision:** reuse the base scalar index without rewriting it for each foreground patch. Correct its query results using the field's visible updates. Section 2.9 provides mature precedents for suppressing stale base results and supplementing changes, with explicit implementation differences.

For a single-field predicate `P(age)`, fix a compatible base/index generation, segment layout, and supported snapshot `T`:

```text
I    = offsets matching P in the base scalar index
U(T) = offsets with a visible age update not incorporated into this base/index
D(T) = offsets in U(T) whose latest visible age value satisfies P

Matches(T) = (I - U(T)) union D(T)
```

Apply the row's insert/delete visibility rules as well. The equation expresses logical positive-match sets; it does not change the existing executor bitmap polarity. If an index produces only candidates, its required exact recheck still applies.

**Snapshot masking and supplementation rules:**

- Subtract by **row offset for this field**, not by equal values, PK alone, or the delta's own match set. Every offset in `I` whose indexed field has a visible unmerged update is removed before supplementation, including updates whose new values fail the predicate.
- `U(T)` is snapshot-specific: an update after `T` must not hide the old match. A patch to `gender` alone does not mask the base `age` index entry.
- Resolve each updated row to **one latest visible value** before evaluating the predicate. Do not union matches from every historical delta version. An assignment that restores the base value still follows the same rule: mask the old entry, then re-add it if the visible value matches.
- Supplement independently of base-index hits. An empty base result cannot suppress the delta branch, and base-only min/max or other pruning must not discard updated rows that could match.
- For compound predicates, combine corrected field selections with the proper boolean/NULL semantics, or evaluate against a coherent versioned row view. A stale base result from one operand must not prune candidates needed by another corrected operand.

For the `age` history in Section 10.6, the same offset demonstrates both removal and addition:

| Snapshot and predicate | Base hit at offset 3 | After masking | Delta supplementation | Final hit |
| --- | --- | --- | --- | --- |
| `T=95`, `age=17` | Yes | Retained; no visible update | None | Yes |
| `T=105`, `age=17` | Yes | Removed | 23 does not match | No |
| `T=105`, `age=23` | No | Empty | Latest visible value 23 matches | Yes |
| `T=125`, `age=23` | No | Empty | Latest visible value is 25; historical 23 is ignored | No |

**Initial execution direction:** a separate delta index is not required. Scan the updated rows for the referenced field, resolve their latest visible values, and evaluate the predicate; retain base-index access for unaffected rows. This is not a requirement to scan every historical delta record on each query. The version lookup/iteration structure is deferred. As update density or retained history grows, evaluate delta indexing or consolidation; scanning deltas is not assumed to remain cheap indefinitely.

**Coherent publication requirement:** a read must use a mutually consistent base/index generation, updated-offset set, resolved values, and complete apply progress. Otherwise it can observe a new mask without its replacement value, or an advanced progress marker while matches are still stale. Pin the relevant state across evaluation; exact synchronization and publication granularity remain open. Rebase/consolidation may retire overlay entries only when the replacement base/index and retained-reader contract safely cover them.

**Scope and current-code boundary:** this settles scalar-predicate result semantics, not a C++ data structure, delta-index format, vector-index update algorithm, or cache invalidation protocol. The current [UnaryExpr.cpp](../../internal/core/src/exec/expression/UnaryExpr.cpp) routes evaluation through scalar-index/raw-data paths and performs chunk pruning; [Expr.h](../../internal/core/src/exec/expression/Expr.h) manages access-path-dependent prefetch. These are integration points to make update-aware, not evidence that this merge already exists. Projection, raw scans, expression caches, and later mutation predicates must agree with the same field versions; correcting only returned output values cannot prevent missed index matches.

## 11. Deferred integration and open contracts

This document does not fully specify WAL payloads, deterministic replay, L0 persistence and catch-up during segment load, release/reload behavior, CDC, compaction execution, or replica-wide visibility. Section 9.1.3 settles the drop-field safety requirement, both mandatory drain phases, and the all-VChannel release condition, while preferring unified drainage and reuse of existing checkpoint/L0 progress. Section 9.3 settles target-QN predicate-delta loading's access-path policy and records the asynchronous-recovery direction: separate base load from background apply, with per-request `GuaranteeTs` admission against complete relevant progress. Its PG precedent and applicability limits are explicit; the progress protocol remains open. Section 9.4 settles coarse predicate dispatch and rejects ingress association tables with lineage repair; its durable protocol and cost remain open. These remain required integration work, not benefits already delivered by local bitmap application. Support for accepting general predicate deletes against unloaded data is deferred; this is distinct from allowing already-accepted mutations to progress through L0 and offline compaction after release.

Before coding the predicate-delete part, finalize:

- The delete plan/context and write-capable segment handle, including the CGo boundary.
- Bitmap ownership, whole-segment versus batch-local input, and the applied-row-count definition.
- Mutation serialization, snapshot publication granularity, and cancellation/partial-application behavior.
- The restricted Go compaction evaluator's exact supported expression/type set, parity tests, and schema/version contract; preserve the agreed legacy fallback for expressions outside that set.
- The per-VChannel completion proof for both Streaming -> L0 and L0 -> target segment delta-log publication, including coherent checkpoint/L0 observations, old growing data, replacement segments, retry/restart, and gating every rewrite that can remove the field; preserve the two-phase requirement and unified-drain preference in Section 9.1.3.
- The dedicated stats RPC's detailed contract, coherent count/apply-progress sampling, source/freshness validation, and cache invalidation for Section 9.2's best-effort cumulative observations; define predicate-backlog accounting and scheduling thresholds without requiring stats availability or introducing a durable stats cache.
- The predicate-delta loader/CGo wiring, replay bookkeeping, failure handling, and resource budgeting; preserve Section 9.3's access-path policy and per-request progress gate. Define the initial readable state, complete apply-progress publication/aggregation, and recovery-mode/readiness controls without silently weakening requested consistency. Delegator L0/streaming catch-up must participate in the completeness proof, but its detailed protocol remains separate work.
- The durable coarse-dispatch protocol and fanout measurements in Section 9.4; preserve lightweight L0, current-storage-view selection, and release-independent progress without reintroducing a request-time association table or lineage-repair subsystem.

For update deltas, Section 10.4 settles multi-row predicate selection and multi-column constant assignment, while excluding old-value-dependent assignments. Section 10.5 settles the observable results of independent and overlapping patches and their predicate-order dependency. Sections 10.6–10.7 settle versioned field values and scalar query merging: mask snapshot-visible updated offsets in the base result, then supplement matches from each row's latest visible field value, initially without requiring a separate delta index. The benefit target, exact supported fields/types/predicates, and UPSERT semantics still require discussion. Detailed physical row targeting, value-version storage/GC, overlay/index/cache implementation, coherent publication, concurrency enforcement, durable replay, load, and compaction mechanisms remain open. The shared ordering invariant in Section 6 applies to all of them; it is a requirement, not an implemented guarantee.

No implementation or performance validation accompanies this draft. This revision records agreed query semantics and industry applicability boundaries; it does not authorize implementation or mark the complete update design as settled.
