# Delta Mutations — Real-Time PK Updates and Async Predicate Deletes

Date: 2026-10-27

Status: Architecture draft, narrowed on 2026-10-27. Phase one selects real-time PK point UPDATE, asynchronous predicate DELETE without cross-segment atomic visibility, and typed-predicate WAL messages with PK-equality-only UPDATE admission. Row insertion timestamps remain unchanged; updates carry independent mutation timestamps. Follow existing Milvus semantics rather than add historical-query support across Consolidation/full rewrite; protect in-flight readers without retaining folded history indefinitely. Query-side versioned update support is required. Row-version edge cases, exact API contracts, distributed publication, and implementation remain incomplete. No implementation or performance validation accompanies this document.

Source: [Frozen real-time design](20261016-predicate-delete-and-update-deltas.md), revision `94530c48ce3502b8fc212720630522766ee4913e`, plus the subsequent async and point-update discussions. This is the active follow-up document; its original `-async.md` filename is retained for link continuity. The initial all-async UPDATE proposal is superseded here, not in the frozen baseline. Section 12 identifies which earlier decisions are reused, adapted, or excluded.

## Core system benefits

**The first two benefits reduce foreground write costs; the third reduces background maintenance costs. PK point updates require query-side version support, but none requires query-side execution of pending predicate DELETE commands.**

- **Predicate DELETE: avoid foreground matching-PK expansion.** Append a supported predicate directly to WAL, without a matching-PK/count query or transporting a match-count-sized PK list.
- **UPDATE: avoid whole-row read-modify-write amplification.** Submit the supplied PK as a typed equality predicate and changed-field assignments, without reading or reconstructing the complete old row or generating a whole-row insert-plus-delete pair. Consumer-side PK lookup is still required.
- **Delta Consolidation: avoid full-segment rewriting and ANN-index rebuilding solely for scalar updates.** Preserve row count/order/offsets, replace affected column data, and reuse unchanged vector data and indexes. Changed scalar indexes must remain compatible with the replacement columns.

These are design targets, not measured gains. Restricting UPDATE to PK equality does not itself deliver the third benefit: retaining layout-preserving Delta Consolidation is essential. Predicate evaluation, data-side delta memory, QN update-history memory and query correction, replacement-column IO, and resource publication still cost resources. Existing partial upsert is not the selected implementation: its whole-row reconstruction and insert-plus-delete path would give up the intended update savings.

## 1. Decision boundary and status

**Phase one separates operation scope from visibility.** UPDATE targets one supplied PK and is real-time; predicate DELETE remains background work whose materialized effects become visible progressively. QN must receive/apply PK point updates and recover their unmaterialized suffix. QN does not load or execute pending predicate DELETE commands during fresh load, recovery, Reopen, or live serving.

"Real-time" means point updates become query-visible through the live apply/read-admission path, without waiting for Consolidation. It does not equate a durable WAL acknowledgement with completed application on every QN. "Async" applies to predicate DELETE visibility, not to UPDATE or ordinary insert/PK-delete processing. Neither changes the data-side complete ordered execution contract.

| Operation | Phase-one target | Serving path |
| --- | --- | --- |
| UPDATE | One PK equality and supported constant assignments | Stream to QN; resolve applicable offsets and append field versions; recover unapplied point-update history |
| Existing PK DELETE | Explicit PKs under the existing contract | Retain the existing online path and timestamp semantics |
| Predicate DELETE | Supported typed Boolean predicate | Execute on the data side; QN receives materialized deletion results, not pending predicates |

| Status | Meaning in this document |
| --- | --- |
| Agreed phase-one scope | Real-time PK point updates; asynchronous predicate deletes; no mutation-wide atomic visibility for predicate deletes; ordinary PK deletes retain their existing path |
| Retained design contract | WAL ordering, complete cross-type persistence/distribution, restricted expressions, complete-prefix execution, and layout-preserving Consolidation are extracted from the frozen design; they are not claims of implemented support |
| Reused query-side decisions | UpdatesRecord, snapshot-aware index/raw correction and projection, field locks, coherent Reopen, and published-boundary GC; not a new QN UPDATE-WHERE evaluator |
| Required integration | PK-targeted apply/recovery, update-aware read admission, baseline-aware deletion loading, and compatible index/state publication |
| Open decision | Mixed-operation semantics, exact completion/API and replica adoption, PK row-version targeting, Growing coverage, and resource/liveness limits |

**Agreed: one predicate DELETE does not require atomic query visibility across all affected segments.** Completed segment states may become visible while other segments still expose older predicate-delete results. This does not relax atomic commit of an individual Consolidation/compaction task's complete result, coherent local QN state publication, or data-side ordered mutation execution. It is not permission to expose a half-applied point update. Section 3.3 separates these boundaries.

### 1.1 Industry basis and applicability

Primary sources checked on 2026-10-27:

| Precedent | Adopt | Adapt or do not copy |
| --- | --- | --- |
| [ClickHouse classic ALTER mutations](https://clickhouse.com/docs/reference/statements/alter#mutations) replace each completed part independently, so a query can see modified and unmodified parts together; each part still applies mutations in creation order | Adopt progressive, non-atomic visibility across segments while retaining ordered execution within each segment; separate durable acceptance from visibility | Adapt to Milvus WAL/L0, task-atomic result commit, and QN Reopen. Do not copy the whole-part rewrite strategy or infer that task-local partial results are safe. This comparison is not to query-time patch application |
| [Kudu native row updates](https://kudu.apache.org/releases/1.18.1/docs/schema_design.html#primary-key-design) require the full primary key | Restrict the phase-one write target to a supplied PK, not an arbitrary data predicate | Do not import Kudu's uniqueness, row-existence, tablet, or acknowledgement semantics; Milvus physical row versions still require a targeting contract |
| [Kudu non-unique keys](https://kudu.apache.org/releases/1.18.1/docs/schema_design.html#non-unique-primary-key-index) add an auto-incrementing component; update/delete require the full effective key, obtained by a scan if necessary | Adopt the distinction between a repeated user key and an exact row identity | Do not assume PK-only local lookup identifies the latest Milvus row across segments, or silently add a scan/identity parameter to the agreed API |
| [Elasticsearch delete-by-query](https://www.elastic.co/docs/api/doc/elasticsearch/operation/operation-delete-by-query) selects from an initial snapshot and checks document versions | Comparison considered during the conflict discussion, not the selected same-row execution policy | Do not copy its scan/version-conflict skipping. The selected ordered executor deletes an earlier matched row and does not let a later point UPDATE revive it; a later upsert insertion is a different row incarnation |
| [MongoDB update commands](https://www.mongodb.com/docs/manual/reference/command/update/) separate a target selector `q` from modifications `u` | Keep target expression and assignments distinct in the logical interface | Adapt this separation to a bound typed WAL predicate; do not claim MongoDB persists our command format or copies our replay protocol |
| [Kudu background maintenance](https://kudu.apache.org/docs/background_tasks.html) distinguishes delta-store maintenance from merging RowSets | Separate delta maintenance from segment merging | Use Milvus's selected Delta Consolidation and versioned-read boundaries, not Kudu's UNDO/history-retention or tablet implementation |
| [Kudu MVCC model](https://github.com/apache/kudu/blob/master/docs/design-docs/tablet.md#mvcc-overview) separates insertion and subsequent mutation timestamps | Separate row birth from field-mutation time while resolving retained versions at a query timestamp | This is a modeling principle, not a promise to preserve Kudu's history or to make Milvus support historical queries after rewriting |
| [Iceberg deletion vectors](https://iceberg.apache.org/spec/#deletion-vectors) represent deleted positions for a data file and merge prior deletion state | Persist cumulative deletion facts without removing physical row positions | Bind the bitmap to a Milvus segment/layout and materialized prefix. Do not copy Iceberg's file format or snapshot protocol |

These are established principles, not an industry-standard hybrid protocol. Their composition with Milvus WAL, non-unique physical PK versions, asynchronous predicate deletion, real-time update overlays, and QN publication requires validation. We reject both unrestricted UPDATE admission in phase one and the earlier assumption that all QN update machinery can be removed.

## 2. End-to-end data flow

```text
SDK / Proxy: validate one PK point UPDATE or supported predicate DELETE
  no foreground data query on either native path
  -> StreamingNode / WAL: durable acceptance and logical order
  |-> serving consumer / Delegator
  |     PK UPDATE -> QN PK lookup -> UpdatesRecord -> read-admission completion
  |     PK DELETE / INSERT -> existing serving paths
  |     predicate DELETE -> no query-side command execution
  `-> persistence consumer: retain ordered commands of all three delta types
  -> separate PK-delete / predicate-delete / update files
  -> PublishDeltaPrefix: publish a complete VChannel delta range
  -> L0 dispatch: distribute all applicable instructions to current targets
  -> DataNode: execute a complete ordered delta prefix
       |-> Delta Consolidation: replacement columns + cumulative delete bitmap
       `-> ordinary full rewrite: surviving rows in replacement segments
  -> durable publication of the complete result
  -> QueryCoord / QN: fresh load or Reopen / replacement-segment handoff
  -> query-visible materialized state

The online update path and data-side materialization are separate WAL consumers.
Only predicate DELETE waits for materialization to become query-visible.
```

| Component | Responsibility | Must not do |
| --- | --- | --- |
| Proxy | Bind/validate typed predicates and assignments; restrict UPDATE to PK equality | Query matches, reconstruct old rows, or collect segment stats before native WAL append |
| SN persistence consumer | Buffer commands, persist complete ranges, recover outstanding ownership | Evaluate predicates or collapse dependent intermediate updates |
| MixCoord / L0 scheduling | Publish/distribute complete ranges using current storage metadata | Infer predicate hits from missing stats or depend on loaded QNs |
| DataNode | Execute ordered mutations and produce durable state | Evaluate historical predicates against an already-modified final base out of order |
| Serving consumer / Delegator | Forward and complete point updates; preserve update history/catch-up and existing insert/PK-delete safeguards | Treat queue acceptance as completed apply, or evaluate pending predicate deletes |
| QN worker segment | Load materialized resources; apply/recover PK point updates; provide snapshot-aware values and deletion visibility | Execute pending predicate-delete commands or use a generic UPDATE-WHERE executor for PK equality |

No predicate-delete evaluator is moved into Delegator as a workaround. UpdatesRecord belongs to the segment's versioned field state; narrowing target selection does not eliminate it. Existing PK-delete catch-up is a reuse point for point-update delivery, not an already implemented UPDATE path. Eligible Growing rows must receive live point updates and remain covered by data-side mutation processing; Section 13 tracks the lifecycle details.

## 3. Acceptance, ordering, and visibility

### 3.1 Durable acceptance is not query completion

A successful native WAL append means durable acceptance. For predicate DELETE, evaluation and visibility await background processing. For point UPDATE, visibility follows the real-time consumer and query-admission contract, not a Consolidation round. Neither acknowledgement alone proves completed QN apply, known affected-row counts, or adoption on every replica. No cross-channel transaction guarantee is added.

Queries combine a coherent materialized representation with visible real-time point updates and PK deletes, while predicate deletes may remain pending. Predicate-delete eventual visibility still requires a live scheduling/retry policy; no bounded latency is promised. A query is not thereby promised a complete global WAL-prefix snapshot including pending predicate deletions.

TSafe, distribution, and snapshot checks remain. **TSafe is not proof of asynchronous predicate-delete completion.** For real-time point updates, the selected live path must await actual apply before certifying the relevant serving prefix; recovering targets additionally need historical readiness. Exact acknowledgement, Strong/Session/explicit-MVCC behavior, and replica completion remain open. Section 9 distinguishes reused query mechanisms from hybrid-specific integration work.

### 3.2 Background execution preserves logical history

Use the relevant VChannel's committed logical TimeTick order, not physical WAL MessageID order, Proxy arrival order, upload order, or file listing order. The existing consumer reorders messages using confirmed progress; a maximum observed timestamp alone is not a complete-prefix proof.

Retain the effective WAL timestamp throughout buffering, persistence, and execution. Existing internal transactions use their effective commit TimeTick; this is not a new clock or a user DML-transaction facility. Native requests remain one mutation per target VChannel.

Later predicates see earlier applicable updates. Deleted rows cannot be revived by a later UPDATE. Preserve logical insert/commit timestamps and operation-specific eligibility: existing PK-delete semantics require `insert_ts < delete_ts`; later inserts are not retroactively changed. Repeated PK values may identify multiple physical row versions, so PK equality must not be treated as a unique-offset guarantee. Rewrites must not replace row age with file creation time.

Example for an eligible row initially containing `age=17`:

```text
@90 UPDATE SET age=23 WHERE pk=3
@95 DELETE WHERE age=23

After the real-time UPDATE is covered by query readiness: QN can show age=23.
The pending predicate DELETE need not yet hide that row on QN.
DataNode execution: age becomes 23, then the row is logically deleted.
After result publication: QN excludes the row using the materialized bitmap.
QN stores the point-update version, but never evaluates DELETE WHERE age=23.
```

The converse order also matters: an eligible row initially at age=18, predicate DELETE at 100, and point UPDATE to 23 at 110 ends deleted when the earlier DELETE matches. QN may temporarily show age=23 before learning the deletion; this is the accepted asynchronous predicate-delete visibility window, not the final ordered result. A second predicate DELETE after the UPDATE does not change the first deletion's result for that row: the row remains non-live. No conflict-skip algorithm or update-based resurrection is introduced. Section 3.4 distinguishes a later upsert insertion from a point update of the original row.

### 3.3 Agreed: progressive visibility, atomic task results

**Asynchronous completion and atomic visibility are independent choices. A predicate DELETE does not require a mutation-wide, cross-segment query-visible switch.** A query may combine coherent segment states at different predicate-materialization prefixes. This does not waive complete point-update application or permit fields from a half-applied update to become visible.

The selected precedent is ClickHouse classic mutations: completed parts are replaced independently, yet each part executes its mutations in order (Section 1.1). Borrow that distinction, not a weaker execution order or ClickHouse's physical storage protocol.

| Boundary | Required guarantee |
| --- | --- |
| One predicate DELETE spanning segments and maintenance tasks | Effects may become visible progressively; no all-segments-at-once publication requirement |
| One Consolidation or compaction task | Atomically commit its complete logical result, including output columns/segments, deletion state where applicable, covered prefix, and input/output ownership; do not publish a successful partial task result |
| One QN segment state | Publish compatible base, indexes, deletion baseline, and update/delete read views together; never expose incompatible generations |
| Data-side execution within a segment | Preserve the complete ordered mutation prefix: later predicates see prior updates; no omissions, reordering, or resurrection |
| One point UPDATE in QN | Complete all applicable rows/fields before certifying apply; field-level locks alone are not a visibility proof |

Files may be prepared and uploaded independently. Task atomicity concerns authoritative result publication and crash recovery, not simultaneous object uploads. A failed or uncertain commit must not expose half the task as completed. The concrete protocol remains implementation work, including the catalog batching risk in Section 11; the requirement is not a claim that the existing implementation already guarantees it.

For an isolated `DELETE WHERE age=18`, an allowed observation sequence is:

```text
Accepted: pending background work across several segments.
After 1 hour: completed tasks adopted by QN have removed 100 matching rows.
After 2 hours: more completed tasks have been adopted; 5,000 rows removed in total.
```

These are illustrative observations, not latency or throughput promises. Each observed segment state is complete; the example does not permit a task to publish only its new columns while omitting its deletion result. Task-atomic metadata commit also does not imply simultaneous adoption by every QN. Replica routing/completion semantics remain to be specified, and existing safe segment-replacement/handoff requirements are not weakened.

### 3.4 Trigger-time targets and later writes

An asynchronous predicate DELETE is scoped to eligible row state at its logical trigger time D, not a standing prohibition on that PK or on future matching values. Here D is the operation's effective WAL TimeTick, not the wall-clock time when a background task happens to run. The data side must evaluate the predicate against the state produced by preceding applicable mutations, not against a later QN value. Accepted commands cannot be lost or omitted.

The later discussion supersedes the earlier suggestion that point-update conflicts might skip an already-matching deletion. Distinguish a new row from a new field version on the original row; no new conflict-skipping policy is selected.

Distinguish two cases:

```text
Initial row R0: pk=100, insert_ts=80, age=18
@100 predicate DELETE WHERE age=18
@110 upsert inserts replacement row R1 with the same PK

R0 is in the earlier DELETE's time scope; R1 is not.
R1 surviving does not prove that deletion of R0 failed.
```

Current upsert packs insert and delete messages; partial upsert also retrieves/merges existing values. It is not the proposed field-delta point UPDATE. If the operation at 110 is instead a point UPDATE of R0, the row retains insert_ts=80. QN may temporarily show the updated value while DELETE@100 is pending, but ordered data-side execution deletes R0 first and the UPDATE has no effect on that deleted row.

This remains true for `predicate DELETE -> point UPDATE -> predicate DELETE`: a row deleted by the first predicate remains deleted. A client re-delete is a new operation, not a prerequisite for eventually applying the first accepted deletion. Point-update acknowledgement and affected-row reporting during this temporary online state remain open API questions.

Per-segment execution follows TimeTick order. The data-side executor considers all three delta types; QN orders its delivered point-update/PK-delete subset and does not execute pending predicate deletes. Progressive query visibility does not reorder the authoritative program or permit an effective deletion to be undone by UPDATE.

### 3.5 Repeated PKs and the current point-targeting direction

The earlier latest-only proposal would require cross-segment information: a segment's local latest row is not necessarily the global latest row for that PK. The subsequent discussion instead proposes applying the constant assignments to every eligible, still-live row with the supplied PK and `insert_ts < update_ts` in the operation's scope. This avoids selecting a global winner before applying the mutation; exact scope, ties, missing/deleted targets, result deduplication, and response semantics still require validation.

```text
Segment S1: pk=100, row insert_ts=80
Segment S2: pk=100, row insert_ts=95
UPDATE@110: SET age=23 WHERE pk=100
```

Under this direction, both eligible rows receive age@110=23 while retaining their original insertion timestamps, 80 and 95. An UPDATE creates field versions, not replacement rows. It does not reset row-version ordering to 110, insert a missing row, or revive a deleted row.

Do not add a global PK/version resolver merely to implement the superseded latest-only proposal. Equally, do not infer that all query/search paths select the latest whole row: unchanged columns and vectors can differ across repeated-PK rows, and their existing result-selection behavior needs path-specific validation. The timestamp model below does not change that behavior by rewriting row timestamps.

### 3.6 Agreed time model: row birth is not field modification time

Use one logical timestamp domain with distinct meanings:

| Timestamp | Meaning | Behavior under UPDATE and maintenance |
| --- | --- | --- |
| `insert_ts` | Birth/visibility time of a physical row incarnation, retaining existing insert/import commit interpretation | A point UPDATE does not change it; Consolidation/full rewrite preserves its effective insertion meaning, not the maintenance time |
| `update_ts` | Effective WAL TimeTick of the UPDATE | All assignments in one mutation carry this time; append field versions without retimestamping unchanged columns or the row |
| `Q` | Query MVCC timestamp used to resolve retained insert/delete/update versions | Not a guarantee that an arbitrarily old historical representation remains available after maintenance |
| `T` | Complete data-side mutation prefix represented by `base@T + delete@T` | Not a row birth time, last-write time for every cell, or a new mandatory query-rejection threshold |

Before folding, UpdatesRecord holds `(update_ts, NULL state, value/reference)` for changed cells. For each field, resolve the latest visible retained version at Q; unchanged fields fall back to base. Different field last-write times can form one coherent read at Q. A multi-field mutation still requires complete application before visibility is certified.

```text
INSERT@80:  pk=100, age=18, city="Shanghai"
UPDATE@200: SET age=23 WHERE pk=100

Before folding:
  row.insert_ts=80; base.age=18; age history=[@200 -> 23]
  with the required read readiness: Q=150 reads 18; Q=210 reads 23

After complete materialization through T=200 and protected publication:
  row.insert_ts=80; base.age=23; city remains "Shanghai"
  folded age=18 history is not retained for future historical queries
```

The insertion timestamp does not claim that current cell values still represent the row at insertion time. Keeping update_ts alone after discarding the old value cannot reconstruct that value. Section 8.5 defines the agreed history boundary.

Original insertion-time ranges are not expanded to include later point updates or maintenance times. Concrete `StartPosition`/`DmlPosition` fields retain their WAL/recovery semantics and must not be reinterpreted as the latest timestamp of all field modifications or assumed to be exact per-row min/max values. Delta coverage and materialization progress remain separate metadata.

## 4. Request, WAL, and expression contracts retained

### 4.1 Logical message shape

The following is conceptual, not a finalized protobuf or implemented message type:

```text
Properties: existing WAL envelope
  message_type: PREDICATE_DELETE | UPDATE
  message_version, vchannel, timetick, existing internal txn_context if applicable

Header:
  collection_id, schema_version

Body:
  partition_scope: ALL | PARTITION_IDS(...)
  predicate: bound typed Boolean expression
             UPDATE admission in phase one: EQ(PK column, typed literal)
             PREDICATE_DELETE: separately supported expression subset
  assignments:                            # UPDATE only
    - field_id
      value_expr: typed literal           # phase one
```

Persist logical field IDs/types and bound values, not raw text alone or a segment/index-specific execution plan. Business content belongs in the body; do not add ingress segment IDs, row offsets, match statistics, or query-expanded PKs. A PK literal supplied as a predicate operand is valid input. Public affected-row-count semantics remain open; an expression count is not a deleted-row count.

One SDK/Proxy/SN request contains one logical mutation. A point UPDATE names one PK and may patch several supported fields; a predicate DELETE may match many rows. Updating 1,000 supplied PKs means 1,000 logical point updates, not an arbitrary UPDATE predicate or a cross-PK atomic transaction. Phase one retains the no-batch request boundary. Future transport packing and physical file batching do not change logical atomicity.

**Keep the expression-capable WAL shape, not a PK-only replacement protocol.** For example:

```text
UPDATE body:
  predicate: EQ(Column(pk_field_id, INT64), Literal(Int64(100)))
  assignments:
    - field_id: age_field_id
      value_expr: Literal(Int64(18))
```

The supported consumer extracts PK=100 from this validated canonical form and uses PK lookup; storing an expression does not mandate generic expression evaluation. The scope, typed field binding, and operation TimeTick still constrain which physical rows are applicable. A future UPDATE-by-expression feature may reuse the predicate field only after capability admission, consumer execution, and recovery/upgrade semantics are implemented. Serializability is not authorization to append unsupported commands.

### 4.2 Validation and capability boundary

Retain the rule: **Proxy validates that the request is legal under version V; SN validates that V is still admissible at commit.** Static validation includes syntax, field/type binding, allowed assignments, and complete expression support. Exact schema/capability negotiation and write-side guards remain implementation work.

DELETE and UPDATE retain a shared typed expression vocabulary, but **not an identical phase-one admission set**: predicate DELETE accepts its supported subset; UPDATE accepts only canonical `PK = typed literal`. Reject non-PK conditions, PK ranges, PK-IN lists, and additional filters such as `pk=100 AND status='active'` for UPDATE. Do not solve arbitrary predicates and expand them into point updates in Proxy. SET has its own allowlist: typed constants, including explicit NULL for nullable fields; omitted fields are unchanged. QN still needs snapshot-aware value/validity correction for user queries over updated fields.

Retain unsupported DELETE -> legacy query-PK/delete, unsupported UPDATE target or SET -> reject before append, supported native mutation -> append without a foreground data query. Making point updates real-time removes the specific pending-async-update mis-selection problem only when that legacy query's chosen snapshot and readiness actually cover prior point updates. PK targeting or WAL acknowledgement alone does not prove this. Section 3.4 distinguishes later insertions from updates of an earlier matched row; exact legacy-path behavior still needs validation (Section 13.2). Never change routes after partial acceptance or approximate unsupported persisted nodes.

Carry forward the discussed field scope: ordinary eligible scalar fields, including indexed scalars, strings, JSON, and scalar arrays, use whole-cell replacement. Phase one excludes PK/partition/clustering-key changes, vector updates, function input/output updates, JSON-path or array-element patches, and append operations. Do not change data distribution through this feature. Exact supported types, sizes, and expression combinations still require validation.

The long-term assignment ceiling remains simple deterministic row-local values: same-row copies and basic arithmetic, with a few built-ins only for demonstrated needs. Full SQL, subqueries, joins, arbitrary UDFs, random/time-dependent computation, and external inference are not targets. These extensions are not phase-one support.

No new unloaded-collection admission capability is silently granted by moving execution to DataNode. Already-accepted work must progress independently of collection loading; whether acceptance itself drops the existing loaded-data prerequisite is an open product decision.

## 5. Buffering and atomic VChannel publication

### 5.1 Ordered commands and a common sync cut

The persistence consumer retains predicate, assignments where applicable, scope, schema interpretation, and effective TimeTick. It does not query rows or perform last-write-wins coalescing before evaluating intervening predicates.

Update and predicate-delete files remain separate types; PK deltas retain their existing form. Preserve ordered records within each file and merge across types at execution. Reuse PK-sync trigger categories such as bytes, oldest-pending age, explicit flush, and recovery/pressure requests, but publish a common complete VChannel cut across all three types.

For `(B,T]`, 100 update records, two PK deletes, and one predicate delete must all be persisted even when the sparse types are below their thresholds. An empty type needs no empty file, but its absence must be confirmed. Uploads can run independently; none may expose an incomplete usable range. Small files and slow-type stalls are accepted costs to measure, not reasons to weaken completeness.

### 5.2 Dedicated PublishDeltaPrefix contract

Keep L0's segment representation. Use the separately designed VChannel-level range publication interface rather than interpreting single-segment `SaveBinlogPaths` as a cross-type transaction:

```text
PublishDeltaPrefix(vchannel_identity, (start,end], complete_outputs, ownership)
```

SN selects/writes complete output. MixCoord's DataCoord module durably owns `delta_prefix_barrier[v]` and its output associations. Consumers acquire a coherent published view; staged files are not schedulable merely because they exist.

After validating lifecycle, ownership, and `start < end`, with current barrier B:

| Condition | Result |
| --- | --- |
| `start == B`, `end > B`, complete durable output | Atomically publish associations and advance to end; `RangeApplied` |
| `end <= B`, wholly inside the established prefix | `RangeApplied` for logical coverage; do not register the submitted files again |
| `start < B < end` | `RangeMismatch`: partial overlap; do not clip/merge |
| `B < start < end` | `RangeMismatch`: gap |

The range check and publication are one conditional logical transition. `RangeApplied` is not proof that a particular retry's files were adopted, that targets received the deltas, or that QNs see the result. Unknown transport outcomes require retry/readback, not speculative file deletion. Exact catalog transactions, fencing, consumer acquisition, and orphan cleanup remain correctness-critical implementation work.

### 5.3 SN recovery

Recover valid ownership, then read every relevant VChannel's authoritative published prefix and associations from MixCoord before scheduling new delta output. A failed lookup is not prefix zero. Skip already-covered delta rematerialization, including accepted publications with lost ACKs; continue with complete contiguous ranges beyond that boundary.

This does not skip WAL replay needed by inserts, metadata, or other consumers and does not replace the PChannel recovery checkpoint. If only objects were uploaded but publication did not commit, the range remains pending. A live publisher with an unknown outcome retries the same request until coverage or a genuine mismatch/ownership change is established.

## 6. L0 distribution retained

MixCoord selects candidates from current collection/channel/partition/lifecycle and safe logical data-time metadata. No request-time QueryView segment-association table, descendant-repair service, loaded-QN dependency, or expression execution is introduced into L0.

DataNode retains existing Bloom-filter filtering for explicit PK deletes. Predicate DELETE instructions are conservatively attached to every applicable candidate; zero-hit fanout is allowed. Phase-one UPDATE has an extractable PK, but that alone does not select a new L0 routing optimization: conservative dispatch is correct, and any future PK/Bloom-filter pruning must prove no false negatives across applicable row versions. Mutable-field statistics from old base values cannot justify dropping candidates after earlier updates. L0 does not evaluate mutations.

Let P be the complete published VChannel delta prefix and C the complete durably dispatched prefix. Dispatch chooses its own batches within P; it need not mirror SN sync batches. For example, publications `(0,100]`, `(100,200]` can feed dispatch intervals `(0,50]`, `(50,150]`, `(150,200]`.

Retain all three guarantees:

1. **Complete contiguous input:** include every applicable delta type/record in the selected interval.
2. **Ordered downstream visibility:** later intervals cannot become executable across missing earlier dependencies; physical tasks may prepare in parallel.
3. **Complete target coverage:** finish all required durable target associations before declaring that interval dispatched or retiring its source.

Asynchronous query visibility does not make partial data-side prefixes safe. Missing an earlier UPDATE can cause a later predicate to permanently delete the wrong row. Execution exclusion with rewrites, target changes, recovery, and prefix ownership still need verification.

## 7. Data-side ordered DeltaExecutor

Reuse the frozen design's restricted Go-layer DeltaExecutor direction. Predicate DELETE evaluates its supported WHERE; point UPDATE resolves PK equality and fills assigned constants. Preserve a shared typed predicate representation without requiring generic UPDATE-WHERE execution in phase one. Extend filtering into ordered state transformation; QN's separate point-update apply path must preserve the compatible value/row-eligibility semantics.

```text
Prepare the complete applicable ordered program once:
  [Update@100, PKDelete@105, PredicateDelete@110, Update@120]

For each base Record/Arrow batch:
  current values + logical live-row state + time/scope eligibility
    -> execute the program in order
    -> emit the maintenance operation's required physical result
```

Later WHERE evaluation uses earlier updates. A deletion marks the row non-live, and subsequent updates do not revive it. Do not run all deletes before updates, evaluate predicates on the final column values, or fold only selected updated fields while claiming the full prefix complete. A zero-match predicate is still consumed after successful complete execution/publication.

The shared contract covers Delta Consolidation and Sort, Mix, Clustering, and full-rewrite schema-bump paths. Version-only/additive schema operations do not automatically become full rewrites. Preserve existing TTL and rewrite-specific processing. A safe bypass needs proof that no row is eligible; rows inserted at `10..20` may be affected by mutations at `100..120`, despite non-overlapping time ranges.

**Retained phase-one capacity baseline: all applicable deltas are memory-resident for the active execution scope.** Base batching does not bound the program/payload footprint: each later batch may need the same commands. No spill/disk-cache solution is assumed. Large-value backlog can cause both DataNode executor pressure and QN update-history pressure. Exact admission limits, executor concurrency, and oversized-work handling remain open; PK targeting is not an OOM bound.

Execution failure must not publish a partial result or advance a complete boundary. Retry from a valid durable state with well-defined consumed-input ownership. Neither constant assignments nor PK-delete deduplication makes arbitrary predicate replay safe over already-modified data.

## 8. Delta Consolidation and persisted state

### 8.1 One complete prefix, two physical representations

For segment S and a complete materialization boundary T:

| State | Meaning |
| --- | --- |
| `base@T` | Field-value state after ordered execution of all applicable deltas through T, interpreted with `delete@T` |
| `delete@T` | Cumulative deleted-offset bitmap after that same execution, including UPDATE, PK DELETE, and predicate DELETE dependencies |

**Only the pair expresses the complete materialized result.** Consolidation cannot physically remove rows without changing layout, so the updated base may still contain the physical slots of deleted rows. QN does not evaluate pending predicate DELETE commands and must load `delete@T` to hide their materialized victims. The bitmap is durable state, not an optional export of QN's in-memory deletion cache. Real-time point-update/PK-delete suffixes above T are separate from this complete data-side prefix.

T certifies complete execution, not the newest individual UPDATE timestamp. All effectively updated columns must be represented before advancing it. Unchanged columns can be reused. No residual unapplied columns or missing predicates are allowed inside the claimed prefix. T does not overwrite row insertion timestamps.

### 8.2 Layout-preserving column replacement

Retain segment identity, physical row count/order, and offset mapping. For a changed `b` in group `[a,b,c]`, write a complete aligned replacement column and switch manifest mappings:

```text
Old manifest: group-0 [a,b,c] -> old files
New manifest: group-0 [a,c]   -> same old files
              group-1 [b]     -> replacement files
```

This is immutable-file replacement, not in-place mutation of S3 objects. The storage primitives investigated in the frozen design support column-mapping replacement; they are not an offset-addressed update API or an implemented Milvus Consolidation workflow. Unchanged vector data and ANN indexes remain reusable. Changed scalar indexes must be replaced/invalidated with a correct read path before exposure; serving a stale scalar index without an overlay would miss matches.

The old `b` bytes remain physically present in shared files still needed by `a/c`, though the new manifest no longer exposes them. Reclaiming those bytes requires later file/group rewriting. Subsequent rounds can replace an existing `[b]` group rather than continually add active groups. Several changed columns may be grouped together; fragmentation can justify regrouping/rewrite of a broader scalar range while retaining offsets.

### 8.3 Cumulative deleted-offset persistence

Each Consolidation round reads the old cumulative bitset, or an empty initial one, and writes a new immutable complete bitset. The retained phase-one choice does this even when the deletion set is unchanged; no incremental bitmap chain or no-change reuse branch is required.

`delete@T` needs no per-row delete timestamp: the new representation does not promise reconstruction of the folded deletion history (Section 8.5). It remains bound to the segment/layout, T, and publication generation. Publish replacement columns, compatible indexes/read resources, bitmap reference, and T as one coherent result. Writing one component successfully is not publication of the result. This is part of the task-atomic result contract in Section 3.3, not an optional optimization waived by cross-segment progressive visibility.

Example:

```text
Initial: offset 3 age=17; offset 8 age=23
@95  DELETE WHERE age=23             -> delete offset 8
@100 UPDATE SET age=23 WHERE pk=3     -> update offset 3

Materialized result:
  base@100:   offset 3 age=23; offset 8 still physically present
  delete@100: {8}
```

Loading only base would expose offset 8. Re-evaluating the old predicate against the new base would incorrectly delete offset 3. QN instead loads both results and does not re-execute consumed instructions at or before T. Applicable point updates and PK deletes after T still need query-side recovery/application.

### 8.4 Full rewrite and ownership

Ordinary Mix, Sort, Clustering, and full-rewrite schema maintenance consume the applicable base, materialized deletion state, and complete pending delta prefix. They may physically remove deleted rows and change offsets; old-layout bitmaps cannot be copied unchanged into their outputs.

Consumed instructions, including zero-hit predicates, are not carried forward as pending work. Later/undispatched deltas remain owned by the data-side pipeline and must reach the appropriate current targets. L0/rewrite exclusion is not global drainage. A full-rewrite task must commit its complete replacement/retirement result atomically, just as a Consolidation task must commit its complete materialized state (Section 3.3). Extending maintenance exclusion to the new Consolidation task and making its output publication crash-safe remain required work.

### 8.5 Agreed history boundary: follow existing Milvus rewrite semantics

**Do not add historical-query support across Consolidation or full rewrite.** Retain timestamp-aware reads while the required base and delta versions exist, and protect in-flight readers of that representation. After safe publication/replacement, the new representation need not retain already-folded intermediate values or physical rows removed by full rewrite. No UNDO, fixed history-retention window, or restoration of retired segments on demand is introduced.

The current Milvus baseline was checked against checkout `0b0ad3ee8d` on 2026-10-27 by source tracing, not an integration experiment:

- Full rewrite composes delete timestamps, filters eligible deleted rows, and omits them from its output. The replacement segment cannot reconstruct those rows merely because a request carries an older MVCC timestamp.
- Ordinary serving selects the current readable segment distribution; it does not choose a retired generation by Q. Existing pin/unpin lifetime protection covers in-flight users, not arbitrary future historical requests.
- The checked Proxy code marks `travel_timestamp` as no longer consumed. Internal MVCC timestamps, including iterator paths, still exist; they do not themselves supply cross-rewrite history retention. No general compaction-boundary `SnapshotTooOld` check was found in the traced ordinary query path. This observation is not a new promise that every old-timestamp request must succeed.

For example, a row inserted at 80 and deleted at 200 can still be read at Q=150 from a retained old segment. Once serving switches to the full-rewrite output that omits it, a new request carrying Q=150 cannot recover the row from that output. Folding `age=18 -> age=23` has the analogous history boundary: keeping insert_ts=80 does not restore the discarded value 18.

**The earlier proposed mandatory `Q < T -> reject` gate is not selected.** T remains a complete materialization/replay/GC boundary, not a newly mandated snapshot-admission floor. Preserve existing query admission behavior; do not promise exact historical reconstruction beyond what the currently retained representation supports. Implementation must still preserve a coherent view for each protected in-flight query, including multi-stage reads and failure cleanup.

Skipping Consolidation would only postpone history folding until the next full rewrite, while retaining update-history memory and query-correction costs. It is not required for historical-query compatibility. Keep Consolidation for its maintenance benefits and apply the same no-cross-rewrite-history contract to both paths. Histories above the published T remain necessary; this decision does not authorize dropping unconsumed mutations or suffix versions.

## 9. Query-side loading, serving, and Reopen

### 9.1 Fresh load and recovery

Load a coherent published base/index generation with its `delete@T`, then reconstruct the applicable point-update and PK-delete suffix above T. Pending predicate-delete commands remain data-side-only inputs; their absence from QN is intentional, not a replay hole in the selected online operation subset.

The retained handoff direction is:

```text
base@T + delete@T
  -> segment-owned point-update / PK-delete deltas
  -> applicable L0 point-update / PK-delete records
  -> Delegator buffer catch-up and final protected handoff at H
  -> subsequent live point-update / PK-delete delivery
```

Sources can overlap physically; establish complete, duplicate-safe logical ownership before applying records. DataNode still consumes all three delta types. QN must distinguish types and must not read/evaluate pending predicate-delete payloads. This is an explicit new capability, not permission for old loaders to ignore unknown records.

**The prior all-async claim that fresh load needs no UpdatesRecord reconstruction is superseded.** Point-update recovery has historical application work again. Reuse the frozen design's asynchronous-history / synchronous-live direction, with H proving complete handoff of the required online subset and local historical progress proving actual application. The segment-local wait and channel TSafe provide separate proofs; exact range selection, deduplication, and integration for this hybrid remain to be validated. Do not reinterpret either progress as completion of pending predicate deletes.

A large pending predicate-delete backlog is not evaluated during load. A large point-update backlog can delay read readiness and synchronous live forwarding. Base/index/bitmap IO and ordinary PK-delete catch-up remain; removing predicate replay does not promise backlog-independent recovery latency.

### 9.2 Serving

PK equality removes generic WHERE evaluation from mutation target selection, not from user queries. Retain the previously selected update-aware query contracts:

- For each data-access expression, compute base index/raw-chunk results, resolve visible field versions, and rectify the result. For a single-field predicate, matched offsets are `(base_matches - U) union delta_matches`, where U contains offsets with a visible update in the current lookup window, not every historically updated offset.
- A visible NULL version overrides base; absence of a visible version falls back to base. Correct both result and validity bitmaps. Logical AND/OR/NOT reuse three-valued logic without accessing UpdatesRecord.
- Phase one is serial `scan_base -> scan_delta -> rectify`; parallel execution waits for performance evidence.
- SELECT resolves visible updates first, including NULL, and reads base only for unresolved offsets. Actual IO savings depend on their physical chunk distribution and cache state.
- Use the same query snapshot and compatible state across filtering, projection, sorting/grouping/aggregation, and other supported field readers; audit all access paths, not only final output.

Use PK lookup and row/time eligibility to locate point-update targets; do not issue a query RPC or reconstruct unchanged columns. A PK does not necessarily identify one physical offset. Missing-PK outcomes, repeated PK versions, and interactions with upsert or pending predicate deletion remain explicit open contracts.

Field-level reader/writer locks protect Record memory structures. Complete apply/readiness publication, together with timestamp visibility, prevents queries from observing partial multi-field writes; locks need not be held across every field for the entire operation. Allocation failure must not certify success or let dependent operations overtake incomplete work. Exact retry/rollback/quarantine behavior remains implementation work.

For live point updates, await actual application before the channel pipeline certifies completion; queue acceptance is insufficient. For historical point-update/PK-delete replay, retain a local readiness check. The earlier `A >= min(Q,H)` plus TSafe coverage of Q is the reuse baseline for this online subset, not proof that predicate deletes through Q are visible. Verify explicit MVCC timestamps rather than only GuaranteeTs. Empty live ticks need not advance a segment history counter once history has completed through H. These are integration obligations, not current UPDATE support.

### 9.3 UpdatesRecord choices retained for point updates

| Aspect | Selected baseline | Remaining work |
| --- | --- | --- |
| Logical access | Field -> sparse ordered offset directory -> timestamp-ordered history | Field lookup container and exact HistoryHandle ownership/layout |
| Offset directory | Abseil `btree_map`, storing only offset + HistoryHandle | Point/range/insert/erase measurements; bytes per entry |
| Per-cell history | Boost.Container deque; timestamp predecessor search, append, prefix retirement | Block size, short-history overhead, lookup/append/GC benchmarks |
| Version | Timestamp, per-version NULL state, inline small value or owned payload | Physical packing, inline threshold, encoding |
| Large payload | Immutable Arrow owning buffer; shared owner retained during use | Allocation/reference overhead, accounting, limits; no custom 16 KiB allocator |
| Synchronization | Field-level RW locks; copy small values or retain payload owners before unlocking | Failure handling, contention measurements; row-level locks deferred |

These choices were settled in the frozen design and are reused, not reopened as generic library alternatives. Neither container choice nor PK targeting eliminates update-history residency or promises measured query performance. Histories remain outside movable B-tree entries; do not retain invalidatable iterators or bare payload pointers after unlocking.

### 9.4 Consolidation Reopen and GC

Reuse the existing state-publication mechanism rather than build an update-specific publisher:

```text
Prepare compatible new base/index resources, delete@T, and Record views at T
  -> establish required point-update / PK-delete coverage before switching
  -> close publication admission and drain queries and whole online mutations
  -> atomically publish base/index/baseline/Record read boundary together
  -> resume admission; retire old resources when safe
```

Reuse shared update/delete backing stores with generation-bound read views rather than recreate/copy all histories. Above T, preserve already-applied point updates and PK deletes, including arrivals during Prepare. The new `delete@T` supplies predicate-delete effects that QN never executed. Pending predicates are not caught up on QN as a prerequisite.

For Q at or after the materialized boundary, the new update view resolves versions in `(T,Q]`; deletion visibility includes `delete@T` and the applicable later PK-delete suffix. This is suffix-read semantics, not a new mandatory `Q < T` rejection rule or a cross-rewrite historical-query promise (Section 8.5). GC can reclaim histories at or before a successfully published T after old users are drained. Never use candidate T, historical apply progress, or L0 dispatch progress as reclamation authority. Under the field write lock, trim deque prefixes and erase an offset entry only when its history is empty; owning payload references protect concurrent readers.

**Catch-up proof still needs hybrid-specific closure.** A historical A need not follow empty live TimeTicks, so blindly requiring `A >= T` can stall forever. The reuse candidate is historical readiness through `min(T,H)` plus a properly scoped synchronous-live completion proof through T, followed by the existing protected switch. Validate target registration, races, gate ownership, and callback/lock ordering before treating that condition as implemented or sufficient on every entry path. Wait outside the gate needed by apply. Do not claim completeness of asynchronous predicates from the online proof; their prefix result comes from DataNode.

This does not make QN entirely unchanged:

- Integrate the cumulative deletion baseline and its boundary with the compatible published base/index state. Existing PK-based `LoadPush` is not a ready bitmap loader.
- Preserve later ordinary PK-delete effects; never reset a live DeletedRecord's accumulated knowledge to an older bitmap. For `delete@100={8}` and a known PK delete of offset 9 at 120, the new baseline must not resurrect offset 9 for reads where that PK delete applies.
- Keep baseline-aware deletion queries correct even when no timestamped suffix remains. Adapt duplicate suppression, counts, caches, and bitmap ownership without introducing another predicate executor.
- Preserve coherent state lifetime across Search/reduce/output and two-phase Retrieve (`ignoreNonPk`). Retain the same segment read lease through selection and deferred projection; do not acquire a second lease while a pending publisher waits for the first. Cross-phase lease integration remains work to verify, not an automatically solved property of Reopen.
- Keep the old state valid on failed preparation/publication. File reclamation uses successfully published ownership and reader protection, never a candidate T that may fail to publish.

The history policy is settled in Section 8.5: no extra historical-query support across Consolidation/full rewrite, no UNDO or fixed retention window, and no new mandatory snapshot-age rejection gate. This does not weaken in-flight protection. A query using the old representation must finish its dependent reads before that representation or its needed history is reclaimed. Search/reduce/output and two-phase Retrieve must not silently switch base generations midway. These lease/publication integrations remain implementation work, not a reason to reopen the settled history-retention scope.

## 10. Scheduling, progress, and statistics

### 10.1 Retained maintenance tiers

| Signal/work | Selected responsibility |
| --- | --- |
| Update-delta backlog | Determine Delta Consolidation timing; no extra Mix trigger solely from updates |
| Scalar column-group fragmentation / obsolete column bytes | Determine Consolidation rewrite/regrouping scope while retaining layout |
| Deletes and existing segment-organization conditions | Drive normal Mix/full-rewrite scheduling; execution consumes all applicable deltas |
| Pending predicates with zero or unknown hits | Still require eventual evaluation; exact async scheduling policy is open |

**Predicate-only liveness needs an explicit policy.** A workload with only predicate deletes cannot wait indefinitely for an UPDATE to trigger Consolidation or for QN to report their hits: QN will not execute them. The discussed trigger direction is oldest-pending age OR command count OR pending bytes. The user's 72-hour example is a possible trigger deadline, not a selected default or a proven query-visible SLA. Thresholds, scheduling priority, and completion budgets remain open. These signals are not deleted-row estimates or automatic new Mix triggers.

### 10.2 Statistics change with execution placement

QN can observe its point-update history and memory, published deletion baseline, and later PK deletes. It cannot report the actual matches of unexecuted predicate deletes. Counts remain identity/progress-bound observations, not timeless facts; unknown is not zero, and replicas must not be summed.

A data-side result determines `delete@T`, whose cardinality can describe materialized deletions. Reporting/using that value is a candidate integration, not a finalized stats RPC or scheduling policy. Keep statistics optional for correctness and separate from authoritative delta coverage. QN update-memory protection and DataNode executor budgets both need coverage; pooled compaction does not supply an assumed RocksDB-style direct feedback loop to frontend admission.

### 10.3 Do not collapse progress domains

| Progress/state | Meaning |
| --- | --- |
| WAL durable acceptance | The instruction is durably recorded, not materialized or query-visible |
| P, VChannel delta-prefix publication | All delta types through the cut have durable, discoverable L0 coverage |
| C, VChannel dispatch completion | Applicable targets have complete durable delta coverage; batching is independent of P |
| T, per-segment materialization | `base@T + delete@T` represents complete ordered execution for that segment |
| QN published generation | The compatible materialized resources actually installed by that QN; can lag data-side publication |
| H and historical A for a load attempt | Complete handoff versus completed replay of required point-update/PK-delete history; not pending predicate-delete execution |
| TSafe / distribution / query readiness | Must cover actual online point-update and PK-delete application under the selected integration; not asynchronous predicate-delete completion |

Also retain the checkpoint clarifications from the frozen design: RecoveryCP is PChannel recovery/truncation safety; the reported VChannelCP can lag it; an L0's DmlPosition is not whole-channel completeness. DeleteCP protects Delegator PK-delete buffer/L0 sources needed for load catch-up, not WAL durability or the new materialization boundary. `transform_materialized_time_tick` tracks the existing L0 materializer's persistence work, not QN visibility.

WALSummary has PChannel-scoped immutable chunks, potentially with VChannel sections, and its own recovery manifest; it is not the segment column-group manifest. Integrating new mutation types with recovery, Summary, retention, and truncation remains implementation work. Skipping query-side predicate-delete execution does not release data-side durability obligations or point-update catch-up obligations.

## 11. Durable publication and lifecycle obligations

- **Crash/retry:** preserve accepted commands and complete-prefix ownership across failure. SN recovery suppression, DataNode retries, and QN resource refresh are distinct responsibilities; none can infer completion from object existence alone.
- **Catalog batching risk:** retain the frozen design's independent risk item. One L0 task may update enough target/input SegmentInfo KVs to span several `SaveByBatch` / `MultiSave` transactions. Manifest-backed files do not make those catalog writes one crash-atomic commit. No remediation is selected here; partial recovered metadata must not become a complete executable prefix.
- **Schema changes:** reject incompatible post-barrier requests and durably drain/retain all needed pre-barrier mutations and old schema/data. Physical field removal must not destroy columns needed by pending WHERE/SET evaluation. Merely persisting an expression without its required input data is insufficient. Extend the prior complete-prefix drainage reasoning to all mutation types; detailed DDL completion remains open.
- **CDC:** native mutations describe logical user data, not cluster-local maintenance. Their supported replicated form must be WAL-carried logical instructions, with compatible schema/order and the same consuming semantics on the secondary; each cluster derives its own segments, offsets, bitmaps, and file paths. Do not replicate physical offset results as logical mutations or mark user data local-only. CDC enablement is not settled: until that replay path is supported and validated, the feature must reject use on a replicating topology rather than silently lose changes across switchover. This is a release safety requirement, not a claim of current support.
- **Compatibility:** rollout must distinguish new durable record types and publication metadata before any component encounters them. Old QNs cannot be assumed to apply point updates or load the new bitmap/baseline, and old DataNodes cannot silently skip instructions. Future UPDATE predicates remain rejected until all required consumers and recovery paths support them; a typed expression field is not a capability guarantee. Detailed version admission and upgrade order remain open.
- **GC:** retire source instructions only after complete durable successor ownership is established. Retain replaced data/bitmap files while referenced by old published states, snapshots, or readers. Existing live-segment orphan-cleanup gaps are not solved by this design; honor the [storage path and GC contract](../agent_guides/storage/path_contract.md).

## 12. Extraction map: retained, replaced, and removed

Section numbers below refer to the [frozen real-time document](20261016-predicate-delete-and-update-deltas.md).

| Original area | Current hybrid treatment |
| --- | --- |
| Core benefits; Sections 9.5–9.6, 10.4–10.5, 10.8 | Retain foreground savings, typed predicates and restricted assignments; narrow UPDATE to PK equality and make only predicate DELETE asynchronous |
| Section 6; Sections 9.7.1–9.7.4 | Retain logical ordering, common-cut persistence, atomic VChannel range publication, and recovery-first lookup |
| Sections 9.4, 9.1.4, 9.1.6–9.1.7 | Retain coarse dispatch, ordered DataNode execution, maintenance separation, stable offsets, column replacement, and persisted cumulative deletion state |
| Sections 4–5 | Exclude query-side predicate-delete execution; use direct PK targeting for UPDATE, not a generic mutation-WHERE evaluator |
| Section 9.3 | Adapt historical/live handoff and readiness to point updates plus PK deletes; do not claim online A/TSafe completes pending predicate deletes |
| Sections 10.6–10.7, 10.10 | Reuse version overlays, result/validity rectification, serial WHERE, and delta-first projection for real-time point updates |
| Sections 10.9.1–10.9.6 | Reuse Abseil B-tree, Boost.Container deque, Arrow owning buffers, field locks, and complete apply visibility; keep performance/layout details open |
| Sections 10.9.7–10.9.10 | Reuse shared Record views, coherent publication, reader protection, and published-boundary GC; revalidate catch-up for the hybrid operation subset |
| Section 9.2 | QN can report update residency, not pending predicate-delete matches; retain hit-independent burden and progress-bound statistics |
| Sections 9.1.3, 11–12 | Retain schema/durability/CDC/compatibility obligations and the catalog publication risk |

The earlier all-async exclusions of UpdatesRecord, update recovery, bitmap rectification, and update-history GC are explicitly superseded. The frozen document remains unchanged and supplies detailed rationale; it is not blanket authority to enable general UPDATE predicates or query-side predicate DELETE in this hybrid.

## 13. Design review: settled contracts and remaining work

### 13.1 Settled at the design-contract level

| Area | Selected contract | Not implied |
| --- | --- | --- |
| Product scope | Real-time single-PK UPDATE; asynchronous predicate DELETE with progressive cross-segment visibility | General UPDATE predicates, batch transactions, or a universal current-WAL snapshot |
| Deletion guarantee | Trigger-time row scope; earlier matching predicate deletion wins for the original row; QN may temporarily show a later point update before deletion publication | Deletion of later upsert insertions, conflict skipping, resurrection, or arbitrary lost deletes |
| Time model | Preserve row insertion time; UPDATE carries independent mutation time; Q resolves retained versions; T certifies materialization | Retimestamping rows/unchanged columns at UPDATE or rewrite time; T is a new clock or a mandatory read-rejection threshold |
| Frontend and WAL | No foreground data query for native operations; typed predicate retained; UPDATE admits only PK equality and constant SET | A finalized protobuf, arbitrary-expression capability, or known affected-row counts at append |
| Field scope | Eligible scalar whole-cell replacement; nullable values; exclude distribution keys, vectors, and function input/output columns in phase one | Every type/size/path has been implemented or benchmarked |
| Persistence | Ordered separate delta types; complete common-cut range publication; strict range extension and recovery-first lookup | Existing SaveBinlogPaths already provides cross-type atomicity |
| Distribution/execution | Coarse L0 dispatch with complete ordered coverage; DataNode merges all applicable delta types | QN executes predicate deletes or L0 evaluates expressions |
| Maintenance | Layout-preserving Consolidation; complete `base@T + delete@T`; updates do not trigger/accelerate Mix; normal rewrites consume all applicable deltas | Physical row removal during Consolidation or partial-prefix column folding |
| Query values | UpdatesRecord, index/raw result-validity correction, NULL semantics, serial WHERE, delta-first SELECT | Generic UPDATE-WHERE execution or no query-performance cost |
| Record storage/concurrency | Abseil B-tree + HistoryHandle, Boost.Container deque, Arrow owning payloads, field-level RW locks | Optimal block sizes, measured overhead, or bounded memory |
| Reopen/GC | Reuse the publication gate, shared backing/read views, atomic resource/baseline boundary, retain suffix, GC only from published T | Hybrid catch-up wiring is finished; candidate T or A is GC authority |
| History/atomicity | Follow existing Milvus semantics: no added history support across Consolidation/full rewrite; folded history may retire after safe publication; protect in-flight readers; complete task/local-state publication | UNDO, fixed retention windows, a mandatory Q<T rejection gate, cross-segment simultaneous predicate visibility, or old-snapshot reconstruction after folding |

"Settled" means selected semantics or a reuse direction, not completed implementation or proven performance.

### 13.2 Product and correctness questions needing the next deep dives

1. **Exact point-update target and result behavior.** Validate the current apply-to-all-eligible-PK-rows direction rather than require a global latest-row resolver (Section 3.5). Preserve update-only intent; define scope, ties, missing/deleted targets, responses/counts, result deduplication on each read path, and consistent QN/DataNode targeting. Row birth and update timestamps remain distinct regardless of the outcome.
2. **Point-update responses during pending predicate deletion.** The ordered final outcome and temporary QN visibility window are clarified in Section 3.4. Define what acknowledgement, affected-row counts, and status mean when QN temporarily applies an update to an original row that DataNode will delete. Do not reopen this as an assumed conflict-skip mechanism or confuse a later upsert insertion with resurrection.
3. **Mixed visibility and completion APIs.** Clarify WAL acknowledgement versus real-time update readiness, asynchronous-delete accepted/completed states, wait/status/cancellation behavior, replica adoption, and Strong/Session/internal-MVCC integration. Cross-segment progressive predicate visibility and the no-added-cross-rewrite-history boundary (Section 8.5) are settled; an exact point-update consistency contract is not.
4. **Legacy query-PK delete interoperability.** Ensure its actual query snapshot/readiness includes required earlier point updates; then the stale-async-update selection failure is removed. Still verify interactions with pending predicate deletes, later writes, and existing replacement-row upsert. Do not promise a globally ordered conditional mutation from this existing query-then-delete path without proof.
5. **Admission and liveness.** Decide unloaded-collection acceptance, idle Growing progress, predicate-only trigger thresholds, and whether any age target is a scheduling deadline or a visibility guarantee. Already-accepted work must remain recoverable and eventually processable independently of collection loading.

### 13.3 Implementation planning and validation still required

- **Load/live handoff and Reopen:** exact H ownership, duplicate-safe point-update/PK-delete replay, historical versus synchronous-live readiness, explicit-MVCC gates, and the two-part catch-up proof before publishing T. Do not reintroduce pending predicate execution on QN or a channel-wide minimum-A aggregation as an implicit fix.
- **Durable atomic publication:** concrete message/file/bitmap encodings, VChannel identity/fencing, range retries, target ownership, task commit/recovery, and resolution of the multi-batch catalog risk. Distinguish required atomicity from current implementation evidence.
- **Query integration:** integrate update views and DeletedRecord baseline/caches/counts with PublishedSegmentState; audit every field reader, scalar-index replacement, NULL path, multi-stage lease, failure cleanup, and GC lifetime.
- **Resources and scheduling:** data-side all-resident program limits and concurrency; QN history growth/write protection; command age/count/bytes accounting; regrouping scope, fairness, and oversized values. Measure short histories, contention, query latency, index reuse, and backlog clearance; no numbers are settled.
- **Lifecycle coverage:** reuse existing safe L0/growing time boundaries and maintenance exclusion where applicable, then prove no lost targets across flush, sort, rewrite/replacement, and recovery. Retention safety alone does not promise timely async visibility.
- **Schema, evaluator, CDC, and upgrade:** exact supported predicates/types and error behavior, drop-field drainage, commit-time capability guards, deterministic replay, cross-cluster logical mutation replication, and rolling-version admission. Replicated deployment must reject the feature until support is validated, not silently omit mutations.
- **Observability:** point-update memory and async-delete burden/completion metrics, materialized deletion counts, generation/freshness binding, and replica deduplication. QN cannot count matches of predicates it never executes.

The original real-time design remains frozen. This active hybrid reduces target-selection and predicate-delete serving work, but deliberately retains real-time field-version reads. It is not a finished implementation plan.

## 14. Verification plan

These are required future checks, not completed tests or measured benefits:

| Scenario | Required evidence |
| --- | --- |
| Native predicate deletion or 1,000 supplied point updates | No foreground matching-PK query, whole-row retrieval, or request-time segment stats; UPDATE messages retain typed PK equality and independent logical operations |
| Unsupported UPDATE predicate | Reject non-PK/range/IN/additional-filter input before WAL append; do not expand it via query or enable it merely because the schema can encode it |
| One predicate DELETE spanning independently completed tasks/segments | Permit coherent segment states with different predicate completion progress; no mutation-wide visibility barrier; DataNode still executes in order |
| Failure during one Consolidation/full-rewrite task's result commit | Never expose a partial authoritative task result or advance consumed-prefix/input-retirement metadata without its complete outputs; validate recovery as well as the live path |
| Large pending predicate-delete backlog during fresh load/recovery | QN excludes predicate-delete commands; published base/bitmap plus point-update/PK-delete recovery still work; do not claim point-update backlog is free |
| Point-update historical replay versus live apply | Complete handoff without holes; no repeated logical updates from overlapping sources; queries wait for required history and actual synchronous live completion |
| Real-time UPDATE followed by legacy query-PK DELETE | The selection query reads the required updated values under its actual snapshot/admission contract; WAL acknowledgement alone is not accepted as evidence |
| Earlier predicate DELETE followed by replacement-row upsert | Delete the eligible old row when matched; do not retroactively delete the later insertion merely because it has the same PK |
| Earlier pending predicate DELETE followed by point UPDATE and another predicate DELETE | QN may temporarily show the update; DataNode executes in order, preserves the first matching deletion, and never revives the row; validate response semantics separately |
| Repeated PK in S1 at 80 and S2 at 95, followed by UPDATE at 110 | Validate the eligible-row targeting direction and read-path deduplication without changing row timestamps to 110; include a DataNode task that reads only one segment, recovery, ties, and replacement-segment races |
| Interleaved updates and both delete types | Independent ordered-row oracle agrees with DataNode output, including NULL, zero hits, later inserts, and no resurrection |
| Sparse cross-type sync and out-of-order uploads | No partial range becomes schedulable; strict range extension, retries, MixCoord restart, and ownership fencing preserve coverage |
| L0 fanout/rewrite races | No missing targets, duplicates with repeated logical effects, or prefix holes; test more metadata KVs than one catalog batch can hold |
| Multiple columns changed within one prefix | All effective changes are represented before advancing T; unchanged vector data and ANN indexes remain reused |
| Non-NULL/NULL scalar point updates with indexes | Before folding, base index/raw results are corrected by visible versions; SELECT honors explicit NULL; after folding, compatible indexes/read views prevent stale matches |
| Predicate deletion and crash/reload | `base@T + delete@T` restores all deletions without re-evaluating consumed predicates or relying on QN memory |
| PK deletion at 120 while installing `delete@100` | Preserve the later timestamped deletion, correct counts/deduplication, and snapshot behavior; no resurrection |
| Point update at 120 while preparing `base@100` | Retain the applied suffix through atomic publication; GC only published prefixes; shared payloads remain alive for every reader |
| Idle live ticks and materialization beyond historical H | Reopen readiness uses a validated historical/live proof rather than waiting forever for historical A to follow empty ticks |
| Reopen failure and multi-stage reads | Failed Prepare does not corrupt serving state; Search/reduce/projection and two-phase Retrieve never mix incompatible generations |
| INSERT@80 followed by UPDATE@200 before folding | Preserve insert_ts=80; when read readiness permits, Q=150 resolves the old field value and Q=210 resolves the update; unchanged columns are not retimestamped |
| Consolidation or full rewrite folds UPDATE@200 | Preserve row insertion meaning and latest materialized values; retire folded versions only after safe publication; retain all required suffix history; do not require old-value reconstruction or add a Q<T rejection gate solely for this feature |
| Old query still using the pre-fold representation | Protect its base/history through dependent read stages; retirement waits for the relevant users, without promising that a later new old-timestamp request can reuse the retired state |
| Predicate-only, unloaded, or idle-Growing workload | Selected future liveness/coverage policy eventually produces the required durable result without QN predicate execution |
| Full rewrite after Consolidation | Consume old-layout bitmap and pending ordered deltas; write correct survivors without carrying obsolete offsets |
| Compatibility, DDL, and CDC | Unknown instructions are not silently ignored; old input columns remain available until safe; replicated use is rejected until supported or validated end to end |
| Resource/benefit comparison | Measure foreground bytes/latency, DataNode peak memory and IO, index rebuild work, QN load/query latency, and accepted-to-visible delay separately |

Cross-segment predicate-delete visibility tests exercise the selected progressive contract. Task-result, point-update completion, and local-state atomicity remain mandatory. Mixed-operation and replica-completion tests depend on the remaining contracts in Section 13.

## 15. Existing implementation anchors

The following source paths distinguish reuse from new work. They do not implement this hybrid feature today.

- [WAL message model](../agent_guides/streaming-system/message/message.md), [TimeTick/transaction guide](../agent_guides/streaming-system/wal/timetick_and_txn.md), and [WAL recovery](../agent_guides/streaming-system/wal/recovery-storage.md): infrastructure ordering, payload ownership, and persistence responsibilities.
- [Current partial-upsert retrieval](../../internal/proxy/dml/task_upsert.go) and [streaming append](../../internal/proxy/dml/task_upsert_streaming.go): partial upsert requests all fields, merges values, and packs insert plus delete; this is not the proposed point-update delta path.
- [InsertRecord PK lookup](../../internal/core/src/segcore/InsertRecord.h): `search_pk` returns local offsets meeting timestamp eligibility, not a cross-segment latest-row winner. This lookup is useful infrastructure, not a complete point-update targeting contract.
- [Delegator forwarding](../../internal/querynodev2/delegator/delta_forward.go), [delete pipeline](../../internal/querynodev2/pipeline/delete_node.go), and [Delegator](../../internal/querynodev2/delegator/delegator.go): existing synchronous delete completion/TSafe and catch-up are reuse points, not proof of point-update integration.
- [Current L0 materializer](../../internal/streamingnode/server/wal/vchannel/l0materializer/materializer.go): PK-delete output groups run per-group sync; this is not cross-type atomic range publication.
- [L0 scheduler/task](../../internal/datacoord/compaction_task_l0.go), [DataNode L0 compactor](../../internal/datanode/compactor/l0_compactor.go), and [EntityFilter](../../internal/compaction/entity_filter.go): candidate selection, PK filtering, and current rewrite baseline.
- [Mix rewrite](../../internal/datanode/compactor/mix_compactor.go), [delete composition](../../internal/compaction/common.go), and [MVCC filtering](../../internal/core/src/exec/operator/MvccNode.cpp): full rewrite omits deleted rows; query-time timestamp filtering does not reconstruct omitted rows.
- [Proxy feature tracking at the inspected revision](https://github.com/milvus-io/milvus/blob/0b0ad3ee8dd69b1140af75753d5b78486b051adf/internal/proxy/dql/feature_usage_hooks.go#L84), [query iterator timestamps](../../internal/proxy/dql/task_query.go), and [readable distribution/lifetime](../../internal/querynodev2/delegator/distribution.go): deprecated travel timestamps, retained internal MVCC use, and in-flight protection are different from historical routing or a retained time-travel window. Section 8.5 records the inspected baseline and limits of the source check; the pinned feature-tracking file is newer than this design branch's source baseline.
- [Segment loader](../../internal/querynodev2/segments/segment_loader.go): fresh loading has a deltalog path; `ReopenSegments` calls segment Reopen, not that deltalog loader. New bitmap loading cannot be assumed to happen automatically.
- [Segcore Reopen/publication](../../internal/core/src/segcore/ChunkedSegmentSealedImpl.cpp) and [read/publication gate](../../internal/core/src/segcore/SegmentReadLease.h): staged resource preparation and lease-protected atomic state publication.
- [DeletedRecord](../../internal/core/src/segcore/DeletedRecord.h): current PK/timestamp ingestion, deletion masks/history, and caches. Its query path can return early for an empty timestamped list; a materialized baseline must still participate after integration.
- [Catalog batching](../../internal/metastore/kv/datacoord/kv_catalog.go): `SaveByBatch` invokes multiple `MultiSave` calls when necessary, rather than providing one transaction for an arbitrary task result.
- [Replication/CDC contract](../agent_guides/streaming-system/replication/replicate.md) and [storage/GC contract](../agent_guides/storage/path_contract.md): required lifecycle boundaries for the remaining implementation design.
