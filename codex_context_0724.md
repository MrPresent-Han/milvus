# Context handoff — add_function_field + bump_schema_version data-safety (2026-07-24)

Handoff for Codex to continue. Two intertwined tracks: (T1) a code review of the compaction
`RecordMaterializer`, and (T2) a datacoord/datanode data-safety test plan. Repo:
`/home/hanchun/Documents/project/milvus`, branch `test-master`.

## Environment
- Local cluster already running from this dir (mixcoord/datanode/querynode/streamingnode/proxy),
  proxy on `localhost:19530`, binary built 2026-07-24. Reuse it (no rebuild needed for T2 TC1/3/4/5;
  TC2 prod-oracle needs the datanode instrumentation patch + rebuild).
- pymilvus 3.1.0rc61 installed (exact version the earlier runs used).
- Artifacts dir (scripts, corpus, prod oracle): `~/hc-claude-projects/add_function_field_1m/`
  - `data_safety_test_plan.md` — the 5-case plan (T2).
  - `test_add_function_field_1m.py` — 1M synthetic BM25 test (markers).
  - `prod/` — 1M real wikitext-103 oracle: server `run_analyzer` tokenizer + exact Okapi BM25
    recompute + datanode `[schema-bump-verify]` bit-exact checks. Already ran green (20/20 queries).
  - `datanode_verify_instrumentation.patch` — the `[schema-bump-verify]` logging patch (not applied).
- Review notes: `review_0724.md` (in repo root).

## The choke point (verified)
"Fill absent output/missing field" is centralized in ONE component reached by 4 compaction paths:
- `internal/datanode/compactor/record_materializer.go` — `RecordMaterializer`:
  - `materializers` = function-output cols (BM25 sparse / MinHash), recomputed from input;
  - `missingFields` = plain schema cols absent from a segment, filled with default/null.
- Callers: `sort_compaction.go:263`, `mix_compactor.go:304` + `merge_sort.go:84` (mix plain +
  merge-sort), `bump_schema_version_compactor.go:458`/`:1012` (bump full-rewrite + incremental),
  `clustering_compactor.go:622`. L0 does NOT materialize (delete-only).

## Verified invariants (do not re-derive)
1. **Per-segment `existingFields`** decides materialize-vs-passthrough. Derived from the segment's
   PHYSICAL columns: `compactionSegmentStorageFields` (compactor_common.go:232) → V3 manifest field
   IDs `packed.GetManifestFieldIDs` (:234) or legacy field-binlog IDs (:236). So a mixed-state mix
   merge (some segments backfilled, some not) is safe by construction: each `seg` gets its own reader
   + own existingFields + own brand-new materializer (mix_compactor.go:292/304; merge_sort.go:75/84).
   "Recompute an already-present column" is structurally impossible.
2. **Materializer is stateless.** `Wrap`/`WrapWithSelection` (record_materializer.go:93/97) is a pure
   per-record transform; BM25 materialize emits per-doc term-freq sparse only (global stats applied at
   query time), no cross-record/segment state.
3. **`writerSchema` = dataCoord captured currentSchema** at bump-trigger time (policy captures
   `capturedSchema := proto.Clone(collection.Schema)`, compaction_policy_bump_schema_version.go:105);
   same for all input segments in one plan. Only the 3rd arg `existingFields` differs per segment.
4. **Atomic commit — no manifest↔schemaVersion divergence.** New segment's `ManifestPath` AND
   `SchemaVersion` are two fields of the SAME SegmentInfo, committed in one `catalog.AlterSegments`
   txn (meta.go:2453/2455/2479 and 2562/2565). Datanode writes manifest to storage first; crash
   before reporting → `CompleteCompactionMutation` never runs → old segment intact, orphan manifest
   GC'd. Plus per-segment mutual exclusion: `isCompacting` + `isSegmentCompactionProtected`
   (bump policy :68-69; `CheckAndSetSegmentsCompacting` inspector:597) + pre-commit
   `ValidateSegmentStateBeforeCompleteCompactionMutation` (bump task:278).
5. **Cross-path value core is SHARED.** Compaction (record_materializer.go:72 `function.NewFunctionRunner`
   → :454/:495 `runner.BatchRun`) and ingest (streamingnode `function_materializer.go:60`
   `function.GetManager().Materialize`) both delegate to the same `function.FunctionRunner`. So
   cross-path equivalence risk narrows to (i) input assembly divergence, (ii) runner-version skew —
   NOT two independent math implementations.

## Filter principle (governs triage)
Only a failure that is **non-crashing AND commits a wrong result** threatens data safety. Crash-class
failures (panic / OOM-fatal / node death) are caught: nothing is committed until the atomic
`CompleteCompactionMutation`; old segment survives, retry re-runs. Triage every finding by
"does it crash, or commit wrong?" — only the latter matters.

## Proxy guards (context)
- `validateAddFunctionInputNotText` (proxy task.go:252, issue #51167): BM25/MinHash on a TEXT input
  field is rejected (kills the old LOB×bump "backfill never completes" defect).
- `validateAddFunctionRequiresStorageV3` (proxy task.go:229, called :1177): add_function_field needs
  ALL of `useLoonFFI` + `bumpSchemaVersion.enabled` + `storageVersion.enabled`; adding a function
  with bump OFF is rejected at DDL (obsoletes the old mix/test_mix_rewrite_bump_disabled.py premise).

## Review status (review_0724.md) — selection path (bump full-rewrite, the highest-risk path)
Path: `WrapWithSelection` + `selectedRecord` (record_materializer.go:97/200) +
`selectFullRewriteRecord` (bump:331). Only bump full-rewrite passes a non-nil selection (bump:532);
all other callers use nil (delete-filter done by the caller).
- ✅ Row-alignment invariant HOLDS: same `recordSelection.ranges` slice every field (record_mat:233),
  input text + computed sparse + pk/ts aligned.
- ⚪ Finding 1 (DOWNGRADED, not data-safety): swallowed `selected.err` for lazily-built base cols.
  Trigger is not OOM (Go OOM = crash); only a deterministic type-mismatch (appendValueAt arrow_util.go:52),
  and a nil column hits the writer's unchecked type assertion (segment_writer.go:344/357/363/371) →
  panic → crash → safe. Residual = lost root-cause message only.
- 🟠 Finding 2 (CONFIRMED, test-only dead code): `ttlValues` slice accumulated in selectFullRewriteRecord
  (bump:381-383) is discarded by production (bump:521 `_`) and only asserted by a unit test
  (bump_..._test.go:913). Redundant: writer computes ExpirQuantiles internally from the written TTL
  column (binlog_record_writer.go:256-274, GetExpirQuantiles :120-121). NOTE: `expireTs` (single value)
  IS used for the filter (bump:371) — ALIVE; only the SLICE accumulation is dead. Cleanup spans code +
  test:913.
- 🟠 Finding 3 (fragile, currently correct): asymmetric release ownership on the selection!=nil error
  path (loop bump:534 only releases record when selection==nil; relies on WrapWithSelection internal
  base.Release()). Document/unify.

## Open next steps
- Review: continue into `bm25FunctionMaterializer.Materialize` (record_materializer.go:454) input
  assembly (text column extraction, null/empty handling); then cross-path equivalence vs ingest
  `function_materializer.go` — check input-assembly + runner-version skew (invariant #5).
- Cleanup (awaiting go-ahead): Finding 2 dead-code removal (code + test); Finding 3 comment/unify.
- Tests (T2): TC1 mixed-state was reframed — since atomicity + per-segment existingFields close the
  corruption angle, TC1 becomes a value-consistency regression, not a data-loss hunt. Highest-value
  real risk is TC2 (scale exact-recompute) + a cross-path consistency check (invariant #5).
  Deterministic mixed-state construction (if still wanted): etcd-freeze
  `dataCoord.compaction.bumpSchemaVersion.enabled=false` mid-backfill (policy re-reads it each cycle,
  compaction_policy_bump_schema_version.go:48; bump is per-segment one-plan-each, :143), then manual
  `client.compact()`.
