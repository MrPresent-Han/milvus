# Review 2026-07-24 — RecordMaterializer selection path (bump full-rewrite)

Scope: correctness/robustness of the highest-risk materialization path — `WrapWithSelection` +
`selectedRecord` + `selectFullRewriteRecord`, i.e. the ONLY path where the materializer combines
row-selection (delete/TTL filter) with function-output materialization. Files:
- `internal/datanode/compactor/record_materializer.go`
- `internal/datanode/compactor/bump_schema_version_compactor.go`

## ✅ Core invariant HOLDS — row alignment is correct

The feared corruption ("selected rows misaligned with computed sparse / column skew") does NOT occur.
`recordSelection.ranges` are row intervals over the source `record`; `selectedRecord.Column`
(record_materializer.go:233) slices **every** field with the **same** ranges, and the function
materializer reads its input from that same `selectedRecord` base (record_materializer.go:110).
So input text, computed sparse, pk/ts, and all other columns are sliced by identical ranges →
per-row alignment preserved.

## ⚪ Finding 1 — DOWNGRADED (not a data-safety hole; diagnosability nit only)

Original claim: `selectedRecord.Column` (record_materializer.go:234) swallows `builder.Append` errors
(`r.err = err; return nil`) for lazily-built base columns, and `WrapWithSelection` only checks
`selected.err` inside the materializer loop (record_materializer.go:121), not after the downstream
writer pulls non-materialized columns.

Why it is NOT a data-safety hole (verified):
- `builder.Append` never errors on OOM. Go OOM is a runtime-fatal crash, not a returned `error`.
  Its only non-crash error is a type mismatch (`appendValueAt` arrow_util.go:52; the function even
  asserts `// a could never be nil here`) — a deterministic schema/type bug, not a transient one.
- If `selectedRecord.Column` returns nil, the writer reads columns via UNCHECKED type assertions
  `r.Column(x).(*array.Int64)` (segment_writer.go:344/357/363/371) → nil → **panic → crash**, never
  a silent write.
- A crash mid-compaction is safe: `CompleteCompactionMutation` (atomic) is never reached, the old
  segment is intact, retry re-runs. No committed corruption.
- Residual: at most a lost root-cause message (panic shows a nil type-assertion instead of the real
  "type mismatch"), and only reachable behind a pre-existing type/schema bug. Diagnosability nit.

## 🔴 Finding 4 — REAL BUG (FIXED): text-index stats files double-prefixed in bump full-rewrite

In `runFullSchemaRewrite`, after `createTextIndex`, a loop re-prefixed each text-index stats file:
`stats.Files[i] = fmt.Sprintf("%s/_stats/text_index.%d", basePath, fieldID) + "/" + f`.
But `createTextIndex` already returns manifest-absolute Files: it sets
`Files = metautil.BuildStatsFilePaths(statsBasePath, ...)` (compactor_common.go:200/192), and
`statsBasePath = <base>/_stats/text_index.<fid>` (compactor_common.go:141); `BuildStatsFilePaths`
prepends basePath (pkg/util/metautil/binlog.go:137) — with its own idempotency guard. The bump loop
re-prepended the same prefix (no guard) →
`<base>/_stats/text_index.101/<base>/_stats/text_index.101/<file>`. `AddStatsToManifest` then
relativizes this doubled path → the manifest points at a non-existent text-index location →
text_match broken for bump full-rewrite segments with a match-enabled TEXT field.

Proof it is wrong: mix/sort do NOT have this loop — they pass `textStatsLogs` straight to
`TextIndexStatEntries` + `AddStatsToManifest` (sort_compaction.go:489, mix_compactor.go:548, comment
"AddStatsToManifest stores the manifest-relative representation at commit time"); bump's own comment
claims "Mirrors sort/mix" yet diverged.

Fix (applied, working tree): delete the loop + the now-unused `UnmarshalManifestPath`/`basePath`, so
bump matches sort/mix exactly.

Open (G2): earlier `test_text_lob_drop_field_bump.py` (2026-07-08) reported "text_match intact" on a
full-rewrite — either the loop postdates it, or that test's TEXT field was not match-enabled (empty
`textStatsLogs` → loop skipped). Recommend a real-cgo regression: bump full-rewrite with an
`enable_match` TEXT field, assert text_match works (fails pre-fix, passes post-fix).

## 🔴 Finding 5 — REAL BUG (FIXED): bump omits plan.FileResources (RefMode analyzer resources)

`bumpSchemaVersionTask.BuildCompactionRequest` never set `plan.FileResources`, whereas mix/sort
(compaction_task_mix.go:375-384, covers both types) and clustering (compaction_task_clustering.go:375-383)
set it under `fileresource.IsRefMode(...) && len(schema.GetFileResourceIds()) > 0`. bump full-rewrite
rebuilds the text-match index via `createTextIndex`, which in RefMode `Download`s + registers the
analyzer resources (compactor_common.go:108-114) — without `plan.FileResources` the referenced
analyzer resource is unresolved. Confirmed hard-error, not silent: Rust returns "file resource not
found" (tantivy resource_info.rs:105) → propagates → CreateIndex fails → createTextIndex errors →
bump compaction fails. So it is a deterministic **liveness** bug in RefMode (bump keeps failing, old
segment intact, no silent-wrong index) — P1/P2, not P0. SyncMode unaffected (resources pre-synced).

Fix (applied, working tree): add the same RefMode-gated FileResources block to bump's
BuildCompactionRequest (mirrors mix/sort). UT added:
`compaction_task_bump_schema_version_test.go` `TestBuildCompactionRequest_BumpFileResourcesInRefMode`
(ref mode carries resources / sync mode skips / GetFileResources error propagates).

DataNode side needs NO change for this path: bump passes `t.plan` to `createTextIndex`
(bump:623), which reads `plan.GetFileResources()` and downloads + builds per-call AnalyzerExtraInfo
(compactor_common.go:108-114). Setting `plan.FileResources` in DataCoord is the complete fix for the
text-index path (same as mix/sort).

## 🟠 Finding 6 — OPEN (cross-compaction, NOT bump-specific, NOT fixed by Finding 5): materialization runner has no analyzer resources in RefMode (deterministic API gap)

Corrected mechanism (NOT a timing window): the BM25/MinHash runner
(`record_materializer.go:72` → `bm25_function.go:160/124`) calls `NewAnalyzer(params, "")` with EMPTY
per-call extraInfo and relies on the global resource map. That map is populated ONLY by
`SyncManager.Sync` (fileresource/manager.go:243 `UpdateGlobalResourceInfo`); `RefManager.Download`
(manager.go:275) only ref-counts + downloads files and NEVER updates the global map. Meanwhile
`createTextIndex` uses per-call `BuildIndexInfo.AnalyzerExtraInfo` (compactor_common.go:163), not the
global map. So in RefMode a function input with a remote-resource analyzer → global map empty →
`NewAnalyzer` fails deterministically. Not a window; a certain failure.

Because materialization runs the runner directly, this affects ALL compactions that backfill a
function output (mix/sort/clustering/bump) — Finding 5 does NOT fix it (it only wires plan resources
for `createTextIndex`, which the runner does not use). Proper fix is API-level: download+hold the
resources before the materialization loop, build a per-call AnalyzerExtraInfo, inject it into the
BM25/MinHash/MultiAnalyzer runner, Release after — do NOT merely "download earlier" and do NOT mutate
the global map in RefMode. Scope: every plan-construction path that can materialize a missing function
output (note: normal clustering currently carries FileResources only in namespace mode). SyncMode
unaffected. Separate design + commit.

## Filter principle (applies to the whole test plan)

Only a failure that is **non-crashing AND commits a wrong result** threatens data safety. Crash-class
failures (panic, OOM-fatal, node death) are caught by construction: nothing is committed until the
atomic `CompleteCompactionMutation`, so the old segment survives and retry re-runs. When triaging any
finding, first ask "does this crash, or does it commit wrong?" — only the latter matters.

## 🟠 Finding 2 — ttlValues is test-only dead output (dead code, test-protected)

`selectFullRewriteRecord` accumulates and returns `ttlValues` (bump:381-383, 393, 395). Consumers
across the whole repo:
- Production: `selection, _, err = selectFullRewriteRecord(record, …, nil)` (bump:521) — DISCARDED.
- Test: `selection, ttlValues, err := selectFullRewriteRecord(...)` (bump_..._test.go:908) then
  `s.Equal([]int64{keptTTLField, keptTTLField}, ttlValues)` (test:913) — asserted.

So it is **test-only dead output**: no production path consumes it, but a unit test locks it in,
falsely signalling it's an exercised feature. Verified redundant (not a missing wire): the writer
computes ExpirQuantiles internally from the written TTL column — `packedBinlogRecordWriterBase`
accumulates `pw.ttlFieldValues` during `Write` (binlog_record_writer.go:256-274) and
`GetExpirQuantiles()` = `calculateExpirQuantiles(ttlFieldID, rowNum, ttlFieldValues)` (:120-121);
the full-rewrite record still carries the TTL column, so quantiles are correct without the slice.

- Fix (cleanup spans code + test): drop the `ttlValues` param/append, change the return to
  `(*recordSelection, error)`, update caller bump:521, and remove the test:913 assertion + capture
  (the test's selection.ranges / Len / entityFilter-count assertions stay valid).
- Not a data-safety issue (passes the crash-vs-commit filter: affects nothing committed).

## 🟠 Finding 3 — asymmetric/implicit release ownership on error (fragile, currently correct)

On the `selection != nil` error path the loop does NOT release `record`
(bump_schema_version_compactor.go:534 `if selection == nil { record.Release() }`); it relies on
`WrapWithSelection` internally releasing `base == record` via `base.Release()`
(record_materializer.go:114/123/139). Currently correct, but the contract is implicit and asymmetric
with the `selection == nil` manual release — any new early-return added before `base.Release()` inside
`WrapWithSelection` leaks `record`.
- Fix: document the ownership contract, or unify release handling.

## Next
- Continue review into `bm25FunctionMaterializer.Materialize` (record_materializer.go:454) — input
  assembly (text column extraction, null/empty handling) and `runner.BatchRun` feeding.
- Then cross-path equivalence vs ingest side `function_materializer.go` (shared `function.FunctionRunner`;
  check input-assembly + runner-version skew).
