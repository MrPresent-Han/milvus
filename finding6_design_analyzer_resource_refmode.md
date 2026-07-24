# Finding 6 design — analyzer FileResources for function-output materialization in RefMode

## Problem (root cause)
BM25/MinHash function runners were born on the StreamingNode ingest path, where the file-resource mode
is `[sync, close]` only (QNFileResourceMode). They resolve custom-analyzer resources via the C++
**global resource map** (populated by `SyncManager.Sync` → `UpdateGlobalResourceInfo`), so every runner
constructor hardcodes `analyzer.NewAnalyzer(params, "")` (empty per-call extraInfo).

`add_function_field` newly requires computing a function output during **compaction on the DataNode**
(bump/mix/sort/clustering full-rewrite → `RecordMaterializer` → the same runner). The DataNode supports
`ref` mode (DNFileResourceMode `[sync, ref, close]`). In `ref` mode `RefManager.Download`
(fileresource/manager.go:275) only ref-counts + downloads files and does NOT populate the global map.
So the runner's `NewAnalyzer(params, "")` has no resource source → deterministic failure for a
remote-resource analyzer. The text-index path was already fixed for this (per-call
`BuildIndexInfo.AnalyzerExtraInfo`), but the runner interface was never extended → Finding 6.

## Pre-check (validated): NewAnalyzer extraInfo semantics
`BuildExtraResourceInfo(storage, resources)` (canalyzer/c_analyzer_factory.go:96) returns
`{"storage_name": <storage>, "resource_map": {name→id}}` — a SELF-CONTAINED per-call name→id map.
`NewAnalyzer(param, extraInfo)` (:120) passes it to C++ per call. Empty extraInfo → analyzer falls back
to the global map (SyncMode); non-empty extraInfo → analyzer uses the per-call map (RefMode),
independent of the global map. Proof it works in RefMode: `createTextIndex` already builds its
`AnalyzerExtraInfo` via the same `BuildExtraResourceInfo` and succeeds in ref mode. So threading the
same extraInfo into the runner is the correct, mechanism-consistent fix.

Symmetry: DataCoord only sets `plan.FileResources` under `IsRefMode`. So SyncMode → resources empty →
extraInfo `""` → global map; RefMode → resources present → extraInfo carries the map. One gate, both
ends self-consistent, ingest path unchanged.

## API design — thread `extraInfo string` (task-scope single lease)

### 1. Signatures (add `extraInfo string`; leaf `NewAnalyzer(params, extraInfo)` already supports it)
- `NewFunctionRunner(coll, schema, extraInfo)` — 3 prod callers:
  - manager.go:950 — ingest (StreamingNode, `[sync,close]` only) → `""` (global map). Unchanged.
  - record_materializer.go:72 — compaction materializer → real value (this design).
  - embedding/runner.go:135 (`runOne`, shared by `RunBM25`/`RunMinHash`) — **ALSO a DataNode RefMode
    path**: callers `importv2/util.go:646` (import) and `external/function_executor.go:362`
    (external-table refresh). In ref mode these deterministically fail for a remote-analyzer BM25/MinHash
    exactly like compaction. Must thread a task-local extraInfo here too (prepare a lease at each caller,
    which has its own ChunkManager), OR track as a sibling fix (Finding 6b: import + external-table).
    The runner-side signature change below serves all three uniformly; only the lease plumbing differs.
- `NewBM25FunctionRunner(coll, schema, extraInfo)` → `NewAnalyzer(params, extraInfo)` / `NewMultiAnalyzerBM25FunctionRunner(..., extraInfo)`
- `NewMinHashFunctionRunner(coll, schema, extraInfo)` → `NewAnalyzer(params, extraInfo)`
- `NewMultiAnalyzerBM25FunctionRunner(..., extraInfo)` → `NewAnalyzer(param, extraInfo)`
- `NewAnalyzerRunner(field)` — ingest-only (manager.go:798), NOT on the materializer path → unchanged.
- Pass the STRING, not the resources: runner stays resource-agnostic, mirrors `BuildIndexInfo.AnalyzerExtraInfo`.

### 2. Lifecycle — one task-scope lease (helper), shared by materializer + createTextIndex
New helper (compactor_common.go):
```
func prepareAnalyzerExtraInfo(ctx, plan, cm, params) (extraInfo string, release func(), err error)
```
- `len(plan.GetFileResources())==0` → return `"", noop, nil`.
- else `GlobalFileManager.Download` + `analyzer.BuildExtraResourceInfo` → extraInfo; `release` closes over
  `Release(...)`.
- On Download-ok-but-Build-fail → Release internally before returning err. `release` ALWAYS non-nil.
  Guard with `sync.Once` for idempotency.

⚠️ Prepare LAZILY — gate on an actually-missing function output, not merely on `plan.FileResources` being
set. DataCoord sets `plan.FileResources` whenever the collection has FileResourceIds, but a compaction
whose input segments already contain every BM25/MinHash output builds no runner and (non-namespace
clustering) no text index. Downloading unconditionally would make such a compaction fail on an unrelated
resource-storage outage. So only acquire the lease when materialization will occur: check per-segment
`existingFields` vs the schema's function outputs first, and skip prep (extraInfo `""`, noop) when nothing
is missing. (mix/sort/bump that DO build a text index still need it regardless — gate on
"builds-text-index OR materializes".)

Each materialize path (sort/mix/merge_sort/bump/clustering):
```
extraInfo, release, err := prepareAnalyzerExtraInfo(ctx, t.plan, cm, params)
if err != nil { return ... }
defer release()                              // FIRST defer → LIFO fires LAST, after all closes
... existing per-path materializer setup, unchanged ...
createTextIndex(..., extraInfo)              // no longer Downloads/Releases internally
```

⚠️ Do NOT add a blanket `defer materializer.Close()`. Materializer ownership DIFFERS per path and must
be preserved (else double-close → double-destroy of the C tokenizer → crash):
- Reader-owned (materializer closed by `materializedRecordReader.Close`, record_materializer.go:312):
  `sort_compaction.go:270`, `merge_sort.go:89`, `clustering_compactor.go:628`. Leave as-is.
- Direct-owned (`defer materializer.Close()` already present): `mix_compactor.go:309`,
  `bump_schema_version_compactor.go:459/1006`. Leave as-is.
The ONLY new defer is `release()`, written first so it fires after the reader/materializer close and the
tokenizer no longer references the on-disk resource files.

merge-sort caveat: `mergeSortMultipleSegments` (merge_sort.go:24) is a FREE function that builds the
materializer at :84 and is called from `mix_compactor.go:493`, while `createTextIndex` runs later in
`mixCompactionTask.Compact`. Prepare the lease ONCE in the outer mix task and pass `extraInfo`
explicitly into `mergeSortMultipleSegments` (new param) → forwarded to its `NewRecordMaterializer`.
Otherwise this path silently keeps `""` or needs a second lease.

### 3. Refactor createTextIndex to consume extraInfo (remove its internal Download/Release)
`createTextIndex` (compactor_common.go:56) currently Downloads + BuildExtraResourceInfo + defer Release
(:108-114). Change it to take `extraInfo string` and drop that block. Direct callers to update:
mix_compactor.go:635, sort_compaction.go:562, bump_schema_version_compactor.go:623 (+ the two wrapper
methods mix:538 / sort:474). Avoids the double-download that would otherwise occur.

### 4. RecordMaterializer forwards extraInfo
`NewRecordMaterializer(schema, functions, existingFields, extraInfo string)` → `NewFunctionRunner(schema, fn, extraInfo)`.
6 call sites: sort_compaction.go:263, mix_compactor.go:304, merge_sort.go:84,
bump_schema_version_compactor.go:458/1012, clustering_compactor.go:622.

### 5. DataCoord side — clustering must drop the namespace gate
`compaction_task_clustering.go:376` currently also gates on `taskSchema.GetEnableNamespace()`, but normal
(non-namespace) clustering still calls `NewRecordMaterializer` (clustering_compactor.go:622) and can
backfill BM25/MinHash. Remove `GetEnableNamespace()` so the condition matches mix/bump:
`IsRefMode && len(FileResourceIds) > 0`. (mix/sort/bump already carry resources — sort/mix via
compaction_task_mix.go:375, bump via Finding 5.)

### 6. DataNode side — give the clustering task a ChunkManager
`clusteringCompactionTask` stores only `binlogIO io.BinlogIO` (clustering_compactor.go:65);
`NewClusteringCompactionTask` (:165) takes only a `binlogIO`. The helper needs a
`storage.ChunkManager` for `GlobalFileManager.Download(ctx, cm, ...)` (mix/sort/bump already have
`t.cm`/`t.chunkManager` — that is what `createTextIndex` uses). So the exact path enabled by removing
the namespace gate (step 5) cannot download resources. Fix: thread the `cm` already available in
`DataNode.CreateCompactionPlan` into `NewClusteringCompactionTask` + the task struct, OR narrow the
helper's downloader dependency to a `{Download, RootPath}` interface that `binlogIO` also satisfies.
Recommended: add the `cm` (uniform with mix/sort/bump).

## Call chain (final)
```
compaction task: prepareAnalyzerExtraInfo → (extraInfo, release); defer release()   // FIRST defer, fires last
  ├─ NewRecordMaterializer(..., extraInfo)                  // ownership UNCHANGED per path:
  │    │                                                    //   reader-owned: sort/merge_sort/clustering
  │    │                                                    //   direct-owned: mix-plain/bump
  │    └─ NewFunctionRunner(..., extraInfo)
  │         ├─ BM25 / MultiAnalyzer → NewAnalyzer(params, extraInfo)
  │         └─ MinHash             → NewAnalyzer(params, extraInfo)
  │  (merge-sort: extraInfo passed explicitly into mergeSortMultipleSegments)
  └─ createTextIndex(..., extraInfo)                        // no internal Download/Release
release() runs after all reader/materializer closes (LIFO) → tokenizer done before files removed
```

## Tests
- Runner unit: RefMode plain BM25 / MultiAnalyzer BM25 / MinHash resolve a remote-resource analyzer via
  per-call extraInfo (and `""` still resolves via the global map / SyncMode).
- DataCoord: non-namespace clustering carries FileResources in RefMode (mirror the bump/mix tests).
- Lifecycle: `prepareAnalyzerExtraInfo` returns non-nil release + releases on Build failure; empty-resource
  path returns `""` + noop; lazy-skip when no function output is missing.
- **Path-level RefMode wiring tests (critical — constructor tests alone miss mis-wiring)**: drive each
  distinct compaction path END-TO-END through `NewRecordMaterializer` (not just constructors) in ref mode
  with a remote-analyzer BM25/MinHash and assert materialization succeeds — so a path that forgot to thread
  extraInfo FAILS the test. Cover: mix plain, **mix merge-sort** (`mergeSortMultipleSegments`), sort,
  **non-namespace clustering**, and **bump partial + full** materialization. Also import + external-table
  if 6b is in scope.

## Scope / notes
- Blast radius: 4 runner constructors + NewFunctionRunner (3 callers) + createTextIndex (3 direct callers
  + 2 wrappers) + NewRecordMaterializer (6 callers) + `mergeSortMultipleSegments` free-func param +
  clustering DataCoord gate + clustering task ChunkManager (constructor + DataNode.CreateCompactionPlan).
  All internal APIs.
- Backward-compat: ingest + SyncMode unchanged (extraInfo `""`).
- Do NOT add a blanket `defer materializer.Close()` — preserve per-path materializer ownership (double-close
  crash). Only `defer release()` is new.
- Independent of Findings 2/4/5 (this is the shared-runner API debt); its own commit.
