# Test C: partial-upsert resource increment

Status: completed and verified on 2026-10-30. Two 10,000-request partial-upsert
passes completed without errors. All 10,000 target ages were verified, 40 unchanged
field fingerprints matched, and the logical row count remained 1,000,000.

## Outcome

The first partial-upsert pass reproduced the raw-vector cache residency seen in
Query-only testing, plus additional write-path state. Its control-adjusted retained
RSS increment was **3.027 GiB**, versus **2.894–2.904 GiB** in the prior Query-only
rounds. The observed difference is approximately **126–136 MiB**, not a precisely
isolated causal write cost: operation duration and physical segment layout differ.

The repeat pass added **106.96 MiB** of adjusted retained RSS without loading
another collection-sized raw-vector cache. Total adjusted retained RSS after both
passes was approximately **3.131 GiB** above the initial baseline.

## Memory results

All absolute values below are GiB. Post120 is the final 15-second median of the
120-second idle period. Peak is the maximum sampled during requests.

| Pass | Control before → post120 | Test C before → peak → post120 | Adjusted peak increment | Adjusted retained increment |
|---|---:|---:|---:|---:|
| First upsert | 1.932 → 1.938 | 1.857 → 5.070 → 4.890 | 3.213 GiB | 3.027 GiB |
| Repeat upsert | 1.938 → 1.933 | 4.890 → 5.146 → 4.989 | 259.88 MiB | 106.96 MiB |

| Pass | Adjusted RssAnon peak / retained | Adjusted RssFile peak / retained |
|---|---:|---:|
| First upsert | 311.39 / 110.04 MiB | 2989.11 / 2989.18 MiB |
| Repeat upsert | 220.13 / 67.03 MiB | 39.74 / 39.93 MiB |

Component maxima need not occur at the same instant; do not sum them to derive a
total peak. Anonymous RSS is not exclusively live request allocations.

The cgroup memory peak was 5.523 GiB in the first pass and 5.525 GiB in the repeat,
below the 8 GiB limit. During the repeat interval, idle control cgroup usage fell
by 328.96 MiB and Test C usage fell by 159.54 MiB despite its RSS increase. Therefore
cgroup and RSS must not be conflated; the experiment does not attribute that
cgroup reduction to a specific reclamation trigger or claim a capacity limit.

## What grew

| Evidence | After first pass | After repeat pass |
|---|---:|---:|
| Raw-vector cache bytes | 3,072,000,000 | unchanged |
| Existing scalar-field cache bytes | 1,052,001,645, unchanged | unchanged |
| Growing entity-size gauge | 41,280,000 bytes | 82,720,000 bytes |
| Live growing entity count | 10,000 | 10,000 |
| Live sealed entity count | 990,000 | 990,000 |
| deleteBuffer rows / bytes | 10,000 / 1,200,000 | 20,000 / 2,400,000 |
| Cache `other/disk` bytes | 49,344,000 | 98,688,000 |

These gauges are distinct views with potentially overlapping accounting. They are
not additive RSS components. A repeated update can leave additional physical
versions while the live row count remains unchanged.

Final post-verification smaps corroboration:

- Raw-vector CG2 mappings: 3,001,952 KiB resident (~2.863 GiB).
- Existing VARCHAR CG3 mappings: 1,004,252 KiB resident (~0.958 GiB); this scalar
  cache was already warmed before measurement and did not grow during the passes.
- `growing_mmap` files: 80,644 KiB resident (~78.75 MiB), zero dirty KiB at this
  observation. Thus write-path data is not exclusively anonymous RSS.

The new full-row versions also copy the VARCHAR and vector values despite only
age changing. The absence of growth in the *existing scalar cache* does not mean
that these new versions consume no VARCHAR storage. The physical new-row payload
and cache-capacity gauges are not exact per-field RAM allocations.

## CPU and I/O

CPU uses process samples during active requests only. Other counter deltas below
include the active interval and its following 120-second idle observation.

| Metric | First upsert | Repeat upsert |
|---|---:|---:|
| Test C process CPU seconds | 129.47 | 149.60 |
| Idle control CPU seconds | 70.88 | 75.77 |
| Adjusted process CPU seconds | 58.59 | 73.83 |
| WAL append bytes, Test C | 55,145,102 | 55,157,801 |
| Storage filesystem read bytes | 3,072,597,126 | 15,720 |
| Storage filesystem write bytes | 29,998,174 | 45,000,273 |
| Remote scan counter | 2930.46 MiB | 0 |
| Internal cache evicted bytes | 0 | 0 |

The control appended only 2,670 and 1,602 WAL bytes respectively, with no storage
read/write counter increase. WAL bytes are logical append-counter bytes, not
network-wire bytes, replicated bytes, or final object-store space. Filesystem
counter values are not full end-to-end physical write amplification measurements.

## Latency and request results

| Pass | Requests / errors | Duration s | P50 ms | P95 ms | P99 ms | Mean ms | Stddev ms | QPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| First upsert | 10000 / 0 | 810.12 | 802.89 | 816.52 | 1042.83 | 809.94 | 76.55 | 12.34 |
| Repeat upsert | 10000 / 0 | 815.35 | 803.13 | 829.92 | 1200.97 | 815.25 | 87.87 | 12.26 |

No software-version regression claim is made: Query-only and upsert are different
operations. The much longer upsert duration was observed, but its latency cause
was not diagnosed in this memory/resource experiment.

## Environment and comparability

- Test C: `pu-ab-upsert-1030`; idle reference: `pu-ab-base-1030`.
- 1 CU standalone Tiered each: 2 CPU / 8 GiB limits, 500m CPU / 2 GiB requests.
- Image `cardinal-master-20261030-602701f`; required digest
  `sha256:3f324b13d5911236612fd5acdb0b669a6f9bf2a36c71a0a60c90ff4eaee94b1e`.
- Independent etcd/MinIO per instance. This is not production S3 or dedicated
  hardware. Standalone process metrics combine Proxy, QN, and other components.
- Same schema as the earlier A/B: INT64 PK, INT64 age, random FLOAT_VECTOR(768),
  random 1024-character VARCHAR payload; one million rows, one shard/replica.
- Same generation stream and deterministic 10,000 PK selection as A/B.
- Dataset SHA-256: `12a015d0cbcab3874fc13f00121fd22da35a2084bf773c9b22b70c7d5b6c4520`.
- Test C compacted into four sealed segments (248550, 259450, 251530, 240470 rows),
  whereas the prior A/B arms had five. The loaded sealed IDs remained unchanged
  across both measured passes. No background index jobs were active at the
  measurement boundaries; this is not a guarantee that every background subsystem
  was idle throughout the approximately 31-minute measurement period.
- Cardinal Tiered vector index. Scalar/index warmup enabled, raw vector warmup
  disabled. Auto compaction remains enabled without overrides.

## Procedure

The preparation smoke test verifies that a request containing only PK and age,
with `partial_update=True`, changes age while preserving vector and payload. This
uses a separate disposable collection, not the benchmark dataset.

After ingestion, flush/index/compaction and stable loading, replace Test C's Pod
to remove preparation heap and local raw-vector caches. Leave the idle reference
untouched. Require Test C's original compacted segment layout, complete non-vector
warmup, and zero raw-vector cache on both arms before measurement.

Run 10,000 one-PK partial upserts at concurrency 10, setting age=101, followed by
120 seconds idle. Repeat the same PKs setting age=102, followed by another 120
seconds idle. No client-side full-row Query or merge. No manual flush/compaction
inside these windows; naturally triggered maintenance is included and recorded.

Sample server process RSS/RssAnon/RssFile, cgroup memory, CPU time/throttling and
process I/O every 100 ms. Baselines and retained values are 15-second medians.
Adjusted changes subtract simultaneous changes in the idle control. The adjusted
peak pairs samples by timestamp rather than subtracting unrelated peak times.

Only after measurement, verify all target ages, unchanged field fingerprints,
untouched samples, and one million logical rows. Exclude these reads from metrics.

## Interpretation boundaries

- Compare descriptively with the prior Query-only A/B; do not label the entire
  difference as a precisely isolated write cost. Layout, time, and scheduling differ.
- The second upsert pass may retrieve newly written rows, not the original sealed
  rows used by Query-only hot repeats. It is a repeated-write scenario, not merely
  another cold-cache replicate.
- RSS is not exclusively live allocations; file-backed residency may be reclaimed.
  A 120-second retained value is not permanently required memory.
- Process I/O is not object-store traffic; report those counters separately from
  storage read/write and WAL payload metrics. No network-wire-byte claim is made.
- Current partial upsert still has the real write path. This is not a measurement
  of the proposed in-place-update implementation or its net savings.

## Verification and artifact integrity

- Successful partial-update smoke test before dataset preparation, including
  unchanged vector/VARCHAR, ANN search, deletion, and disposable collection cleanup.
- Requested compaction: Completed, three completed plans, zero executing/timeouts.
  Waited for indexes, segment replacement, and 120 seconds of stable loaded state.
- Actual running image digest and 1 CU limits verified; auto compaction stayed true.
- Per-pass server Upsert success-counter increase exactly 10,000; idle control
  received no Query/Upsert workload. No measured Pod replacement or container restart.
- Final correctness: all 10,000 target ages, 40 full-field fingerprints including
  untouched-row samples, and a Strong count of 1,000,000 logical rows.
- Maximum observed sampling gap during requests was 120.2 ms. Sub-sampling-interval
  peaks may be missed.
- A raw kubectl file copy ended with unexpected EOF **after** execution and
  verification completed. Original stopped samplers remained intact. Compressed
  retrieval with SHA-256 verification recovered both files; no workload was rerun:
  - Control: `4619f1ed1c780bf32da4d10ef2c5565996739d4a0a4ef86618d1a37fb1de9c1f`
  - Test C: `63fffa4b2b678c2bdef1382571a68139b129bc241601d97e9c112911b98e4355`

Artifacts: `PLAN.md`, `experiment.py`, `sample.py`, `control.py`, `analyze.py`,
`summary.json`, `results/`, both `c1-*-memory.jsonl` files, deployment/Pod metadata,
server logs, `final-verification.json`, and final smaps summaries. Shared dataset
preparation helpers are preserved in the sibling A/B experiment's `experiment.py`.

Test C and the two earlier 1 CU instances are retained for inspection. The old
4 CU instance remains absent. The temporary client is removed after evidence
collection. No Milvus product source or Git branch was modified.
