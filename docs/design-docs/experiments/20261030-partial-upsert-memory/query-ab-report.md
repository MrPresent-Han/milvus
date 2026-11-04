# Two-instance 1 CU Query memory A/B

Status: completed and validated, 2026-10-30. Two paired cold/hot rounds,
40,000 successful treatment queries, zero control queries during measurement.

## Findings

The two rounds isolated 2.89–2.90 GiB of retained process RSS after the first
full-row Query pass, after subtracting the idle control's concurrent change.
This is observed residency 120 seconds after requests stopped, not permanently
allocated or unreclaimable memory. The hot repeat did not accumulate another
similar increment. Adjusted cold peak RSS increments were 2.94 GiB in both rounds.

The result supports an avoidable raw-vector read/cache working-set cost for this
specific workload. It does not establish the net memory savings of a complete
in-place-update implementation, whose UpdatesRecord and execution costs are absent.

The first pass loaded all 971 raw-vector CacheCells on the treatment, representing
3,072,000,000 bytes of logical raw vectors. The control loaded zero vector cells.
The 10,000 selected rows are only 1% of rows, but their random distribution touched
every vector CacheCell; loading is not limited to the selected rows' payload bytes.
This does not mean 1,000,000 vectors were returned to the client.

Scalar data was already warmed in both arms, so its existing residency is not
credited as an avoidable Query increment. This differs from comparing against an
incompletely warmed baseline.

## Memory results

All absolute RSS values below are GiB. Post means the final 15-second median of
the 120-second idle interval. Adjusted increments subtract the control's change.

| Round / pass | Control before → post | Treatment before → peak → post | Adjusted peak increment | Adjusted retained increment |
|---|---:|---:|---:|---:|
| 1 / cold | 1.910 → 1.909 | 1.903 → 4.849 → 4.805 | +2.942 GiB | +2.904 GiB |
| 2 / cold | 1.921 → 1.905 | 1.886 → 4.822 → 4.765 | +2.935 GiB | +2.894 GiB |
| 1 / hot | 1.909 → 1.916 | 4.805 → 4.816 → 4.799 | +9.69 MiB | −14.21 MiB |
| 2 / hot | 1.905 → 1.910 | 4.765 → 4.795 → 4.770 | +31.74 MiB | −0.36 MiB |

Negative hot retained increments represent idle drift/reclamation relative to the
control, not a claim that Query saves memory.

| Cold round | Adjusted RssFile peak / retained | Adjusted RssAnon peak / retained | Treatment cgroup peak |
|---|---:|---:|---:|
| 1 | 2933.92 / 2933.98 MiB | 78.42 / 39.49 MiB | 5.325 GiB |
| 2 | 2935.23 / 2935.30 MiB | 70.45 / 28.23 MiB | 5.284 GiB |

Each component's peak is computed independently; do not add component maxima
as though they occurred at the same instant. Anonymous RSS is process residency,
not a direct measurement of exclusively live per-request allocations.

Final smaps attribution corroborates the cache metrics: treatment CG2 local raw
vector mappings have 3,001,904 KiB resident, while the control has no such resident
mapping. CG3 string mappings are approximately 1,004,250 KiB resident in both arms.
These local_chunk mappings have zero Private_Dirty/Shared_Dirty at the final check.
Reclaim behavior under memory pressure was not tested; this is not a minimum-RAM
capacity benchmark or proof that the observed residency is permanently required.

## I/O and validity

- Each cold pass: raw-vector cache miss bytes = 3,072,000,000; filesystem read
  counter approximately 3,072,594,000 bytes; remote scan counter = 2929.6875 MiB.
- Each hot pass: zero increases in those three counters. Control: zero throughout.
- No internal cache eviction in any measured pass. Non-vector cache bytes remained
  unchanged. The scalar payload was already warmed before all measurement windows.
- Requested compactions completed on both arms: three plans each, zero executing
  or timed-out plans. Indexing and loaded layouts stabilized before measurement.
- Segment identities and row totals remained unchanged through both rounds.
- No measured Pod replacement or container restart; auto compaction remained true
  with FileSource configuration, confirmed again by the final read-only audit.
- Maximum sample gap during requests was approximately 102 ms.

## Client latency (context only)

| Round / pass | Requests / errors | P50 ms | P95 ms | P99 ms | Mean ms | Stddev ms | QPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 / cold | 10000 / 0 | 7.526 | 63.851 | 110.380 | 15.468 | 20.891 | 645.5 |
| 2 / cold | 10000 / 0 | 7.251 | 62.116 | 111.387 | 15.071 | 20.679 | 662.5 |
| 1 / hot | 10000 / 0 | 6.761 | 12.887 | 23.638 | 7.651 | 5.805 | 1301.9 |
| 2 / hot | 10000 / 0 | 6.412 | 11.293 | 28.024 | 7.311 | 5.893 | 1362.8 |

All passes have stddev above 30% of mean; tail latency is variable. Cold passes
mix cache misses and later hits. These figures describe the workload, not a stable
throughput benchmark or a software-version performance regression comparison.

## Question

Compare a loaded but idle instance (`base`) with an independently loaded instance
running full-row single-PK queries (`test-query`). This measures the incremental
memory residency of the Query stage, not actual partial-upsert or delta-update savings.

## Environment

- `pu-ab-base-1030` and `pu-ab-query-1030`, independently provisioned with
  milvus-dev-cli in `vc-chun-han / chaos-testing`.
- Each: 1 CU standalone Tiered; limits 2 CPU / 8 GiB; requests 500m CPU / 2 GiB.
- Identical image: `cardinal-master-20261030-602701f`, digest
  `sha256:3f324b13d5911236612fd5acdb0b669a6f9bf2a36c71a0a60c90ff4eaee94b1e`.
- Independent etcd and MinIO. MinIO is not production S3; underlying nodes may
  be shared with unrelated workloads. CPU limits are not dedicated CPU reservations.
- AUTOINDEX resolved through CARDINAL_TIERED. Raw vector warmup disabled;
  scalar/index warmup enabled. Auto compaction stays enabled throughout.
- Separate client Pod; its RSS is not included in server measurements.
- Proxy and QN share the standalone process. RSS is measured for that process;
  it cannot be attributed independently to Proxy versus QN by process metrics.

## Dataset and procedure

Identical batches are sent to both instances: 1,000,000 sequential INT64 PKs,
INT64 age, random float32 vectors of dimension 768, and 1,024-character random
VARCHAR payloads. A generation-stream SHA-256 is recorded in `dataset.json`.
Logical user data is approximately 4.112 GB, of which 3.072 GB is raw vectors.

After flush, request compaction on both collections and wait for completion.
Wait for indexes and loaded segments/cache totals to remain stable for 120 seconds.
Run the same preparation correctness smoke checks on both arms, then restart
both Milvus Pods to remove those smoke-read caches and ingest/build heap residue.
Before measurement, verify complete cold loading against the prepared layout;
raw-vector cache must be empty. Auto compaction is never disabled.

Each paired round has a 60-second baseline, 10,000 distinct random single-PK
full-row Query requests at concurrency 10 on the treatment only, 120 seconds
idle, the identical hot-read sequence, and another 120 seconds idle. Strong
consistency and `output_fields=["*"]` are used. The control receives no user Query.
Both arms receive metadata checks and approximately one metrics scrape per second.

Sample the Milvus process's RSS, RssAnon, RssFile, and cgroup usage at 100 ms.
Use the median of the last 15 seconds before the phase as its baseline, and the
last 15 seconds of the 120-second idle interval as its post-phase value.
Peak means the maximum sampled value while requests were running, not an exact
sub-100-ms allocation high-water mark. Cgroup usage and process RSS are different
measurements and are not added together.

Report absolute RSS and raw changes for each arm. The adjusted retained change is:

`(query_post - query_before) - (base_post - base_before)`

For the adjusted peak, align each treatment sample with the nearest control sample
in time, subtract each arm's baseline, and take the maximum difference. This is
not the difference between two unrelated peak timestamps.

## Validity and interpretation

- Require unchanged segment identities/row counts through measurement; no restarts,
  no incomplete index jobs, and no non-vector cache-size changes. Record exclusions.
- Query counts must increase by 10,000 per treatment pass and zero on the control.
- File cache size is not resident RAM. Distinguish file-backed residency from
  anonymous allocations; clean mmap pages can be reclaimed under memory pressure.
- No forced GC, allocator purge, host cache clearing, or synthetic memory pressure.
- Cold means empty local raw-vector CacheCells after Pod replacement. It does
  not mean cold MinIO server caches or cold caches on every shared host.
- Two arms reduce baseline/idle drift confounding, but independent process startup,
  physical segment layouts, and shared-host scheduling can still differ.
  Both arms have five sealed segments and the same million logical rows, but
  physical segment sizes differ; raw-vector CacheCell counts are 977 (control)
  and 971 (treatment). The experiment is not a byte-identical storage-layout test.
- Full-row SDK Query includes external result serialization, whereas the internal
  partial-upsert retrieval path differs. Proxy merge and subsequent insert/delete
  are deliberately excluded.
- No software-version regression comparison or real in-place-update arm is present.
- Round 1 initially encountered an HTTP 404 from the management configuration
  endpoint during Pod startup, before any Query was issued. After the endpoint
  became ready, measurement was restarted on the same Pods, with empty vector
  cache and stable warmup verified again. The failed preflight is preserved in
  `ab1-startup-failure.log` and excluded from measurement windows.
- The first round-2 artifact transfer was truncated after measurement completed.
  Re-fetching the untouched runner artifacts succeeded. Both rounds then passed
  the same analysis assertions; no benchmark was rerun to replace a poor result.

## Artifacts

`PLAN.md`, `experiment.py`, `control.py`, and `analyze.py` define the protocol.
Runtime data is in `results/`, `abN-{base,query}-memory.jsonl`, deployment/Pod
metadata, and `summary.json`. `final-verification.json` and `final-*-smaps.json`
record the final read-only audit. The prior 4 CU deployment was released at the user's
request; its local results remain in the sibling `pu-query-memory-20261030` directory.

The two 1 CU instances are retained for inspection. The temporary client Pod is
removed after evidence collection. No real partial upsert/update/delete workload
was run, and no product source files or Git branches were changed.
