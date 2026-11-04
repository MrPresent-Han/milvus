> 原始实验报告归档，2026-10-16。来源：`test_results/bulk_delete_growing_caseA_20261016/report.md`。下文保持原报告内容；完整 profiles 和监控采样未随本副本复制，QTP task IDs 与测量限制见正文。

# Growing-Insert + Bulk-Delete Test on the Existing AWS-UAT 12-CU Instance

## Setup and scope

- Instance: `in01-61b8bf666332c3f` (12-CU cluster, AWS-UAT), image `harbor-us-vdc.zilliz.cc/milvus/milvus:3.0.3-20261013-1787fa3f6-5c759fb`.
- Collection: `bulk_delete_growing_100m_12cu_20261016`; 100,000,000 original rows, 32-dimensional vectors, AUTOINDEX, fully flushed/indexed/loaded before the workload.
- Concurrent workload: approximately 550 offered ANN Search QPS (`nq=1`, `topK=10`, 2-second client timeout) and one successful 100-row Insert batch per second. New primary keys start at 100000002 and advance by 5, so all new rows satisfy `id_pk % 5 == 2`.
- Single, unretried Delete request: `id_pk % 5 == 1`; it removed exactly 20,000,000 original rows. Search and Insert remained active before, during, and after the Delete request.
- This is an intentionally high-load dynamic case, not an isolated Delete-only benchmark. Before Delete, StreamingNode was already near its 8-core CPU limit. Automatic growing-segment sealing and later compaction occurred during the observation period.

## Outcome

The server processed the one Delete RPC successfully in 129.622 seconds. Its reported `deletedCount` was 20,000,000. The 40-minute Insert task finished with 229,500 successful rows in 2,295 batches and zero failed batches. Final validation found zero rows matching the Delete predicate, exactly 80,000,000 remaining original rows, and exactly 229,500 new rows, all with `id_pk % 5 == 2` (80,229,500 rows in total).

| Proxy-observed Search metric | Immediate pre-Delete, 05:02:30–05:04:15 UTC | Delete Window, 05:04:30–05:06:45 UTC | Post-Delete steady sample, 05:12–05:14 UTC |
| --- | ---: | ---: | ---: |
| Mean observed completion QPS | 550.0 | 453.1 (-17.6%) | 538.4 |
| Minimum sampled QPS | 549.7 | 350.8 (-36.2% versus pre-Delete mean) | 523.8 |
| Mean average latency | 126.2 ms | 176.0 ms (+39.5%) | 181.2 ms |
| Mean histogram P99 | 253.1 ms | 548.2 ms (+116.6%) | 295.2 ms |
| Maximum sampled histogram P99 | 253.7 ms | 1,342.1 ms | 436.7 ms |
| Mean failed Search RPS | 0 | 9.37 | 0 |
| Successful Insert RPCs/s | 1.00 | 0.96 | 0.99 |
| WAL-inserted rows/s | 100.0 | 96.0 | 99.3 |

The 30-second rate and histogram samples were evaluated every 15 seconds; their sample means are not request-weighted whole-window aggregates. The immediate pre-Delete window is the fairer comparison than an earlier baseline: while Insert continued, the growing segment increased from roughly 146,000 to 157,000 rows even before Delete began. The Delete Window is bounded by the Proxy request counter transitions; its 15-second scrape resolution slightly exceeds the measured 129.622-second RPC latency.

After Insert naturally stopped, a steady 05:18:30–05:20:00 UTC sample observed 550.0 Search QPS, 45.3-ms mean average latency, 111.9-ms mean P99, and zero failed requests. The growing segment remained present but received no further rows. This is a temporal association, not a controlled attribution to Insert cessation alone: segment state and compaction also changed.

## Component resources and recovery

| Component | Immediate pre-Delete mean | Delete Window peak | Observation |
| --- | ---: | ---: | --- |
| StreamingNode CPU | 7.98 / 8 cores | 7.66 cores | Already saturated before Delete; measured CPU decreased as Search throughput fell. |
| StreamingNode memory working set | 4.70 GiB | 9.51 GiB (+4.81 GiB, +102%) | Reached 4.84 GiB by 05:08:15 UTC, about 90 seconds after RPC success; later Insert/segment activity raised it again. |
| Proxy CPU | 0.42 / 4 cores | 0.42 cores | No CPU-limit pressure. |
| Proxy memory working set | 0.53 GiB | 1.41 GiB (+0.88 GiB, +166%) | Returned near 0.53 GiB by 05:07:00 UTC. |
| QueryNode CPU (three pods) | 6.52 / 6.73 / 6.06 of 8 cores | 6.40 / 6.70 / 6.55 cores | In-window CPU did not rise with request work because effective Search throughput fell. |
| QueryNode memory working set (three pods) | 10.61 / 13.27 / 12.75 GiB | 11.68 / 14.15 / 13.27 GiB | Transient increases, below each pod's 32-GiB limit. |

The Delete Window's StreamingNode delete buffer reached about 732 MB. StreamingNode retained one growing segment throughout, but its growing-row gauge dropped from roughly 166,000 to roughly 2,000 as that segment sealed. QueryNode sealed-segment count changed from six to seven, then later decreased during compaction. These concurrent transitions constrain causal attribution of the post-Delete latency plateau. Search throughput and failure rate substantially recovered while Insert continued, but the observed 05:12–05:14 average Search latency was still above its immediate pre-Delete level; it then fell below that level after Insert stopped.

All Milvus pods remained Ready with zero restarts. No OOM was observed.

## Profiles

CPU, Go heap, and goroutine profiles for Proxy, MixCoord, three QueryNodes, and StreamingNode were saved and gzip-validated for baseline, Delete, and post-Delete phases. Profiles are process-wide and include any pre-existing collection on this shared instance. Native C++ symbolization is incomplete in the local `go tool pprof` environment; these profiles should not be treated as a complete accounting of container CPU.

- Proxy Go in-use heap: 38.9 MB before, 152.5 MB during, 42.0 MB after; the Delete-time snapshot attributed about 109 MB flat to protobuf `growInt64Slice`.
- StreamingNode Go in-use heap: 1,258 MB before, 2,655 MB during, 1,384 MB after. The before-to-during diff includes about +390 MB flat in `deleteNode.addDeleteData`, +181 MB in `NewInt64PrimaryKey`, +172 MB in Arrow's Go allocator, +144 MB in the PK-version-index update path, and +119 MB each in delete forwarding and `DeleteData.Append`.
- One sampled QueryNode Go in-use heap: 216 MB before, 290 MB during, 216 MB after. Its 5-second CPU snapshot remained dominated by ANN Search/vector-index functions.
- Go heap figures are not container working-set figures; the latter include native allocations and other memory.

## Evidence and limitations

- QTP task IDs: preparation `16409`, readiness `16413`, Search `16418`, Insert `16416`, Delete `16419`, post-Delete assertions `16420`, and final assertions `16421`.
- The QTP Delete case reported success, `deletedCount=20000000`, and `failCount=0`; the Insert task reported `numEntries=229500` and `failCount=0`; final assertions passed 4/4.
- Prometheus window summaries: `pre_immediate_metrics.json`, `delete_window_metrics.json`, `post_steady_metrics.json`, and `post_insert_steady_metrics.json`; additional phase summaries and raw profiles are in this directory.
- Search client timeout was raised from 800 ms to 2 seconds after an earlier Search-only-plus-Insert probe already produced timeout errors before Delete. The Delete comparison uses only the 2-second Search run and its immediate pre-Delete interval.
- The test intentionally did not freeze growing-segment size, segment sealing, or compaction. It measures the combined dynamic system behavior under sustained Insert and Search; it does not isolate the marginal cost of Delete from every concurrent background transition.

The Search task was intentionally terminated after the final assertions and post-Insert steady observation, so the QTP runner does not have a natural 40-minute terminal Search result. The measured Search windows above are from the instance's Proxy metrics, and no benchmark workload was left running.
