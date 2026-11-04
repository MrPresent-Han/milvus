# 点更新收益实验：旧行 Query 与真实 Partial Upsert 的资源增量

日期：2026-10-30。状态：已完成；本文汇总两阶段实验，不是尚未实现的原生点更新方案的性能验收。

关联设计：[实时 Predicate Delete 与 PK 点更新：Phase 1 设计](20261030-sync-predicate-delete-and-point-update-design-zh.md)。

## 1. 核心结论

**本次负载下，主要的初次驻留增量来自读取完整旧行加载的 raw vector，而不是后续写入。点更新希望避免的，不只是一次 Query 的计算成本，更是为了修改小字段而读取、缓存未修改宽列的成本。**

- 独立 A/B 的 Query-only 实验：冷读后，扣除空闲对照变化的驻留 RSS 增量为 **2.894–2.904 GiB**。
- 独立 C 的真实 partial-upsert：首次驻留 RSS 增量为 **3.027 GiB**；重复更新同一批 PK，再增加 **106.96 MiB**。
- 10,000 个随机 PK 只占总行数的 1%，但 Query-only 的首次读取触及 B 的全部 **971 个 raw-vector CacheCells**，加载 3,072,000,000 bytes 的 raw vectors。返回给客户端的仍只是请求选中的行，不是整列。
- B 的 raw-vector 文件映射驻留约 **2.863 GiB**。原 VARCHAR 映射约 **0.958 GiB**，但测试前已经 warmup，不能将这部分计入本轮 Query 的新增成本。

这支持“稀疏随机旧行读取带来读取与缓存驻留放大”的判断。**不支持“insert 没有内存成本”或“原生点更新已经证明净节省 3 GiB”**：后续写入仍累积版本、删除记录等状态；拟议的 UpdatesRecord 成本尚未测量。

这里的“驻留”统一指停止请求 120 秒后的观测值，不是永久不可回收内存。主要增量是 file-backed RSS，而不是全部来自匿名堆分配。

## 2. 环境与数据

| 项目 | 配置 |
| --- | --- |
| A：空闲对照 | `pu-ab-base-1030` |
| B：Query-only | `pu-ab-query-1030` |
| C：真实 partial-upsert | `pu-ab-upsert-1030` |
| 实例规格 | 三个独立 1 CU standalone；各自 limits 2 CPU / 8 GiB，requests 500m CPU / 2 GiB |
| 镜像 | `cardinal-master-20261030-602701f` |
| 镜像 digest | `sha256:3f324b13d5911236612fd5acdb0b669a6f9bf2a36c71a0a60c90ff4eaee94b1e` |
| 索引 | AUTOINDEX 解析为 `CARDINAL_TIERED` |
| 存储／缓存 | 各实例独立 etcd、MinIO；Vortex 持久化文件，raw local format；scalar/index warmup 开启，raw vector warmup 关闭 |
| 数据 | 1,000,000 行；单 shard、单 replica |
| Schema | INT64 PK、INT64 age、768 维随机 FLOAT_VECTOR、1024 字符随机 VARCHAR |
| 逻辑数据量 | 约 4.112 GB，其中向量 3.072 GB；GB 为十进制，GiB 为二进制 |
| 请求 | 每次 1 PK，并发 10；相同的 10,000 个不重复随机 PK |
| Compaction | 准备阶段等待 compaction、索引与加载稳定；auto-compaction 始终开启 |

三实例使用同一数据生成序列，SHA-256：`12a015d0cbcab3874fc13f00121fd22da35a2084bf773c9b22b70c7d5b6c4520`。A/B 各有 5 个 sealed segments，C 有 4 个；数据相同不等于物理布局逐字节相同。

这是 standalone 整个 Milvus 进程的测量，不能按进程拆分 Proxy 和 QN。独立客户端的内存不计入。CPU limit 不是独占 CPU；MinIO 也不是生产 S3。

## 3. 实验方法

### 第一阶段：A/B 隔离旧行读取

A 保持 loaded、空闲；B 使用 Strong consistency、`output_fields=["*"]`，每次按一个 PK 读取完整行。**不执行 Proxy 合并和后续 insert/delete**，这是旧行读取成本的近似实验，不等同于内部 partial-upsert retrieval 的全部实现路径。

共执行两个冷／热配对轮次，每轮：60 秒 baseline → 10,000 次冷读 → 空闲 120 秒 → 同一批 PK 的 10,000 次热读 → 空闲 120 秒。两轮之间通过 Pod 替换清除前轮本地缓存并重新确认 warmup；测量窗口中不重启。共 40,000 次成功 Query。

### 第二阶段：A/C 真实 partial-upsert

C 独立加载相同数据，compaction 完成后重启 C，排除准备阶段残留；确认 raw-vector cache 为空、其他 warmup 完成。A 继续作为同期空闲对照。

客户端仅提交 PK 和 age，调用 `upsert(..., partial_update=True)`：

1. 10,000 次更新，将 age 设置为 101，空闲 120 秒。
2. 同一批 PK 再更新 10,000 次，将 age 设置为 102，空闲 120 秒。

客户端不预先 Query 或合并整行；旧行读取与整行写入由真实 partial-upsert 路径执行。测量期间不主动 flush/compaction；自然后台维护不被关闭。测量后才执行数据校验，避免校验读取污染窗口。

### 统一计量口径

每 100 ms 采样进程 RSS、RssAnon、RssFile 和 cgroup memory。baseline 为请求前最后 15 秒中位数，驻留值为 120 秒空闲期最后 15 秒中位数。

```text
调整后驻留增量 = (实验组 post - 实验组 before) - (同期空闲 A post - A before)
```

峰值按时间对齐实验组和 A 的样本，计算双方相对 baseline 的差，再取最大值。不是两个不同时刻峰值相减。采样峰值不是纯临时分配峰值：它也包括运行期间累积的缓存和版本；独立分量峰值不可直接相加。

## 4. 内存结果

### 4.1 对照调整后的增量

| 实验／pass | 请求数 | RSS 峰值增量 | 120 秒后驻留 RSS 增量 |
| --- | ---: | ---: | ---: |
| B：冷读第 1 轮 | 10,000 | 2.942 GiB | 2.904 GiB |
| B：冷读第 2 轮 | 10,000 | 2.935 GiB | 2.894 GiB |
| B：热读第 1 轮 | 10,000 | 9.69 MiB | −14.21 MiB |
| B：热读第 2 轮 | 10,000 | 31.74 MiB | −0.36 MiB |
| C：首次 partial-upsert | 10,000 | 3.213 GiB | 3.027 GiB |
| C：同批 PK 再次 partial-upsert | 10,000 | 259.88 MiB | 106.96 MiB |

热读的负增量是相对空闲对照的漂移／回收，不表示 Query 节省内存。C 第二次的增量以第一次完成后的状态为起点；两次后的累计调整驻留增量约 **3.131 GiB**。

### 4.2 C 的绝对 RSS 与内存组成

| Pass | 请求前 RSS | 请求期间峰值 RSS | 120 秒后 RSS | 调整后 RssAnon 驻留增量 | 调整后 RssFile 驻留增量 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 首次更新 | 1.857 GiB | 5.070 GiB | 4.890 GiB | 110.04 MiB | 2989.18 MiB |
| 再次更新 | 4.890 GiB | 5.146 GiB | 4.989 GiB | 67.03 MiB | 39.93 MiB |

C 的 cgroup memory 峰值分别为 5.523、5.525 GiB，未达到 8 GiB limit。RSS 与 cgroup memory 不是同一口径，不能相加；第二次更新期间 cgroup usage 下降、RSS 上升，不能仅凭 RSS 判断总物理内存新增需求。

## 5. 为什么少量点读取会加载大量缓存

第一阶段冷读的 CacheCell miss bytes 为 **3,072,000,000**，filesystem read counter 约 **3,072,594,000 bytes**；热读这两项均不再增长。对照 A 始终没有 raw-vector cache 增量。

10,000 行所对应的逻辑向量仅为 30,720,000 bytes，但它们随机分散，触及所有 971 个 raw-vector CacheCells。按本次向量逻辑字节量比较，首次加载量约为所选行向量总量的 **100 倍**。这是本次布局和访问分布的结果，不是系统固定的放大倍数，也不是网络流量或磁盘编码大小的比值。

后续 smaps 核对：

| 驻留内容 | B Query-only | C partial-upsert 后 | 如何解释 |
| --- | ---: | ---: | --- |
| 原 sealed raw vector CG2 映射 | 约 2.863 GiB | 约 2.863 GiB | 首次旧行读取的主要新增驻留 |
| 原 VARCHAR CG3 映射 | 约 0.958 GiB | 约 0.958 GiB | 测量前已 warmed，不计为首次读取增量 |
| C 的 growing_mmap 映射 | 不适用 | 约 78.75 MiB | 新写入状态也可以是 file-backed，而非全部匿名内存 |

这些 smaps 数据采于最终核查时，支持组成判断，但不是 120 秒边界同一时刻的严格拆账。原 VARCHAR cache 没增长，不等于写入没有 VARCHAR 成本：新的完整行版本仍带有未修改的 VARCHAR 和 vector。

## 6. 写入新增状态、CPU 与 I/O

| C 指标 | 首次 10,000 次后 | 再次 10,000 次后 |
| --- | ---: | ---: |
| Raw-vector cache | 3,072,000,000 bytes | 不变 |
| 原 scalar-field cache | 1,052,001,645 bytes，较 baseline 不变 | 不变 |
| Growing entity-size gauge | 41,280,000 bytes | 82,720,000 bytes |
| Live growing rows | 10,000 | 10,000 |
| Live sealed rows | 990,000 | 990,000 |
| deleteBuffer rows / bytes | 10,000 / 1,200,000 | 20,000 / 2,400,000 |
| 每 pass 调整后 active CPU | 58.59 CPU-sec | 73.83 CPU-sec |
| 每 pass WAL append counter 增量 | 55,145,102 bytes | 55,157,801 bytes |
| 每 pass filesystem read counter 增量 | 3,072,597,126 bytes | 15,720 bytes |
| 每 pass filesystem write counter 增量 | 29,998,174 bytes | 45,000,273 bytes |

CPU 仅统计 active requests；I/O counters 包含 active 加其后 120 秒空闲期。上述 gauges 可能存在重叠，不能相加当成独立的 RSS 分量。WAL counter 也不是最终持久化空间、复制字节数或全链路写放大。

## 7. 延迟与正确性校验

| 实验／pass | P50 ms | P95 ms | P99 ms | Mean ms | Stddev ms | QPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B 冷读 1 | 7.526 | 63.851 | 110.380 | 15.468 | 20.891 | 645.5 |
| B 冷读 2 | 7.251 | 62.116 | 111.387 | 15.071 | 20.679 | 662.5 |
| B 热读 1 | 6.761 | 12.887 | 23.638 | 7.651 | 5.805 | 1301.9 |
| B 热读 2 | 6.412 | 11.293 | 28.024 | 7.311 | 5.893 | 1362.8 |
| C 首次更新 | 802.89 | 816.52 | 1042.83 | 809.94 | 76.55 | 12.34 |
| C 再次更新 | 803.13 | 829.92 | 1200.97 | 815.25 | 87.87 | 12.26 |

B 各 pass 的 stddev 均超过 mean 的 30%，不能视作稳定吞吐基准。C 两次各耗时约 810／815 秒，延迟原因未在本实验中定位。操作不同，不将 B/C 的延迟差异称为版本性能回归。

- B 共 40,000 次 Query、C 共 20,000 次 upsert，均零请求错误；测量期间 A 无用户 Query/Upsert。
- C 校验全部 10,000 个 target age 均为 102；40 个样本的未修改字段 fingerprints 匹配，包含未更新行；Strong count 仍为 1,000,000。
- 测量期间没有 Pod 重启；sealed segment IDs 保持稳定；没有 internal cache eviction。auto-compaction 全程开启。
- B 的最大采样间隔约 102 ms，C 约 120.2 ms，更短的内存尖峰可能遗漏。

## 8. 对设计的支持与结论边界

采用的实验证据是：**不读取完整旧行，可以避免仅因小字段更新而拉入未修改 raw vector 的这类额外工作集。** 该结果支持设计中的独立字段 delta 路径，且不依赖是否最终采用 Consolidation。

必须保留以下限制：

1. C 首次驻留比 B 冷读多约 126–136 MiB，只是描述性差异。C 有 4 个 sealed segments、B 有 5 个；一次 Query pass 约 15 秒，C 约 810 秒，后台运行时间与调度均不同，不能把差值精确拆成 insert 成本。
2. C 的第二次更新可能读取新写入行，而 B 热读是原 sealed rows，不能作为完全等价的热缓存对照。第二次新增驻留仍证明真实写入的成本没有消失。
3. 没有实现原生点更新对照组；UpdatesRecord、payload 生命周期、执行器及其长期成本均未计入，不能声称净节省已验证。
4. 干净 file-backed 页面可能被回收；未制造内存压力，不据此推导最低 CU、永久内存需求或 OOM 边界。
5. 未测长期重复更新、最终 full rewrite、索引重建等后台总成本，也未证明其他 schema、访问分布和 local format 的收益相同。

## 9. 证据归档

以下原始英文报告和汇总数据随本文归档，避免设计引用只依赖本机临时目录：

- [A/B 原始报告](experiments/20261030-partial-upsert-memory/query-ab-report.md)／[汇总 JSON](experiments/20261030-partial-upsert-memory/query-ab-summary.json)。
- [C 原始报告](experiments/20261030-partial-upsert-memory/partial-upsert-c-report.md)／[汇总 JSON](experiments/20261030-partial-upsert-memory/partial-upsert-c-summary.json)。

更大体量的采样、运行脚本、Pod 元数据、日志与最终校验保存在实验机器的以下目录，**未上传到仓库**：

```text
/Users/zilliz/Documents/project/pu-query-ab-1cu-20261030/
/Users/zilliz/Documents/project/pu-upsert-c-1cu-20261030/
```

C 原始采样文件 SHA-256：

```text
control: 4619f1ed1c780bf32da4d10ef2c5565996739d4a0a4ef86618d1a37fb1de9c1f
test-c:  63fffa4b2b678c2bdef1382571a68139b129bc241601d97e9c112911b98e4355
```

两阶段都曾在测量完成后的 artifact 传输遇到中断，重新取回原始文件后校验通过；没有为替换结果重跑负载。测试完成时 A/B/C 三个 1 CU 实例保留，临时客户端已删除，旧 4 CU 实例已释放。
