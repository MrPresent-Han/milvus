# 异步 Predicate Delete 与实时 PK 点更新：正确性与逻辑设计

日期：2026-10-28

状态：设计整理稿。本文区分已确定的设计选择、仍待讨论的逻辑问题，以及已定方向下尚需完成的实现对接；不表示功能已经实现或通过验证。

## 1. 文档范围

本文梳理当前选定的混合方案：**Predicate Delete 异步执行，PK 点更新实时执行**。

本轮只讨论功能语义、执行逻辑、数据完整性及实现可行性。不展开资源管控、调度阈值、稳定性专项、性能优化、任务状态查询、进度展示或跨副本完成状态。查询语义只保留判断“会不会读错数据”所必需的部分，不展开 A/H/TSafe 等进度协议。

文档来源：

- [当前英文设计](20261027-predicate-delete-and-update-deltas-async.md)：本文的主要依据。文件名保留了 `async`，但内容已从完全异步方案收窄为当前混合方案。
- [冻结的实时设计](20261016-predicate-delete-and-update-deltas.md)：保留早期完整实时方案及实现讨论，不随本文修改。

本文不是逐段翻译，也不重新打开已确定的选择。英文设计中的资源、安全预算和运行状态议题不纳入本轮待决清单。

后续讨论在本文持续维护；与英文稿尚未同步的部分，以本文明确标记的已定项为准。

### 1.1 三项核心收益

1. **降低 Predicate Delete 的前台写入成本。** 支持的 predicate 直接进入 WAL，不先查询全部匹配 PK，再传输、写入大量 PK deletes。
2. **降低点更新的前台写入成本。** 只提交 PK 与被修改字段的常量值，不读取、补齐整行，也不转换为整行 insert＋delete。
3. **降低局部更新的后台维护成本。** Delta Consolidation 保持行布局，重写受影响的列，复用未修改的数据与索引，避免仅因 scalar 更新而重建向量数据及 ANN index。

前两项针对前台，第三项针对后台。这些是设计目标，不是已测得的性能结论。

### 1.2 行业依据与采用边界

先区分成熟系统已有的原则与本方案自己的组合：

| 成熟实践 | 本方案采用的原则 | 不直接照搬的部分 |
| --- | --- | --- |
| [PostgreSQL UPDATE](https://www.postgresql.org/docs/current/sql-update.html) 修改所有满足条件的行，并保留未指定列的值 | 同一个非唯一键值可以匹配多行，UPDATE 的目标由明确的条件决定 | 不引入 PostgreSQL 的事务隔离、唯一性或完整 SQL UPDATE 能力 |
| [ClickHouse 论文 §3.4](https://clickhouse.com/uploads/Click_House_Schulze_1_c12ecfaed4.pdf) 描述 mutation 期间查询可能同时读到已修改和未修改的 parts | 允许异步 predicate delete 在不同 segments 上分批体现 | 不据此放宽单个任务结果的完整性，也不照搬其整 part 重写方式 |
| [Kudu 后台维护](https://kudu.apache.org/docs/background_tasks.html#compacting-on-disk-data) 区分 delta 维护与 RowSet 合并 | 将 Delta Consolidation 与 segment 合并分开 | 不引入 Kudu 的 UNDO、历史保留窗口或唯一键定位机制 |
| [Iceberg deletion vectors](https://iceberg.apache.org/spec/#deletion-vectors) 用绑定数据文件的行位置集合表达删除 | 保存删除结果而不立即物理移除行 | 不照搬文件格式或快照协议；本方案绑定 Milvus segment 的行布局 |

这些原则有成熟依据；**实时点更新＋异步 predicate delete＋Milvus 非唯一 PK 的组合不是现成的行业标准协议**。其具体正确性仍需按本文约束实现并验证。

## 2. 已定：Phase 1 功能边界

| 操作 | 输入范围 | 执行方式 |
| --- | --- | --- |
| PK 点更新 | 单个 PK 等值条件，以及一个或多个字段的常量赋值 | WAL 后经实时链路应用到 QN；无需等待 Consolidation |
| Predicate Delete | 本功能支持的 typed predicate | 数据侧异步执行；QN 不执行尚未物化的 predicate delete |
| 现有 PK Delete | 显式 PK，沿用既有时间资格与删除语义 | 保留现有实时处理路径 |

以下语法仅用于表达语义，不代表已经确定新的 SQL/API 接口：

```text
UPDATE SET age=18 WHERE pk=100
DELETE WHERE age < 18
```

### 2.1 点更新的输入限制

- UPDATE 只开放 PK 等值条件，不开放任意表达式、范围条件或附加字段过滤。
- 更新 1,000 个 PK，逻辑上是 1,000 次点更新；不增加跨 PK 原子事务保证。
- 允许修改符合资格的普通 scalar 字段，包括整格替换字符串、JSON、数组，以及对 nullable 字段赋 NULL。
- 不允许修改 PK、partition key、clustering key 等可能破坏现有数据分布或布局约定的列。
- Phase 1 不允许修改向量列、function 输入列和 function 输出列。
- 不支持 JSON path 修改、数组元素修改、append 或依赖旧值计算的赋值表达式。

WAL 仍保留 typed predicate 的表达能力。**接口只开放 PK 等值，不等于把 WAL 永久设计成仅能表达 PK。** 后续开放表达式 UPDATE 需要另行设计，不能因消息能编码就直接接受。

### 2.2 异步 Predicate Delete 的表达式白名单

Phase 1 仅支持以下基础表达式及其组合：

| 类别 | 支持的操作 |
| --- | --- |
| 逻辑组合 | `AND`、`OR`、`NOT`，以及括号分组 |
| 比较 | `>`、`<`、`>=`、`<=`、`!=`、`==` |
| 集合成员判断 | `IN` |

例如 `age >= 18 AND region IN ["Shanghai", "Beijing"]`。这些操作沿用 [Milvus 既有基础表达式](https://milvus.io/docs/basic-operators.md)的类型检查及求值规则，包括已确定的 NULL 三值逻辑，不为异步 DELETE 另造一套比较语义。

白名单约束完整表达式树，不只是最外层运算符。包含不支持的函数、算术或其他操作时，不能因为外层是比较就放行；操作数类型或组合不支持时同样拒绝。完整节点／类型校验属于实现对接，不再作为开放更多表达式的待决设计。

**异步接口不支持的表达式，在 WAL append 前直接拒绝，不自动回退到同步 delete-by-expr。** 客户端需要时显式改用现有同步 delete-by-expr 路径；该路径的既有语义与使用前提不变。这样不会在一次异步请求中悄悄引入前台查询或改变执行方式。

该白名单只规定 predicate DELETE 的准入范围；UPDATE 仍仅支持 PK 等值条件与既定的常量赋值。

### 2.3 返回成功与处理范围

写请求在所需 WAL append 成功后返回，沿用当前 delete 的返回边界，不等待查询侧完成应用，也不等待后台重写。

这里的“实时点更新”表示更新通过实时消费路径进入查询状态，而不是只有下一次 Consolidation 后才生效；不是“返回成功即所有查询副本完成”。

**已定：原生异步 predicate delete 和 PK 点更新都允许 collection 未加载时接收。** 二者在 append 前均不需要查询旧行或取得 QN 查询结果；完成 schema、类型、作用范围及权限校验后即可写入 WAL，由数据侧持久保存并执行。

未加载期间不要求为了执行点更新而加载 collection。以后 fresh load 通过已物化的 `base@T + delete@T` 及尚未物化的点更新／PK-delete 后缀恢复正确状态；pending predicate deletes 仍只在数据侧执行。“不依赖查询结果”不是“执行器不读取数据”，也不表示未加载时存在可以立即读取的 QN 状态。

客户端另行调用现有同步 delete-by-expr 时，仍受该路径原有前提约束；异步接口不替客户端切换路径。

不支持的 UPDATE 条件应在 append 前拒绝，不通过查询展开为多个点更新，也不暗中回退为 partial upsert。

## 3. 已定：更新对象与时间模型

### 3.1 Update-all-matches，而不是寻找 global latest

对于 `UPDATE@200 WHERE pk=100`，更新所有满足以下条件的物理行：

1. 属于该操作的 collection、channel、partition 等适用范围。
2. PK 等于 100。
3. 有效行诞生时间早于该 UPDATE 的时间。
4. 在相应执行状态中仍然存活。

有效行诞生时间沿用既有 insert/import 等路径的时间解释，不能用文件写入时间替代。

例如：

```text
S1：pk=100，insert_ts=80，age=17，city="Shanghai"
S2：pk=100，insert_ts=95，age=19，city="Beijing"

UPDATE@200 SET age=23 WHERE pk=100

S1：age=23，city="Shanghai"，insert_ts 仍为 80
S2：age=23，city="Beijing"，insert_ts 仍为 95
```

两行都更新，不要求单个 segment 判断自己是否持有全局最新版本。未修改字段及向量分别保留原值；不会将不同物理行合成一行。

没有匹配的存活行时不插入新行。已经删除的行不因 UPDATE 复活。查询侧既有的重复 PK 过滤、排序或去重是另一个步骤，**不是 update-all-matches 正确性的前提，也不代表所有查询路径都承诺选择最新行**。

### 3.2 三种时间含义必须分开

| 概念 | 含义 | 是否因某个 cell 更新而改动 |
| --- | --- | --- |
| Row timestamp | 原行的诞生时间，用于行的时间资格等既有判断 | 不改动 |
| Update timestamp | 这次字段更新的逻辑时间，保存在 update delta／字段版本中 | 每次更新独立携带 |
| 物化边界 T | 本次结果已完整执行到哪个 mutation prefix | 随完整物化结果推进，不是行的新生日 |

UPDATE 不把整行或未修改列的 timestamp 改成 update timestamp。segment 的插入范围也不应被当作“最新字段修改时间”；具体 StartPosition/DmlPosition 仍是既有 WAL／恢复元数据，不另行改写为字段更新时间范围。

在历史仍被保留时，查询可以按自身 timestamp 选择字段版本。Consolidation 或 full rewrite 折叠历史后，**沿用当前 Milvus 的边界，不新增跨重写的历史查询保证**：不引入 UNDO、固定保留窗口，也不要求恢复已被折叠的旧值。

正在使用旧表示的查询必须受到保护，不能在它仍依赖旧 base 或历史时将其回收。这不等于长期保留旧表示供以后到达的历史查询使用。

## 4. 已定：实时更新与异步删除如何共存

### 4.1 数据侧始终按完整逻辑顺序执行

PK Delete、Predicate Delete、UPDATE 在数据侧归并为同一个按有效 TimeTick 排序的序列。文件类型、上传顺序、任务完成顺序都不能替代这个顺序。

执行时：

- 后面的 predicate 看到前面 UPDATE 修改后的值。
- DELETE 命中后，该行不再参与后续 UPDATE。
- 操作不能影响在其时间资格之外、后来插入的行。
- 不能先合并所有 UPDATE，再统一执行 DELETE，也不能丢掉中间会被 predicate 使用的更新值。

### 4.2 接受异步窗口，但不改变最终执行结果

```text
原行：pk=100，age=18

@100 DELETE WHERE age=18        // 异步 predicate delete
@110 UPDATE SET age=23 WHERE pk=100
```

QN 尚未收到删除的物化结果时，可以暂时展示 `age=23`。数据侧仍按先删后改执行：原行在 @100 被删除，@110 对该行不产生更新，最终该行消失。

这不是 UPDATE 撤销 DELETE，也不是允许后台丢掉删除。再追加一次 predicate delete 不会复活原行；第一次已命中的删除仍然成立。

QN 所处理的是实时 point update／PK delete，加上已经物化的 predicate-delete 结果。由于 pending predicate delete 不在 QN 执行，**该混合方案不承诺查询始终呈现包含全部先前 predicate deletes 的完整 WAL 前缀状态**。

反过来，若顺序为 `UPDATE@90 SET age=23`、`DELETE@95 WHERE age=23`，数据侧 DELETE 必须看到 23 并删除该行，不能在旧 base 的 18 上判断。

### 4.3 后续 upsert 与 point update 不同

若 @110 是 upsert，生成了一个诞生于 @110 的新行，那么该新行不属于 DELETE@100 的时间范围，可以存活。

point update 不创建新行；它不能通过更新字段使原行逃离之前已经命中的删除。这两个结果都来自行身份与时间资格，不需要另加“发生冲突就跳过删除”的规则。

### 4.4 不要求跨 segment 同时体现 predicate delete

一次 predicate delete 可以先在部分 segments 中体现，再在其他 segments 中体现。不要求所有受影响 segments 同一时刻切换，也不增加跨 segment 原子事务。

但两个局部完整性要求不变：

- 一次 Consolidation／compaction task 的完整逻辑结果必须原子提交。
- 单个 QN segment 使用的 base、索引、删除基线和字段版本读取规则必须兼容，不能混用新旧表示。

前者是任务结果完整性，后者是本地读正确性；它们不同于整次 predicate delete 的跨 segment 同时可见。

## 5. 已定：端到端职责划分

```text
Proxy：验证 typed predicate／赋值；UPDATE 限定 PK 等值
  → WAL：接收命令并赋予逻辑顺序
      ├─ 实时消费：PK 点更新 → QN 字段版本
      │           PK Delete → 既有删除路径
      │           Predicate Delete → 不在 QN 执行
      └─ 持久化消费：保存三类 delta
          → 完整 VChannel delta range 发布
          → L0 粗分发
          → DataNode 有序执行完整 delta prefix
              ├─ Delta Consolidation：列数据＋累计删除 bitmap
              └─ Full rewrite：写出存活行的新 segments
          → QN 加载或切换到完整物化结果
```

### 5.1 WAL 保存操作，不保存前台查询结果

逻辑消息需要表达操作类型、作用范围、绑定 schema 的 typed predicate、UPDATE assignments，以及有效 mutation timestamp。字段以绑定后的身份和类型解释，而不是在每次回放时重新按当前字段名猜测。

原生路径不在 append 前查询命中行，不把命中 offsets、segment-local 执行计划或实际命中数量作为 WAL 命令的前提。提交时未知的影响行数，不能包装成已经执行得到的精确结果。

### 5.2 跨类型持久化必须形成完整前缀

三类 delta 可以分别写文件，但同一发布区间必须完整包含全部适用类型；某类数量少也不能留到下一轮，某类没有记录则不需要生成空文件。

采用独立的 VChannel 级 `PublishDeltaPrefix`，不将单 segment 的 `SaveBinlogPaths` 简单加一个 timestamp 当作跨类型原子发布。

- SN 负责组织完整区间；MixCoord 维护权威、持久化的已发布前缀及关联文件。
- 新区间必须无缝接续，不接受缺口或新的部分重叠范围。
- 区分已应用请求与范围不匹配，不能仅凭一个更大的结束时间推断某批文件已经接纳。
- SN 恢复时先读取各 VChannel 的已发布状态，再恢复未完成工作。

L0 调度只能消费已完整发布的区间，不能因部分文件已经存在就提前调度。这里规定的是逻辑完整性，不承诺物理文件 exactly-once。

### 5.3 L0 只分发，不执行 predicate

MixCoord 按 collection、channel、partition、segment 生命周期及时间资格进行粗筛选。DataNode 可保留现有 PK-delete Bloom Filter 过滤；不能用 PK BF 过滤一般 predicate delete，也不在此引入新的 UPDATE BF 优化。

分发必须连续、有序、无遗漏，覆盖所有应接收的目标。SN 发布批次与 L0 分发批次不必一致。

Growing 数据不能遗漏：沿用现有 seal policy 与 L0 生命周期约束，保留未完成目标覆盖所需的 delta；不能因为目标尚未 sealed 就丢掉操作。具体生命周期接入需验证，但不另起一种 seal policy。

#### 5.3.1 L0 分发的时间候选条件

在 collection、channel、partition 和 segment 状态均符合条件的前提下，只看时间维度，可以将单个 L0 与目标 segment 的候选关系简化为：

```text
L0：     [l_start, l_end]
目标：   [s_start, s_end]

可能需要分发：l_end > s_start
```

这里的 `s_start` 指有效行起始时间，import 等路径沿用既有有效时间解释。当前条件是严格大于，不是大于等于；相同时间的 DELETE 不删除该次 INSERT。不要求两个时间区间相交，也不要求 `s_end < l_end`：DELETE 可以作用于早于整个 L0 区间插入的行，也可以只影响目标 segment 中的一部分旧行。

实现按整轮 L0 task 的 `Pos` 上界筛选，即 `segmentEffectiveTs(target) < task.Pos`；该上界可能比单个 L0 的结束时间更大，因此这是保守的候选筛选，不是每个目标一定命中的判定。PK BF 及实际执行时的 PK／predicate、行时间资格判断仍各负其责。参见 [L0 目标筛选](../../internal/datacoord/compaction_task_l0.go)与 [L0 view 上界](../../internal/datacoord/compaction_l0_view.go)。

### 5.4 CDC：已定设计原则与实现兼容性检查

依据 [Milvus Replication & CDC 约定](../agent_guides/streaming-system/replication/replicate.md)，跨集群同步复用现有逻辑 WAL 复制链路，不为本功能另建一套复制协议。下面的设计原则已经确定；新消息是否已端到端接通仍需实现验证。

#### 5.4.1 已定：复制完整的逻辑变更，各集群独立执行

1. **WAL 必须足以表达变更。** Predicate DELETE 和 PK point UPDATE 的操作范围、绑定的字段身份与类型、常量值及必要的顺序／时间信息随逻辑消息传递。备集群仅凭复制来的 WAL 及其已有逻辑状态，就应能独立完成同样的逻辑变更，不能依赖主集群额外做过的动作。
2. **复制命令，不复制主集群的物理定位与物化结果。** 主备各自定位目标 segments／offsets，各自执行 predicate、生成列文件、`delete@T` 和 manifest。PK 可以表达逻辑目标；主集群计算的 segment ID、row offset 或文件路径不能作为备集群执行该 mutation 的目标身份。物理布局与物化时机可以不同，但相同逻辑输入下的执行结果应一致。
3. **保护执行语义，而不只传递字节。** 主备遵循相同的逻辑顺序、行时间资格、类型／NULL 规则及删除后不复活规则。表达式不能依赖执行时的随机值、本地环境或临时状态；时间与通道映射沿用现有 CDC 契约，不另造本地时间顺序。相同逻辑 mutation 不要求主备具有相同的本地 WAL MessageID 或物理消息布局。
4. **不是禁止本地 RPC 或 meta 更新。** `API → 仅主集群 RPC/meta 修改` 且 WAL 未完整表达该变更，会漏复制；`API → WAL → 各集群消费者通过 RPC/meta 更新执行变更` 则是正常的本地执行步骤。Consolidation 的本地文件／manifest 发布也属于后者，而不是额外复制这些物理结果。
5. **主集群 ACK 不等于备集群已收到或执行。** 继续沿用现有异步复制语义，不增加主备共同确认的写入承诺。新指令尚不支持复制时，应在复制部署上明确拒绝使用；不能将用户数据 mutation 标为 `_ur`，或接受后静默跳过。

该原则区分两类问题：**变更绕过 WAL** 会使备集群根本不知道发生了什么；**消息／执行能力不兼容** 则是消息已进入复制路径但无法被正确处理。两者都要检查，但不是同一个设计问题。

#### 5.4.2 实现对接：端到端能力兼容，不要求版本号完全相同

| 检查范围 | 必须验证的内容 |
| --- | --- |
| WAL 与复制链路 | 新消息能够编码、传输、接收、持久化与重放；不被跳过；必要的逻辑身份、schema 绑定和有效顺序信息不丢失 |
| 两侧 SN／数据侧消费者 | 能识别并持久化、分发新 delta；消费复制消息可触发与主集群等价的逻辑处理，不依赖主集群的额外调用 |
| 两侧 DataNode，包括云上共享 DN 池 | 能按同一规则执行三类 delta，生成兼容的 Consolidation／full rewrite 结果；相关任务只分配给支持该能力的 workers |
| 两侧 QN | 支持实时 PK 点更新的字段版本，以及物化后新 base、删除 bitmap 和兼容索引的加载／发布；仍不执行 pending predicate DELETE |

要求的是消息格式与执行语义兼容，**不是主备版本号必须完全一致，也不能只检查 instance 版本而忽略实际使用的 worker 能力**。现有 CDC 文档没有规定本功能的版本兼容矩阵、能力协商或共享 DN 池准入机制；这些是需要实现核对的条件，不是已经存在的保证。

启用前应验证上述链路；条件未满足时明确拒绝相关使用，不能以“已经写入 WAL”宣称功能天然支持 CDC。具体版本检查与准入落点留在实现对接，不再作为独立的复制协议设计。

## 6. 已定：统一 DeltaExecutor 与两种物化方式

### 6.1 统一执行语义

DataNode 以 base 数据批次为输入，按顺序执行适用的 PK deletes、predicate deletes、updates，维护当前字段值与存活行集合，再产生输出。

需要消费 delta 的 Sort、Mix、Clustering、bump-schema-version full rewrite，以及 Delta Consolidation，应共用执行语义；纯 additive schema 路径不等同于 full rewrite。

这个执行模型不能与 QN 的现成 raw chunk／index 查询执行模型混为一谈。本文不讨论 DeltaExecutor 的内存预算或并行策略。

### 6.2 Delta Consolidation：完整执行，但不物理删行

Consolidation 是独立的后台维护过程，不是 segment 合并。它必须：

1. 执行本轮完整 delta prefix，不能只执行 UPDATE 而遗漏穿插的 DELETE。
2. 重写该前缀影响的全部必要字段；不能只折叠部分更新列却声称整个前缀完成。
3. 保持行数、行顺序和 row offsets，不物理移除被删除的行。
4. 将累计删除效果持久化为 deleted-offset bitmap。

物理上只改受影响列，不等于语义上只理解这些列的 UPDATE。predicate 依赖的字段同样必须参与正确求值。

### 6.3 `base@T` 与 `delete@T` 共同表达完整结果

- `base@T`：完整执行截至 T 的适用 delta 后形成的字段数据表示；不意味着所有列都重新写文件。
- `delete@T`：完整、按序执行截至 T 的全部适用 delta 后，得到的累计 deleted offsets。

**`delete@T` 不是单独汇总 DELETE 命令的结果。它依赖与 UPDATE 交错执行后的命中情况。** 二者必须来自同一个完整前缀。

例如：

```text
原 base：offset 3 的 age=18；offset 8 的 age=23

@95  DELETE WHERE age=23    → 删除 offset 8
@100 UPDATE SET age=23 WHERE pk 对应 offset 3

base@100：offset 3 的 age=23；行位置仍保留
delete@100：{8}
```

恢复时如果在新 base 上再次执行 DELETE@95，就会错误删除 offset 3。因此，恢复应加载 `base@100 + delete@100`，只继续处理 100 之后的 delta。

`delete@T` 内不需要逐行 delete timestamp：它是该物化表示的累计删除事实。T 之后尚未折叠的实时 PK deletes 仍保留各自 timestamp，不能提前写入 `delete@T`。

每轮读取旧累计 bitmap、合入本轮删除，再写新 bitmap 文件，不原地覆盖旧对象。必须持久化这一结果，不能依赖 QN 的内存 DeletedRecord，否则 crash／release 后无法恢复已被消费的 predicate-delete 效果。

### 6.4 单列重写与 full rewrite 的边界

当 `b` 原来位于 `[a,b,c]` column group 时，选定方案是写新的 `b` 数据，并通过 manifest 将当前 `b` 的列映射切换过去；`a/c` 继续使用旧数据。

行布局不变，未修改列及兼容索引可以复用。旧文件中的 `b` 在新 manifest 下逻辑不可见，但它的物理字节不一定立即回收。

milvus-storage 的写列及列映射操作是可复用基础，不代表 Milvus 已实现 Delta Consolidation，也不是提供按 offset 原地修改 cell 的接口。

full rewrite 则可以移除删除行、改变 offsets、生成新 segments。它消费累计删除 bitmap 与全部适用后续 delta；旧布局的 offsets 不能直接移到新 segment，已消费指令也不再带入输出重复执行。

维护分工已定：updates 由 Consolidation 消化，不因此触发或加速 Mix；Mix 仍按既有删除及 segment 整理条件运行，但执行时消费全部适用 delta。何时调度、采用什么阈值不在本文展开。

### 6.5 Mix 输出时间范围与后续 L0 分发

当前 PK-delete 实现中，普通 Mix 消费各输入 segment 在 plan 中携带的 deltalogs／manifest，不保证所有适用的待分发 L0 已提前分发。未分发的 deletes 可以由后续 L0 compaction 继续分发给替换后的输出 segment。这是现有机制，不表示本设计允许 UPDATE／predicate DELETE 越过尚未完整执行的前序 mutation。

Mix 输出的时间范围描述其中行的时间，而不是此次 compaction 的发生时间：

| 输出元数据的时间戳 | 优先使用的值 | 缺少输出日志时间统计时的回退 |
| --- | --- | --- |
| `StartPosition.Timestamp` | 该输出 insert logs 的最小行时间戳 | 所有输入 segments 的最小 `StartPosition.Timestamp` |
| `DmlPosition.Timestamp` | 该输出 insert logs 的最大行时间戳 | 所有输入 segments 的最大 `DmlPosition.Timestamp`，并沿用 import commit timestamp 的归一化 |

在这一输出范围语境中，可以将它们理解为 start／end；不能据此把所有写入／恢复路径中的 DmlPosition 都定义为精确的最大行时间。缺少时间统计是实现／兼容分支，回退值提供覆盖输入的保守包络，不是另一套逻辑合并规则。普通行的 timestamp 不因 Mix 改成任务时间；import 行沿用既有有效时间归一化。

例如，输入范围为 `[20,40]` 与 `[70,80]`，在时间 200 合并，若输出仍保留两端的行，其范围为 `[20,80]`，不是 `[200,200]`。产生多个输出时，优先按每个输出实际保留的行分别计算；不要求所有输出都使用同一个总包络。参见 [Mix plan](../../internal/datacoord/compaction_task_mix.go)与 [输出范围计算及提交](../../internal/datacoord/meta.go)中的 `recalculateSegmentPosition`、`getCompactionFallbackPositions`、`completeMixCompactionMutation`。

当前调度器对同 channel 的 L0 与 Mix／Sort／bump-schema full rewrite 做互斥；L0 构建执行计划时重新选择目标。正确保留有效起始时间，使替换后的 segment 仍能被后续 L0 覆盖。参见 [调度互斥](../../internal/datacoord/compaction_inspector.go)。

**行时间范围与 mutation 完整前缀是两种不同信息。** 时间候选条件说明后续 delta 还可能影响哪些行，不能证明某个时间之前的 delta 已全部分发或执行。物理删列须额外遵守 §6.6，不能从普通 Mix 的现有行为直接推出旧 mutation 已完整处理。

### 6.6 已定：DROP FIELD 前完整执行旧字段依赖

对于 `DROP FIELD@x`，任何 rewrite 在物理移除字段之前，都必须保证：**该 segment 已收齐并按序执行 x 之前的全部适用 predicate deletes，以及这些 predicates 之前的 updates。** 不能只收齐删除指令，却遗漏决定 predicate 输入值的前序更新；也不能把尚未执行的旧 predicate 留到删列之后再处理。此前已完整物化的前缀不重复执行。

旧列必须保留在读取／执行视图中，以原字段身份、类型和值参与 DeltaExecutor；完成上述执行后，才在输出阶段按新 schema 移除该列。不能在 reader 中提前排除 dropped column，再尝试执行依赖它的 predicate。

这一约束适用于**所有实际物理删列的 rewrite 路径**，包括采用新 schema 的普通 Mix、Sort、Clustering 和 bump-schema full rewrite，不只约束专门的 drop-field 任务。

PK 列不能 drop，因此 PK delete 不需要仅因为删列而另加一项等待条件；这不取消 §4.1 的三类 delta 有序执行或 §5.2 的完整前缀要求。

**以上约束已定。** 如何把完整分发条件接入调度、在计划中保留执行所需的旧 schema，以及让 reader／writer 分别遵守执行／输出 schema，属于实现对接与验证；不再把“是否先执行旧 predicates 再删列”列为待决项，也不据此声称现有代码已满足约束。

### 6.7 已定：Mix 从各输入起点补齐到共同物化边界

每个输入 segment 使用自己的已物化边界 `T_i` 初始化 DeltaExecutor：读取该输入的 base 和累计删除状态，只按序执行 `(T_i, T]` 内的完整适用 delta。`T_i` 是已经执行并物化的边界，不是 L0 已分发到该 segment 的进度。

**所有输入在本次 Mix 内补齐到共同的结束边界 T，不需要事先分别执行额外 Consolidation。** 已物化的前缀不重放；必须确认各输入的适用后缀完整覆盖到 T，不能只取最大 mutation timestamp 作为完整性的证明。T 不得早于任何输入已有的物化边界。

```text
S1：base@100 + delete@100 → 执行 (100,200] 的完整适用 delta
S2：base@150 + delete@150 → 执行 (150,200] 的完整适用 delta

各输入处理后的存活行 → 合并写出 → 所有输出的物化边界 T=200
```

当前 [mergeSplit](../../internal/datanode/compactor/mix_compactor.go) 按输入建立 EntityFilter；[mergeSort](../../internal/datanode/compactor/merge_sort.go) 也为每个输入保留独立的 filter。新 DeltaExecutor 沿用这种按输入组织执行上下文的方向，而不是把不同输入的历史直接混成一份无来源的执行状态。

输入与输出不是一一对应：多个输入共用输出 writer，可能生成一个或多个新 segments。这里“统一 `base@T`”指**所有输出采用同一个物化边界 T**，不是只生成一个 segment；各输出的行时间范围仍按 §6.5 分别计算。Full rewrite 已物理移除删除行，不携带旧布局 bitmap；后续各输出只继续消费 `ts > T` 的 delta。

起点／终点规则已定；计划如何携带各输入边界、证明后缀覆盖、加载累计删除状态及提交输出 T，归入实现对接与验证。

## 7. 已定：QN 必须保留哪些正确性机制

取消 predicate delete 的实时执行，不等于取消 UPDATE 的查询侧支持。QN 在 fresh load、recover、Reopen 和 serving 阶段都不执行 pending predicate deletes，但仍需处理实时点更新及其尚未物化的历史。

### 7.1 字段版本与查询协作

点更新按 PK 定位符合条件的 offsets，将字段版本加入 UpdatesRecord，不原地修改共享 base。

对于读取某个字段的条件节点：

```text
U = 本次快照下、当前计算范围内存在可见更新的 offsets
B = base scalar index／raw chunk 的命中 offsets
D = 对 U 的最新可见更新值求值后，命中的 offsets

修正结果 = (B - U) ∪ D
```

U 表示“有可见更新”，不是“更新值命中条件”。多字段表达式必须使用一致快照下的字段值；不能只在整个 WHERE 的最终 bitmap 上补一次更新。

SELECT 对最终选中的 offsets 先查 UpdatesRecord：有可见版本就使用，包括明确写入的 NULL；没有可见版本才读 base。

NULL 是每个历史版本的属性，不能与“没有可见 update”混淆。条件求值需要同时修正 result／validity；纯 AND／OR／NOT 节点复用三值逻辑，不访问 UpdatesRecord。

### 7.2 旧索引仍可使用，但不能只相信旧索引

新列及其兼容索引尚未准备好时，继续使用旧 base/index＋UpdatesRecord 得到结果。待新资源就绪后，按既有 Reopen 方向整体切换。

不能将新 base 与描述旧值的 scalar index 组合，也不能在仍使用旧 base/index 时提前丢弃必要的 update 历史。原本未建索引的字段不因本方案强制新建索引。

**这个选择已经确定。** 剩下的是新索引与列版本绑定、就绪判断和 Reopen 接入，不再将“等索引还是先切 raw-only”列为待决方案。

### 7.3 加载、切换与历史回收

Fresh load 从完整的 `base@T + delete@T` 开始，补齐尚未物化的实时点更新与 PK-delete 后缀；它们可来自 segment deltas、适用 L0 及现有 buffer catch-up。必须保持完整和有序，但不回放 pending predicate deletes。

Consolidation Reopen 复用现有发布机制，将新 base、兼容索引、删除基线及 Records 的读取边界一起切换。UpdatesRecord／DeletedRecord 复用底层历史存储，保留已执行的 T 之后后缀，不为每轮切换全量重建或重新求值。

逻辑读取边界与物理 GC 分开：只有已经成功发布的本地边界才能授予旧历史回收资格，候选 T 或数据侧刚生成的 T 不能提前破坏旧状态。GC 何时实际释放空间是另一件事。

查询与完整 mutation 对旧状态的使用都要接入既有保护机制。Search 的后续取值／reduce、两阶段 Retrieve 等只要仍依赖旧状态，就不能中途切到不兼容的新表示。

上述是已定的正确性方向；精确的交接、锁和 lease 接入还需实现验证，本文不展开进度与发布时序协议。

## 8. 可行性：可复用基础与必须新增的能力

| 现有基础 | 可复用内容 | 不能据此声称已完成的部分 |
| --- | --- | --- |
| [Delete WAL append](../../internal/proxy/dml/task_delete_streaming.go) | append 成功后返回的写入边界 | 原生 UPDATE／predicate-delete 消息与校验 |
| [WAL TimeTick reorder](../../internal/streamingnode/server/wal/utility/reorder_buffer.go) | 消费端按确认的逻辑时间重排消息 | 三类 delta 文件、L0、执行器之间的完整有序衔接 |
| [InsertRecord PK lookup](../../internal/core/src/segcore/InsertRecord.h) | 查找本 segment 内同 PK 的多个 offsets 与时间资格 | UPDATE 的存活判断、写入版本及全部读取路径对接 |
| [L0 materializer](../../internal/streamingnode/server/wal/vchannel/l0materializer/materializer.go) | 现有 L0 存储表示与 sync 基础 | VChannel 级跨类型完整范围发布 |
| [EntityFilter](../../internal/compaction/entity_filter.go) | 当前删除、TTL 与有效行时间处理 | 交错执行 UPDATE／predicate DELETE 的 DeltaExecutor |
| [Segcore Reopen / PublishState](../../internal/core/src/segcore/ChunkedSegmentSealedImpl.cpp) | 准备资源、保护在途使用、原子发布本地状态 | Records 读取视图与累计删除基线接入 |
| [DeletedRecord](../../internal/core/src/segcore/DeletedRecord.h) | 当前删除记录、mask 及查询能力 | 即使 timestamped 后缀为空，仍正确使用持久化删除基线 |

已选定的 UpdatesRecord 容器方向保留为实现依据，不重新列为架构待决项：字段定位；Abseil B-tree 保存 offset＋HistoryHandle；Boost.Container deque 保存按时间追加的 history；大 payload 使用 Arrow owning Buffer；field-level 读写锁保护内部结构。块大小、内存开销和性能比较均不在本文范围内。

结论：现有架构提供必要的定位、存储与发布基础，**没有发现必须先引入全局 PK 定位器或 QN predicate-delete 执行器的前提**。但新消息、跨类型完整性、DeltaExecutor、Consolidation 和查询字段版本支持都仍是新增工作，不能把“存在复用点”当作端到端实现完成。

## 9. 已确定的设计项汇总

| 设计项 | 已确定结论 |
| --- | --- |
| 操作范围 | 实时 PK 等值点更新；异步 predicate delete；既有 PK delete 路径保留 |
| 重复 PK | Update-all-matches，更新所有符合范围、时间及存活条件的行，不寻找 global latest |
| 字段限制 | 普通合格 scalar 整格常量替换；禁止分布相关列、向量和 function 输入／输出 |
| 返回语义 | WAL append 成功后返回，不将其等同于已应用或已知影响行数 |
| Unloaded 准入 | 原生异步 predicate delete 与 PK 点更新都不要求 collection 已加载；以后加载恢复物化状态及适用后缀 |
| 时间模型 | Row timestamp 不变，UPDATE 独立带 ts，T 表示完整物化边界 |
| 混合执行 | 接受 QN 暂时展示随后被异步删除的更新行；数据侧仍先删后改、不复活 |
| 跨 segment 行为 | Predicate delete 可分批体现；单任务结果和单 segment 资源组合仍必须完整 |
| WAL 表达 | 保留 typed predicate；Phase 1 UPDATE admission 仅 PK 等值 |
| CDC | 完整逻辑 mutation 经 WAL 复制；主备独立定位、执行与物化；端到端能力兼容属于实现检查，不要求版本号完全相同 |
| 异步 DELETE 准入 | 仅基础逻辑、比较与 IN；完整表达式不支持则 append 前拒绝，不自动回退 |
| 持久化与分发 | 独立 VChannel range 发布；跨类型完整；L0 只粗分发，连续、有序、无遗漏 |
| 后台执行 | 三类 delta 归并执行；Consolidation 完整消费 prefix，不物理删行 |
| 持久化结果 | `base@T + delete@T`；累计 bitmap 绑定行布局；不重放已经物化的 predicates |
| Full rewrite | 消费完整适用 delta 并输出存活行；不携带旧布局删除 offsets |
| Mix 与 L0 时间范围 | 输出保留行的有效时间范围，不使用任务时间；L0 时间候选关系为 `l_end > s_start`，不代表已消费完整 mutation prefix |
| Mix 物化边界 | 各输入从自己的 `T_i` 之后执行到共同 T，在 Mix 内补齐；无需额外 Consolidation，所有输出采用同一 T |
| DROP FIELD | 物理删列前收齐并按序执行旧 predicate deletes 及前序 updates；旧列参与执行，输出时移除；所有实际删列的 rewrite 遵守 |
| 查询正确性 | UpdatesRecord 参与条件、取值、NULL 处理；QN 不执行 pending predicate delete |
| Scalar index | 旧 base/index＋UpdatesRecord 继续服务，新兼容资源就绪后整体切换 |
| 历史与回收 | 不新增跨重写历史查询保证；保护在途使用；GC 依据成功发布的读取边界 |

这里的“已定”表示语义或技术方向已选择，不等于实现已验证。

## 10. 还需要讨论什么

### 10.1 真正未闭合的逻辑／支持边界

基础表达式准入、CDC 设计原则、索引切换方向、drop-field 执行约束、unloaded 点更新及 Mix 共同物化边界均已确定，移出待决项。已定规则的具体接入列于 §10.2，不重新包装成顶层方案选择。

本轮复查新增一项待确认的特殊字段语义：**点更新与行级 TTL 的交互。** Milvus 允许通过 `ttl_field` 将一个 TIMESTAMPTZ 字段指定为绝对过期时间；它不改变数据分布，却会改变整行的存活状态，参见 [Milvus TTL 文档](https://milvus.io/docs/set-collection-ttl.md)。目前本文的普通 scalar 支持范围尚未明确是否包含该角色。

例如，行原定 10:00 过期，09:59 提交点更新将过期时间延至 11:00，后台到 10:30 才执行：若先按旧过期值丢弃行，再执行 UPDATE，就无法得到延长后的状态。当前数据侧 [EntityFilter](../../internal/compaction/entity_filter.go) 使用任务的 currentTime 判断过期，查询侧 [TTL expression](../../internal/core/src/exec/expression/Expr.cpp) 也会读取该字段判断存活。需要确认 Phase 1 是否开放这类字段更新，以及开放时 mutation 存活判断、历史回放与 TTL 清理的顺序。此项尚未选定支持或禁止策略，不改变已经确定的“不复活已删除行”规则。

### 10.2 已有方向，但需要实现对接与正确性验证

以下不应再次作为“选 A 还是选 B”的顶层决策，但也不能标成实现已完成：

- **白名单执行一致性：** 前台检查完整表达式树与类型，后台沿用同一求值语义；不支持的异步请求不 append、不查询 PK、不自动切换到同步路径。
- **CDC 与部署兼容：** 按 §5.4.2 核对新消息的复制／消费链路，以及主备实际使用的 SN／QN／DN 能力；验证前不宣称已支持，不兼容时明确拒绝。主备仅凭 WAL 即可独立执行的设计原则已定。
- **Schema 与删列执行：** Append 侧接入绑定 schemaVersion 的校验；物理删列前验证 §6.6 的完整分发／执行条件，保留旧字段参与求值，输出时再移除。覆盖所有实际删列的 rewrite，不能只验证 bump-schema 路径，也不能仅凭 plan 当前携带的文件推断前缀完整。
- **消息到目标的完整覆盖：** Growing／Sealed、load catch-up、L0 分发及 rewrite replacement 后，仍能覆盖全部适用行，且不会重复产生逻辑效果。
- **Unloaded 点更新恢复：** 未加载时仍完整持久化点更新；加载正确恢复物化状态及未物化后缀，不遗漏也不重复产生逻辑效果，不回放 pending predicate deletes。
- **Mix 共同前缀接入：** 按 §6.7 保留各输入的 `T_i` 和累计删除状态，验证 `(T_i,T]` 完整覆盖；所有输出提交相同 T，后续只承接 T 之后的 delta，不能用行时间范围代替物化边界。
- **全部读取入口：** 条件节点、SELECT、Search／Query 的后续处理及既有 query-PK DELETE 使用正确字段版本；不能只接通一条查询路径就声称完成。
- **完整物化输出：** predicate 使用前序更新值；零命中也算操作已经执行；多列更新、删除 bitmap 和所声明的完整前缀一致。
- **索引与 Records 接入：** 新索引绑定正确列版本；新 base、删除基线、读取规则一致；后缀不丢失；在途使用不被 GC 破坏。
- **存储提交落地：** 完整范围与完整任务结果的原子性已是要求；具体接口、编码及 catalog 提交实现仍需完成，不能假定现有 API 自动满足。

### 10.3 本轮明确不进入的层次

资源配额、积压限制、并发数、调度阈值、块大小、性能比较、故障恢复专项、运行状态接口、进度展示和副本完成状态留到后续专项。它们未被删除或视为已经解决，只是不作为本轮正确性／逻辑设计继续扩展的议题。

**当前结论：unloaded 点更新和 Mix 共同物化边界已闭合；下一项可讨论行级 TTL 字段更新这一存活语义边界。其他已定规则继续做实现对接与正确性验证，不扩展到资源／性能或稳定性专项。**
