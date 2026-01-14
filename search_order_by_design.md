# Search Order By - Initial Design Interface

This document is a starting point for user-facing design of `order_by` support after vector search.

## Goal
Provide a basic calling interface that lets users sort vector search results by one or more scalar fields, and optionally combine with `group_by`.

## Basic API

### Python (PyMilvus-style)

```python
res = milvus_client.search(
    collection_name="example",
    data=vectors,
    limit=10,
    anns_field="embeddings",
    output_fields=["id", "price", "rating", "category"],
    order_by=[
        {"field": "price", "order": "asc"},
        {"field": "rating", "order": "desc"}
    ]
)
```

### Group By + Order By

```python
res = milvus_client.search(
    collection_name="example",
    data=vectors,
    limit=10,
    anns_field="embeddings",
    output_fields=["id", "price", "rating", "category"],
    group_by_field="category",
    group_size=3,
    strict_group_size=True,
    order_by=[
        {"field": "price", "order": "asc"}
    ]
)
```

## Notes (TBD)
- `order_by` applies after vector search (post-TopK) unless specified otherwise.
- Multiple fields are applied in order as lexicographic sort keys.
- `group_by` + `order_by`: group first, then order groups by the first item's order_by field value in each group. The order within each group is preserved (determined by SearchGroupByNode).

## Pipeline Sketch (Search)
Relative placement of `OrderByNode` in the segcore search pipeline.

### Order By Only (无过滤条件)
```
MvccNode
  -> VectorSearchNode
    -> [OrderByNode]
```

### Order By Only (有过滤条件)
```
FilterBitsNode
  -> MvccNode
    -> VectorSearchNode
      -> [OrderByNode]
```

### Group By + Order By (无过滤条件)
```
MvccNode
  -> VectorSearchNode
    -> [SearchGroupByNode]
      -> [OrderByNode]
```

### Group By + Order By (有过滤条件)
```
FilterBitsNode
  -> MvccNode
    -> VectorSearchNode
      -> [SearchGroupByNode]
        -> [OrderByNode]
```

**Node Execution Order**:
- `FilterBitsNode` (if predicates exist) -> `MvccNode` -> `VectorSearchNode` -> `[SearchGroupByNode]` -> `[OrderByNode]`
- `FilterBitsNode` filters data based on predicates (business logic filtering)
- `MvccNode` filters deleted and expired data based on query timestamp (MVCC filtering)
- When both exist, `FilterBitsNode` runs before `MvccNode` to reduce the dataset before MVCC check
- `OrderByNode` runs after vector search; if present, it runs after Group By

## SearchOrderByOperator 设计

### 核心概念

`SearchOrderByOperator` 负责对向量搜索结果按标量字段排序。它有两种输入形式，取决于 Pipeline 中是否存在 `SearchGroupByNode`。

### 输入形式

#### 形式 1：无 Group By（从 VectorSearchNode 接收）

**输入来源**：`VectorSearchNode` 通过 `QueryContext` 写入的 `SearchResult`

**输入数据**：
- `SearchResult.distances_`：相似度距离
- `SearchResult.seg_offsets_`：Segment 内的 offset 数组
- `SearchResult.primary_keys_`：主键数组（可选）
- `SearchResult.topk_per_nq_prefix_sum_`：每个 query 的结果数量前缀和

**处理逻辑**：
1. 从 `QueryContext` 读取 `SearchResult`
2. 根据 `seg_offsets_` 读取 `order_by` 字段的值
3. 按 `order_by` 字段对结果排序（支持多字段字典序）
4. 更新 `SearchResult` 中的 `seg_offsets_` 和 `distances_`（保持一致性）

#### 形式 2：有 Group By（从 SearchGroupByNode 接收）

**输入来源**：`SearchGroupByNode` 通过 `QueryContext` 更新的 `SearchResult`

**输入数据**：
- `SearchResult.distances_`：相似度距离
- `SearchResult.seg_offsets_`：Segment 内的 offset 数组（已分组）
- `SearchResult.group_by_values_`：Group By 字段值数组（已存在）
- `SearchResult.topk_per_nq_prefix_sum_`：每个 query 的结果数量前缀和

**处理逻辑**：
1. 从 `QueryContext` 读取 `SearchResult`（包含 `group_by_values_`）
2. 识别每个 Group（通过 `group_by_values_`）
3. **对 Group 进行排序**：使用每个 Group 的第一条数据的 `order_by` 字段值来对 Group 进行排序
4. 保持每个 Group 内的结果顺序不变（由 `SearchGroupByNode` 决定）
5. 更新 `SearchResult` 中的 `seg_offsets_`、`distances_` 和 `group_by_values_`（保持一致性）

### 数据访问模式

**读取 order_by 字段值**：
- 通过 `seg_offsets_` 访问 Segment 数据
- 使用 `DataGetter` 模式（类似 `SearchGroupByOperator`）
- 支持多种数据类型（INT8, INT16, INT32, INT64, FLOAT, DOUBLE, VARCHAR, JSON 等）

**排序策略**：
- 多字段排序：按字段顺序进行字典序排序
- 升序/降序：每个字段可独立指定排序方向
- 稳定性：保持相同排序键的相对顺序

### 与现有组件的交互

**与 VectorSearchNode**：
- `VectorSearchNode` 将搜索结果写入 `QueryContext::search_result_`
- `SearchOrderByOperator` 从 `QueryContext` 读取并更新 `SearchResult`

**与 SearchGroupByNode**：
- `SearchGroupByNode` 更新 `SearchResult`，添加 `group_by_values_`
- `SearchOrderByOperator` 在分组结果基础上进行排序
- 需要保持 `seg_offsets_`、`distances_` 和 `group_by_values_` 的一致性

## Implementation Notes

### Key Design Decisions

1. **数据传递方式**：通过 `QueryContext::search_result_` 传递，而非 Operator 的 `input_`/`output_`
   - 与 `VectorSearchNode` 和 `SearchGroupByNode` 保持一致
   - `SearchResult` 包含大量数据，避免多次拷贝

2. **输入形式统一**：两种输入形式都从 `QueryContext` 读取 `SearchResult`
   - 形式 1：`SearchResult` 由 `VectorSearchNode` 写入
   - 形式 2：`SearchResult` 由 `SearchGroupByNode` 更新（添加 `group_by_values_`）

3. **排序范围**：
   - 无 Group By：对所有结果排序（按 query 分组）
   - 有 Group By：对 Group 进行排序（使用每个 Group 的第一条数据的 order_by 字段值），Group 内的结果顺序保持不变

4. **字段访问**：使用 `DataGetter` 模式（参考 `SearchGroupByOperator`）
   - 支持多种数据类型
   - 支持 JSON 字段路径访问
   - 高效的数据读取

