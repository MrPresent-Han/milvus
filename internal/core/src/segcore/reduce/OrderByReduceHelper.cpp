// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License

#include "OrderByReduceHelper.h"
#include "segcore/SegmentInterface.h"
#include "segcore/ReduceStructure.h"
#include "common/EasyAssert.h"
#include "log/Log.h"
#include "exec/operator/search-groupby/SearchGroupByOperator.h"

namespace milvus::segcore {

int64_t
OrderByReduceHelper::ReduceSearchResultForOneNQ(int64_t qi,
                                                 int64_t topk,
                                                 int64_t& offset) {
    if (order_by_fields_.empty()) {
        // Fallback to base class implementation if no order_by fields
        return ReduceHelper::ReduceSearchResultForOneNQ(qi, topk, offset);
    }

    // Create comparator with order_by_fields
    SearchResultPairComparator comparator(order_by_fields_);

    std::priority_queue<SearchResultPair*,
                        std::vector<SearchResultPair*>,
                        SearchResultPairComparator>
        heap(comparator);
    pk_set_.clear();
    pairs_.clear();
    pairs_.reserve(num_segments_);

    // Initialize OpContext for reading field values
    milvus::OpContext op_ctx;

    for (int i = 0; i < num_segments_; i++) {
        auto search_result = search_results_[i];
        auto offset_beg = search_result->topk_per_nq_prefix_sum_[qi];
        auto offset_end = search_result->topk_per_nq_prefix_sum_[qi + 1];
        if (offset_beg == offset_end) {
            continue;
        }
        auto primary_key = search_result->primary_keys_[offset_beg];
        auto distance = search_result->distances_[offset_beg];

        pairs_.emplace_back(
            primary_key, distance, search_result, i, offset_beg, offset_end);
        auto* pair = &pairs_.back();

        // Read order_by field values
        ReadOrderByValues(pair);

        heap.push(pair);
    }

    // nq has no results for all segments
    if (heap.size() == 0) {
        return 0;
    }

    int64_t dup_cnt = 0;
    auto start = offset;
    while (offset - start < topk && !heap.empty()) {
        auto pilot = heap.top();
        heap.pop();

        auto index = pilot->segment_index_;
        auto pk = pilot->primary_key_;
        // no valid search result for this nq, break to next
        if (pk == INVALID_PK) {
            break;
        }
        // remove duplicates
        if (pk_set_.count(pk) == 0) {
            pilot->search_result_->result_offsets_.push_back(offset++);
            final_search_records_[index][qi].push_back(pilot->offset_);
            pk_set_.insert(pk);
        } else {
            // skip entity with same primary key
            dup_cnt++;
        }
        pilot->advance();
        if (pilot->primary_key_ != INVALID_PK) {
            // Read order_by values for the next result
            ReadOrderByValues(pilot);
            heap.push(pilot);
        }
    }
    return dup_cnt;
}

void
OrderByReduceHelper::ReadOrderByValues(SearchResultPair* pair) {
    if (order_by_fields_.empty() || pair->offset_ >= pair->offset_rb_) {
        pair->order_by_values_ = std::nullopt;
        return;
    }

    auto search_result = pair->search_result_;
    auto segment_interface =
        static_cast<const SegmentInterface*>(search_result->segment_);
    auto segment =
        dynamic_cast<const SegmentInternalInterface*>(segment_interface);
    AssertInfo(segment != nullptr,
               "Failed to cast SegmentInterface to SegmentInternalInterface");
    auto seg_offset = search_result->seg_offsets_[pair->offset_];

    milvus::OpContext op_ctx;
    std::vector<milvus::GroupByValueType> order_by_vals;
    order_by_vals.reserve(order_by_fields_.size());

    for (const auto& field : order_by_fields_) {
        auto data_type = segment->GetFieldDataType(field.field_id_);
        milvus::GroupByValueType val = std::nullopt;

        // Read field value based on type
        switch (data_type) {
            case DataType::BOOL: {
                auto getter = GetDataGetter<bool>(
                    &op_ctx, *segment, field.field_id_, field.json_path_);
                auto bool_val = getter->Get(seg_offset);
                if (bool_val.has_value()) {
                    val = milvus::GroupByValueType(
                        std::make_optional(std::variant<std::monostate, int8_t, int16_t,
                                                         int32_t, int64_t, bool, float, double,
                                                         std::string>(bool_val.value())));
                }
                break;
            }
            case DataType::INT8: {
                auto getter = GetDataGetter<int8_t>(
                    &op_ctx, *segment, field.field_id_, field.json_path_);
                auto int8_val = getter->Get(seg_offset);
                if (int8_val.has_value()) {
                    val = milvus::GroupByValueType(
                        std::make_optional(std::variant<std::monostate, int8_t, int16_t,
                                                         int32_t, int64_t, bool, float, double,
                                                         std::string>(int8_val.value())));
                }
                break;
            }
            case DataType::INT16: {
                auto getter = GetDataGetter<int16_t>(
                    &op_ctx, *segment, field.field_id_, field.json_path_);
                auto int16_val = getter->Get(seg_offset);
                if (int16_val.has_value()) {
                    val = milvus::GroupByValueType(
                        std::make_optional(std::variant<std::monostate, int8_t, int16_t,
                                                         int32_t, int64_t, bool, float, double,
                                                         std::string>(int16_val.value())));
                }
                break;
            }
            case DataType::INT32: {
                auto getter = GetDataGetter<int32_t>(
                    &op_ctx, *segment, field.field_id_, field.json_path_);
                auto int32_val = getter->Get(seg_offset);
                if (int32_val.has_value()) {
                    val = milvus::GroupByValueType(
                        std::make_optional(std::variant<std::monostate, int8_t, int16_t,
                                                         int32_t, int64_t, bool, float, double,
                                                         std::string>(int32_val.value())));
                }
                break;
            }
            case DataType::INT64: {
                auto getter = GetDataGetter<int64_t>(
                    &op_ctx, *segment, field.field_id_, field.json_path_);
                auto int64_val = getter->Get(seg_offset);
                if (int64_val.has_value()) {
                    val = milvus::GroupByValueType(
                        std::make_optional(std::variant<std::monostate, int8_t, int16_t,
                                                         int32_t, int64_t, bool, float, double,
                                                         std::string>(int64_val.value())));
                }
                break;
            }
            case DataType::VARCHAR:
            case DataType::STRING: {
                auto getter = GetDataGetter<std::string>(
                    &op_ctx, *segment, field.field_id_, field.json_path_);
                auto str_val = getter->Get(seg_offset);
                if (str_val.has_value()) {
                    val = milvus::GroupByValueType(
                        std::make_optional(std::variant<std::monostate, int8_t, int16_t,
                                                         int32_t, int64_t, bool, float, double,
                                                         std::string>(str_val.value())));
                }
                break;
            }
            case DataType::JSON: {
                auto getter = GetDataGetter<milvus::GroupByValueType>(
                    &op_ctx, *segment, field.field_id_, field.json_path_);
                val = getter->Get(seg_offset);
                break;
            }
            default:
                // Unsupported type, use null
                val = std::nullopt;
                break;
        }
        order_by_vals.push_back(val);
    }

    pair->order_by_values_ = std::move(order_by_vals);
}

void
OrderByReduceHelper::InitializeDataGetters(SearchResult* search_result) {
    // This method is reserved for future optimization (caching DataGetters)
    (void)search_result;
}

}  // namespace milvus::segcore

