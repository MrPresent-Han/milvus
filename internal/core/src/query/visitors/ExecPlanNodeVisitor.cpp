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

#include "query/generated/ExecPlanNodeVisitor.h"

#include <utility>

#include "query/PlanImpl.h"
#include "query/SubSearchResult.h"
#include "query/generated/ExecExprVisitor.h"
#include "segcore/SegmentGrowing.h"
#include "utils/Json.h"
#include "log/Log.h"
#include "plan/PlanNode.h"
#include "exec/Task.h"

namespace milvus::query {

namespace impl {
// THIS CONTAINS EXTRA BODY FOR VISITOR
// WILL BE USED BY GENERATOR UNDER suvlim/core_gen/
class ExecPlanNodeVisitor : PlanNodeVisitor {
 public:
    ExecPlanNodeVisitor(const segcore::SegmentInterface& segment,
                        Timestamp timestamp,
                        const PlaceholderGroup& placeholder_group)
        : segment_(segment),
          timestamp_(timestamp),
          placeholder_group_(placeholder_group) {
    }

    SearchResult
    get_moved_result(PlanNode& node) {
        assert(!search_result_opt_.has_value());
        node.accept(*this);
        assert(search_result_opt_.has_value());
        auto ret = std::move(search_result_opt_).value();
        search_result_opt_ = std::nullopt;
        return ret;
    }

 private:
    template <typename VectorType>
    void
    VectorVisitorImpl(VectorPlanNode& node);

 private:
    const segcore::SegmentInterface& segment_;
    Timestamp timestamp_;
    const PlaceholderGroup& placeholder_group_;

    SearchResultOpt search_result_opt_;
};
}  // namespace impl

static SearchResult
empty_search_result(int64_t num_queries, SearchInfo& search_info) {
    SearchResult final_result;
    SubSearchResult result(num_queries,
                           search_info.topk_,
                           search_info.metric_type_,
                           search_info.round_decimal_);
    final_result.total_nq_ = num_queries;
    final_result.unity_topK_ = search_info.topk_;
    final_result.seg_offsets_ = std::move(result.mutable_seg_offsets());
    final_result.distances_ = std::move(result.mutable_distances());
    return final_result;
}

void
AppendOneChunk(BitsetType* result, const bool* chunk_ptr, size_t chunk_len) {
    // Append a value once instead of BITSET_BLOCK_BIT_SIZE times.
    auto AppendBlock = [&result](const bool* ptr, int n) {
        for (int i = 0; i < n; ++i) {
#if defined(USE_DYNAMIC_SIMD)
            auto val = milvus::simd::get_bitset_block(ptr);
#else
            BitsetBlockType val = 0;
            // This can use CPU SIMD optimzation
            uint8_t vals[BITSET_BLOCK_SIZE] = {0};
            for (size_t j = 0; j < 8; ++j) {
                for (size_t k = 0; k < BITSET_BLOCK_SIZE; ++k) {
                    vals[k] |= uint8_t(*(ptr + k * 8 + j)) << j;
                }
            }
            for (size_t j = 0; j < BITSET_BLOCK_SIZE; ++j) {
                val |= BitsetBlockType(vals[j]) << (8 * j);
            }
#endif
            result->append(val);
            ptr += BITSET_BLOCK_SIZE * 8;
        }
    };
    // Append bit for these bits that can not be union as a block
    // Usually n less than BITSET_BLOCK_BIT_SIZE.
    auto AppendBit = [&result](const bool* ptr, int n) {
        for (int i = 0; i < n; ++i) {
            bool bit = *ptr++;
            result->push_back(bit);
        }
    };

    size_t res_len = result->size();

    int n_prefix =
        res_len % BITSET_BLOCK_BIT_SIZE == 0
            ? 0
            : std::min(BITSET_BLOCK_BIT_SIZE - res_len % BITSET_BLOCK_BIT_SIZE,
                       chunk_len);

    AppendBit(chunk_ptr, n_prefix);

    if (n_prefix == chunk_len)
        return;

    size_t n_block = (chunk_len - n_prefix) / BITSET_BLOCK_BIT_SIZE;
    size_t n_suffix = (chunk_len - n_prefix) % BITSET_BLOCK_BIT_SIZE;

    AppendBlock(chunk_ptr + n_prefix, n_block);

    AppendBit(chunk_ptr + n_prefix + n_block * BITSET_BLOCK_BIT_SIZE, n_suffix);

    return;
}

void
ExecPlanNodeVisitor::ExecuteExprNode(
    const std::shared_ptr<milvus::plan::PlanNode>& plannode,
    const milvus::segcore::SegmentInternalInterface* segment,
    BitsetType* bitset_holder) {
    auto plan = plan::PlanFragment(plannode);
    auto query_context = std::make_shared<milvus::exec::QueryContext>(
        "query id", segment, timestamp_);

    auto task = milvus::exec::Task::Create("task_expr", plan, 0, query_context);
    for (;;) {
        auto result = task->Next();
        if (!result) {
            break;
        }
        auto childrens = result->childrens();
        std::cout << "output result length:" << childrens[0]->size()
                  << std::endl;
        assert(childrens.size() == 1);
        if (auto child = std::dynamic_pointer_cast<FlatVector>(childrens[0])) {
            AppendOneChunk(bitset_holder,
                           static_cast<bool*>(child->GetRawData()),
                           child->size());
        } else {
            PanicInfo("expr return type not matched");
        }
    }
    // std::string s;
    // boost::to_string(*bitset_holder, s);
    // std::cout << s << std::endl;
}

template <typename VectorType>
void
ExecPlanNodeVisitor::VectorVisitorImpl(VectorPlanNode& node) {
    // TODO: optimize here, remove the dynamic cast
    assert(!search_result_opt_.has_value());
    auto segment =
        dynamic_cast<const segcore::SegmentInternalInterface*>(&segment_);
    AssertInfo(segment, "support SegmentSmallIndex Only");
    SearchResult search_result;
    auto& ph = placeholder_group_->at(0);
    auto src_data = ph.get_blob<EmbeddedType<VectorType>>();
    auto num_queries = ph.num_of_queries_;

    // TODO: add API to unify row_count
    // auto row_count = segment->get_row_count();
    auto active_count = segment->get_active_count(timestamp_);

    // skip all calculation
    if (active_count == 0) {
        search_result_opt_ =
            empty_search_result(num_queries, node.search_info_);
        return;
    }

    std::unique_ptr<BitsetType> bitset_holder;
    if (node.filter_plannode_.has_value()) {
        BitsetType expr_res;
        ExecuteExprNode(node.filter_plannode_.value(), segment, &expr_res);
        bitset_holder = std::make_unique<BitsetType>(expr_res);
        bitset_holder->flip();
    } else {
        if (node.predicate_.has_value()) {
            bitset_holder = std::make_unique<BitsetType>(
                ExecExprVisitor(*segment, this, active_count, timestamp_)
                    .call_child(*node.predicate_.value()));
            bitset_holder->flip();
        } else {
            bitset_holder = std::make_unique<BitsetType>(active_count, false);
        }
    }
    segment->mask_with_timestamps(*bitset_holder, timestamp_);

    segment->mask_with_delete(*bitset_holder, active_count, timestamp_);

    // if bitset_holder is all 1's, we got empty result
    if (bitset_holder->all()) {
        search_result_opt_ =
            empty_search_result(num_queries, node.search_info_);
        return;
    }
    BitsetView final_view = *bitset_holder;
    segment->vector_search(node.search_info_,
                           src_data,
                           num_queries,
                           timestamp_,
                           final_view,
                           search_result);

    search_result_opt_ = std::move(search_result);
}

std::unique_ptr<RetrieveResult>
wrap_num_entities(int64_t cnt) {
    auto retrieve_result = std::make_unique<RetrieveResult>();
    DataArray arr;
    arr.set_type(milvus::proto::schema::Int64);
    auto scalar = arr.mutable_scalars();
    scalar->mutable_long_data()->mutable_data()->Add(cnt);
    retrieve_result->field_data_ = {arr};
    return retrieve_result;
}

void
ExecPlanNodeVisitor::visit(RetrievePlanNode& node) {
    assert(!retrieve_result_opt_.has_value());
    auto segment =
        dynamic_cast<const segcore::SegmentInternalInterface*>(&segment_);
    AssertInfo(segment, "Support SegmentSmallIndex Only");
    RetrieveResult retrieve_result;

    auto active_count = segment->get_active_count(timestamp_);

    if (active_count == 0 && !node.is_count_) {
        retrieve_result_opt_ = std::move(retrieve_result);
        return;
    }

    if (active_count == 0 && node.is_count_) {
        retrieve_result = *(wrap_num_entities(0));
        retrieve_result_opt_ = std::move(retrieve_result);
        return;
    }

    BitsetType bitset_holder;
    // For case that retrieve by expression, bitset will be allocated when expression is being executed.
    if (node.is_count_) {
        bitset_holder.resize(active_count);
    }

    if (node.filter_plannode_.has_value()) {
        ExecuteExprNode(node.filter_plannode_.value(), segment, &bitset_holder);
        bitset_holder.flip();
    } else {
        if (node.predicate_.has_value() && node.predicate_.value() != nullptr) {
            bitset_holder =
                ExecExprVisitor(*segment, this, active_count, timestamp_)
                    .call_child(*(node.predicate_.value()));
            bitset_holder.flip();
        }
    }

    std::cout << bitset_holder.size() << std::endl;
    segment->mask_with_timestamps(bitset_holder, timestamp_);

    segment->mask_with_delete(bitset_holder, active_count, timestamp_);
    // if bitset_holder is all 1's, we got empty result
    if (bitset_holder.all() && !node.is_count_) {
        retrieve_result_opt_ = std::move(retrieve_result);
        return;
    }

    if (node.is_count_) {
        auto cnt = bitset_holder.size() - bitset_holder.count();
        retrieve_result = *(wrap_num_entities(cnt));
        retrieve_result_opt_ = std::move(retrieve_result);
        return;
    }

    bool false_filtered_out = false;
    if (GetExprUsePkIndex() && IsTermExpr(node.predicate_.value().get())) {
        segment->timestamp_filter(
            bitset_holder, expr_cached_pk_id_offsets_, timestamp_);
    } else {
        bitset_holder.flip();
        false_filtered_out = true;
        segment->timestamp_filter(bitset_holder, timestamp_);
    }
    retrieve_result.result_offsets_ =
        segment->find_first(node.limit_, bitset_holder, false_filtered_out);
    retrieve_result_opt_ = std::move(retrieve_result);
}

void
ExecPlanNodeVisitor::visit(FloatVectorANNS& node) {
    VectorVisitorImpl<FloatVector>(node);
}

void
ExecPlanNodeVisitor::visit(BinaryVectorANNS& node) {
    VectorVisitorImpl<BinaryVector>(node);
}

}  // namespace milvus::query
