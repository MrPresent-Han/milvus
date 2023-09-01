// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <memory>
#include <string>

#include "exec/expression/EvalCtx.h"
#include "exec/expression/VectorFunction.h"
#include "exec/QueryContext.h"
#include "expr/ITypeExpr.h"

namespace milvus {
namespace exec {

class Expr {
 public:
    Expr(DataType type,
         const std::vector<std::shared_ptr<Expr>>&& inputs,
         const std::string& name)
        : type_(type),
          inputs_(std::move(inputs)),
          name_(name),
          vector_func_(nullptr) {
    }

    Expr(DataType type,
         const std::vector<std::shared_ptr<Expr>>&& inputs,
         std::shared_ptr<VectorFunction> vec_func,
         const std::string& name)
        : type_(type),
          inputs_(std::move(inputs)),
          name_(name),
          vector_func_(vec_func) {
    }
    virtual ~Expr() = default;

    const DataType&
    type() const {
        return type_;
    }

    virtual void
    Eval(EvalCtx& context, VectorPtr& result) {
    }

 protected:
    DataType type_;
    const std::vector<std::shared_ptr<Expr>> inputs_;
    std::string name_;
    std::shared_ptr<VectorFunction> vector_func_;
};

using ExprPtr = std::shared_ptr<Expr>;

class SegmentExpr : public Expr {
 public:
    SegmentExpr(const std::vector<std::shared_ptr<Expr>>&& input,
                const std::string& name,
                const segcore::SegmentInternalInterface* segment,
                const FieldId& field_id,
                Timestamp query_timestamp,
                int64_t batch_size)
        : Expr(DataType::BOOL, std::move(input), name),
          segment_(segment),
          field_id_(field_id),
          query_timestamp_(query_timestamp),
          batch_size_(batch_size) {
        num_rows_ = segment_->get_active_count(query_timestamp_);
        size_per_chunk_ = segment_->size_per_chunk();
        AssertInfo(
            batch_size_ > 0,
            fmt::format("expr batch size should greater than zero, but now: {}",
                        batch_size_));
        if (segment_->type() == SegmentType::Growing) {
            AssertInfo(
                batch_size_ > size_per_chunk_,
                fmt::format("expr batch size should greater than size per "
                            "chunk {} for growing segment, but now {}",
                            size_per_chunk_,
                            batch_size_));
        }
        InitSegmentExpr();
    }

    void
    InitSegmentExpr() {
        is_index_mode_ = segment_->HasIndex(field_id_);
        if (is_index_mode_) {
            num_index_chunk_ = segment_->num_chunk_index(field_id_);
        } else {
            num_data_chunk_ = segment_->num_chunk_data(field_id_);
        }
    }

    int64_t
    GetNextBatchSize() {
        auto current_chunk =
            is_index_mode_ ? current_index_chunk_ : current_data_chunk_;
        auto current_chunk_pos =
            is_index_mode_ ? current_data_chunk_pos_ : current_index_chunk_pos_;
        auto current_rows =
            segment_->type() == SegmentType::Growing
                ? current_chunk * size_per_chunk_ + current_chunk_pos
                : current_chunk_pos;
        return current_rows + batch_size_ >= num_rows_
                   ? num_rows_ - current_rows
                   : batch_size_;
    }

    template <typename T, typename FUNC, typename... ValTypes>
    int64_t
    ProcessDataChunks(FUNC func, bool* res, ValTypes... values) {
        int processed_size = 0;

        for (size_t i = current_data_chunk_; i < num_data_chunk_; i++) {
            auto chunk = segment_->chunk_data<T>(field_id_, i);
            auto data_pos =
                (i == current_data_chunk_) ? current_data_chunk_pos_ : 0;
            auto size = (i == (num_data_chunk_ - 1))
                            ? (segment_->type() == SegmentType::Growing
                                   ? num_rows_ % size_per_chunk_ - data_pos
                                   : num_rows_ - data_pos)
                            : size_per_chunk_ - data_pos;

            if (processed_size + size >= batch_size_) {
                size = batch_size_ - processed_size;
            }

            const T* data = chunk.data() + data_pos;
            func(data, size, res + processed_size, values...);
            processed_size += size;

            if (processed_size >= batch_size_) {
                current_data_chunk_ = i;
                current_data_chunk_pos_ = data_pos + size;
                break;
            }
        }

        return processed_size;
    }

    int
    ProcessIndexOneChunk(FixedVector<bool>& result,
                         const FixedVector<bool>& chunk_res,
                         int processed_rows) {
        auto data_pos = current_index_chunk_ == num_index_chunk_
                            ? current_index_chunk_pos_
                            : 0;
        auto size = std::min(
            std::min(size_per_chunk_ - data_pos, batch_size_ - processed_rows),
            int64_t(chunk_res.size()));

        result.insert(result.end(),
                      chunk_res.begin() + data_pos,
                      chunk_res.begin() + data_pos + size);
        current_index_chunk_pos_ = data_pos + size;
        return size;
    }

    template <typename T, typename FUNC, typename... ValTypes>
    FixedVector<bool>
    ProcessIndexChunks(FUNC func, ValTypes... values) {
        typedef std::
            conditional_t<std::is_same_v<T, std::string_view>, std::string, T>
                IndexInnerType;
        using Index = index::ScalarIndex<IndexInnerType>;
        FixedVector<bool> result;
        int processed_rows = 0;

        for (size_t i = current_index_chunk_; i < num_index_chunk_; i++) {
            const Index& index =
                segment_->chunk_scalar_index<IndexInnerType>(field_id_, i);
            auto* index_ptr = const_cast<Index*>(&index);
            FixedVector<bool> chunk_res = std::move(func(index_ptr, values...));

            processed_rows +=
                ProcessIndexOneChunk(result, chunk_res, processed_rows);

            if (processed_rows >= batch_size_) {
                current_index_chunk_ = i;
                break;
            }
        }

        return result;
    }

 protected:
    const segcore::SegmentInternalInterface* segment_;
    const FieldId field_id_;
    Timestamp query_timestamp_;
    int64_t batch_size_;

    // State indicate position that expr computing at
    // because expr maybe called for every batch.
    bool is_index_mode_{false};
    bool is_data_mode_{false};

    int64_t num_rows_{0};
    int64_t num_data_chunk_{0};
    int64_t num_index_chunk_{0};
    int64_t current_data_chunk_{0};
    int64_t current_data_chunk_pos_{0};
    int64_t current_index_chunk_{0};
    int64_t current_index_chunk_pos_{0};
    int64_t size_per_chunk_{0};
};

std::vector<ExprPtr>
CompileExpressions(const std::vector<expr::TypedExprPtr>& logical_exprs,
                   ExecContext* context,
                   const std::unordered_set<std::string>& flatten_cadidates =
                       std::unordered_set<std::string>(),
                   bool enable_constant_folding = false);

std::vector<ExprPtr>
CompileInputs(const expr::TypedExprPtr& expr,
              QueryContext* config,
              const std::unordered_set<std::string>& flatten_cadidates);

ExprPtr
CompileExpression(const expr::TypedExprPtr& expr,
                  QueryContext* context,
                  const std::unordered_set<std::string>& flatten_cadidates,
                  bool enable_constant_folding);

class ExprSet {
 public:
    explicit ExprSet(const std::vector<expr::TypedExprPtr>& logical_exprs,
                     ExecContext* exec_ctx) {
        exprs_ = CompileExpressions(logical_exprs, exec_ctx);
    }

    virtual ~ExprSet() = default;

    void
    Eval(EvalCtx& ctx, std::vector<VectorPtr>& results) {
        Eval(0, exprs_.size(), true, ctx, results);
    }

    virtual void
    Eval(int32_t begin,
         int32_t end,
         bool initialize,
         EvalCtx& ctx,
         std::vector<VectorPtr>& result);

    void
    Clear() {
        exprs_.clear();
    }

    ExecContext*
    get_exec_context() const {
        return exec_ctx_;
    }

    size_t
    size() const {
        return exprs_.size();
    }

    const std::vector<std::shared_ptr<Expr>>&
    exprs() const {
        return exprs_;
    }

    const std::shared_ptr<Expr>&
    expr(int32_t index) const {
        return exprs_[index];
    }

 private:
    std::vector<std::shared_ptr<Expr>> exprs_;
    ExecContext* exec_ctx_;
};

template <typename T>
T
GetValueFromProto(const milvus::proto::plan::GenericValue& value_proto) {
    if constexpr (std::is_same_v<T, bool>) {
        Assert(value_proto.val_case() ==
               milvus::proto::plan::GenericValue::kBoolVal);
        return static_cast<T>(value_proto.bool_val());
    } else if constexpr (std::is_integral_v<T>) {
        Assert(value_proto.val_case() ==
               milvus::proto::plan::GenericValue::kInt64Val);
        return static_cast<T>(value_proto.int64_val());
    } else if constexpr (std::is_floating_point_v<T>) {
        Assert(value_proto.val_case() ==
               milvus::proto::plan::GenericValue::kFloatVal);
        return static_cast<T>(value_proto.float_val());
    } else if constexpr (std::is_same_v<T, std::string>) {
        Assert(value_proto.val_case() ==
               milvus::proto::plan::GenericValue::kStringVal);
        return static_cast<T>(value_proto.string_val());
    } else if constexpr (std::is_same_v<T, proto::plan::Array>) {
        Assert(value_proto.val_case() ==
               milvus::proto::plan::GenericValue::kArrayVal);
        return static_cast<T>(value_proto.array_val());
    } else if constexpr (std::is_same_v<T, milvus::proto::plan::GenericValue>) {
        return static_cast<T>(value_proto);
    } else {
        PanicInfo("unsupported generic value type");
    }
};

}  //namespace exec
}  // namespace milvus