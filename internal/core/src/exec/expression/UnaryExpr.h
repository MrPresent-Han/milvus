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

#include <fmt/core.h>

#include "common/Types.h"
#include "common/Vector.h"
#include "exec/expression/Expr.h"
#include "exceptions/EasyAssert.h"
#include "segcore/SegmentInterface.h"
#include "query/Utils.h"

namespace milvus {
namespace exec {

template <typename T, proto::plan::OpType op>
struct UnaryElementFunc {
    void
    operator()(const T* src, size_t size, T val, bool* res) {
        for (int i = 0; i < size; ++i) {
            if constexpr (op == proto::plan::OpType::Equal) {
                res[i] = src[i] == val;
            } else if constexpr (op == proto::plan::OpType::NotEqual) {
                res[i] = src[i] != val;
            } else if constexpr (op == proto::plan::OpType::GreaterThan) {
                res[i] = src[i] > val;
            } else if constexpr (op == proto::plan::OpType::LessThan) {
                res[i] = src[i] < val;
            } else if constexpr (op == proto::plan::OpType::GreaterEqual) {
                res[i] = src[i] >= val;
            } else if constexpr (op == proto::plan::OpType::LessEqual) {
                res[i] = src[i] <= val;
            } else if constexpr (op == proto::plan::OpType::PrefixMatch) {
                res[i] = Match(src[i], val, proto::plan::OpType::PrefixMatch);
            } else {
                PanicInfo(fmt::format(
                    "unsupported op_type:{} for UnaryElementFunc", int(op)));
            }
        }
    }
};

template <typename T, proto::plan::OpType op>
struct UnaryIndexFunc {
    typedef std::
        conditional_t<std::is_same_v<T, std::string_view>, std::string, T>
            IndexInnerType;
    using Index = index::ScalarIndex<IndexInnerType>;
    FixedVector<bool>
    operator()(Index* index, IndexInnerType val) {
        if constexpr (op == proto::plan::OpType::Equal) {
            return index->In(1, &val);
        } else if constexpr (op == proto::plan::OpType::NotEqual) {
            return index->NotIn(1, &val);
        } else if constexpr (op == proto::plan::OpType::GreaterThan) {
            return index->Range(val, OpType::GreaterThan);
        } else if constexpr (op == proto::plan::OpType::LessThan) {
            return index->Range(val, OpType::LessThan);
        } else if constexpr (op == proto::plan::OpType::GreaterEqual) {
            return index->Range(val, OpType::GreaterEqual);
        } else if constexpr (op == proto::plan::OpType::LessEqual) {
            return index->Range(val, OpType::LessEqual);
        } else if constexpr (op == proto::plan::OpType::PrefixMatch) {
            auto dataset = std::make_unique<Dataset>();
            dataset->Set(milvus::index::OPERATOR_TYPE,
                         proto::plan::OpType::PrefixMatch);
            dataset->Set(milvus::index::PREFIX_VALUE, val);
            return index->Query(std::move(dataset));
        } else {
            PanicInfo(fmt::format("unsupported op_type:{} for UnaryIndexFunc",
                                  int(op)));
        }
    }
};

class PhyUnaryRangeFilterExpr : public SegmentExpr {
 public:
    PhyUnaryRangeFilterExpr(
        const std::vector<std::shared_ptr<Expr>>& input,
        const std::shared_ptr<const milvus::expr::UnaryRangeFilterExpr>& expr,
        const std::string& name,
        const segcore::SegmentInternalInterface* segment,
        Timestamp query_timestamp,
        int64_t batch_size)
        : SegmentExpr(std::move(input),
                      name,
                      segment,
                      expr->column_.field_id_,
                      query_timestamp,
                      batch_size),
          expr_(expr) {
    }

    void
    Eval(EvalCtx& context, VectorPtr& result) override;

 private:
    template <typename T>
    VectorPtr
    ExecRangeVisitorImpl();

    template <typename T>
    VectorPtr
    ExecRangeVisitorImplForIndex();

    template <typename T>
    VectorPtr
    ExecRangeVisitorImplForData();

    template <typename ExprValueType>
    VectorPtr
    ExecUnaryRangeVisitorDispatcherJson();

 private:
    std::shared_ptr<const milvus::expr::UnaryRangeFilterExpr> expr_;
};
}  //namespace exec
}  // namespace milvus
