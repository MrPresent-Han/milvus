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
#include "segcore/SegmentInterface.h"
#include "exceptions/EasyAssert.h"

namespace milvus {
namespace exec {

template <typename T, bool lower_inclusive, bool upper_inclusive>
struct BinaryRangeElementFunc {
    typedef std::conditional_t<std::is_integral_v<T> &&
                                   !std::is_same_v<bool, T>,
                               int64_t,
                               T>
        HighPrecisionType;
    void
    operator()(T val1, T val2, const T* src, size_t n, bool* res) {
        for (size_t i = 0; i < n; ++i) {
            if constexpr (lower_inclusive && upper_inclusive) {
                res[i] = val1 <= src[i] && src[i] <= val2;
            } else if constexpr (lower_inclusive && !upper_inclusive) {
                res[i] = val1 <= src[i] && src[i] < val2;
            } else if constexpr (!lower_inclusive && upper_inclusive) {
                res[i] = val1 < src[i] && src[i] <= val2;
            } else {
                res[i] = val1 < src[i] && src[i] < val2;
            }
        }
    }
};

template <typename T>
struct BinaryRangeIndexFunc {
    typedef std::
        conditional_t<std::is_same_v<T, std::string_view>, std::string, T>
            IndexInnerType;
    using Index = index::ScalarIndex<IndexInnerType>;
    typedef std::conditional_t<std::is_integral_v<IndexInnerType> &&
                                   !std::is_same_v<bool, T>,
                               int64_t,
                               IndexInnerType>
        HighPrecisionType;
    FixedVector<bool>
    operator()(Index* index,
               IndexInnerType val1,
               IndexInnerType val2,
               bool lower_inclusive,
               bool upper_inclusive) {
        return index->Range(val1, lower_inclusive, val2, upper_inclusive);
    }
};

class PhyBinaryRangeFilterExpr : public SegmentExpr {
 public:
    PhyBinaryRangeFilterExpr(
        const std::vector<std::shared_ptr<Expr>>& input,
        const std::shared_ptr<const milvus::expr::BinaryRangeFilterExpr>& expr,
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
    template <
        typename T,
        typename IndexInnerType = std::
            conditional_t<std::is_same_v<T, std::string_view>, std::string, T>,
        typename HighPrecisionType = std::conditional_t<
            std::is_integral_v<IndexInnerType> && !std::is_same_v<bool, T>,
            int64_t,
            IndexInnerType>>
    FlatVectorPtr
    PreCheckOverflow(int64_t batch_size,
                     HighPrecisionType& val1,
                     HighPrecisionType& val2,
                     bool& lower_inclusive,
                     bool& upper_inclusive);

    template <typename T>
    VectorPtr
    ExecRangeVisitorImpl();

    template <typename T>
    VectorPtr
    ExecRangeVisitorImplForIndex();

    template <typename T>
    VectorPtr
    ExecRangeVisitorImplForData();

 private:
    std::shared_ptr<const milvus::expr::BinaryRangeFilterExpr> expr_;
};
}  //namespace exec
}  // namespace milvus
