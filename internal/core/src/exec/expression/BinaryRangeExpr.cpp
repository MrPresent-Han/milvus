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

#include "BinaryRangeExpr.h"

#include "query/Utils.h"

namespace milvus {
namespace exec {

void
PhyBinaryRangeFilterExpr::Eval(EvalCtx& context, VectorPtr& result) {
    switch (expr_->column_.data_type_) {
        case DataType::BOOL: {
            result = ExecRangeVisitorImpl<bool>();
            break;
        }
        case DataType::INT8: {
            result = ExecRangeVisitorImpl<int8_t>();
            break;
        }
        case DataType::INT16: {
            result = ExecRangeVisitorImpl<int16_t>();
            break;
        }
        case DataType::INT32: {
            result = ExecRangeVisitorImpl<int32_t>();
            break;
        }
        case DataType::INT64: {
            result = ExecRangeVisitorImpl<int64_t>();
            break;
        }
        case DataType::FLOAT: {
            result = ExecRangeVisitorImpl<float>();
            break;
        }
        case DataType::DOUBLE: {
            result = ExecRangeVisitorImpl<double>();
            break;
        }
        case DataType::VARCHAR: {
            if (segment_->type() == SegmentType::Growing) {
                result = ExecRangeVisitorImpl<std::string>();
            } else {
                result = ExecRangeVisitorImpl<std::string_view>();
            }
            break;
        }
        case DataType::JSON: {
            break;
        }
        default:
            PanicInfo(fmt::format("unsupported data type: {}",
                                  expr_->column_.data_type_));
    }
}

template <typename T>
VectorPtr
PhyBinaryRangeFilterExpr::ExecRangeVisitorImpl() {
    if (is_index_mode_) {
        return ExecRangeVisitorImplForIndex<T>();
    } else {
        return ExecRangeVisitorImplForData<T>();
    }
}

template <typename T, typename IndexInnerType, typename HighPrecisionType>
FlatVectorPtr
PhyBinaryRangeFilterExpr::PreCheckOverflow(int64_t batch_size,
                                           HighPrecisionType& val1,
                                           HighPrecisionType& val2,
                                           bool& lower_inclusive,
                                           bool& upper_inclusive) {
    lower_inclusive = expr_->lower_inclusive_;
    upper_inclusive = expr_->upper_inclusive_;
    val1 = GetValueFromProto<HighPrecisionType>(expr_->lower_val_);
    val2 = GetValueFromProto<HighPrecisionType>(expr_->upper_val_);

    if constexpr (std::is_integral_v<T> && !std::is_same_v<bool, T>) {
        if (milvus::query::gt_ub<T>(val1)) {
            auto res = std::make_shared<FlatVector>(DataType::BOOL, batch_size);
            return res;
        } else if (milvus::query::lt_lb<T>(val1)) {
            val1 = std::numeric_limits<T>::min();
            lower_inclusive = true;
        }

        if (milvus::query::gt_ub<T>(val2)) {
            val2 = std::numeric_limits<T>::max();
            upper_inclusive = true;
        } else if (milvus::query::lt_lb<T>(val2)) {
            auto res = std::make_shared<FlatVector>(DataType::BOOL, batch_size);
            return res;
        }
    }
    return nullptr;
}

template <typename T>
VectorPtr
PhyBinaryRangeFilterExpr::ExecRangeVisitorImplForIndex() {
    typedef std::
        conditional_t<std::is_same_v<T, std::string_view>, std::string, T>
            IndexInnerType;
    using Index = index::ScalarIndex<IndexInnerType>;
    typedef std::conditional_t<std::is_integral_v<IndexInnerType> &&
                                   !std::is_same_v<bool, T>,
                               int64_t,
                               IndexInnerType>
        HighPrecisionType;
    auto real_batch_size = GetNextBatchSize();
    if (real_batch_size == 0) {
        return nullptr;
    }

    const Index& index = segment_->chunk_scalar_index<IndexInnerType>(
        field_id_, current_index_chunk_++);
    auto* index_ptr = const_cast<Index*>(&index);
    auto batch_size = index_ptr->Count();
    HighPrecisionType val1;
    HighPrecisionType val2;
    bool lower_inclusive = false;
    bool upper_inclusive = false;
    if (auto res = PreCheckOverflow<T>(
            batch_size, val1, val2, lower_inclusive, upper_inclusive)) {
        return res;
    }
    auto execute_sub_batch =
        [lower_inclusive, upper_inclusive](
            Index* index_ptr, HighPrecisionType val1, HighPrecisionType val2) {
            BinaryRangeIndexFunc<T> func;
            return std::move(
                func(index_ptr, val1, val2, lower_inclusive, upper_inclusive));
        };
    auto res = ProcessIndexChunks<T>(execute_sub_batch, val1, val2);
    AssertInfo(res.size() == real_batch_size,
               fmt::format("internal error: expr processed rows {} not equal "
                           "expect batch size {}",
                           res.size(),
                           real_batch_size));
    return std::make_shared<FlatVector>(std::move(res));
}

template <typename T>
VectorPtr
PhyBinaryRangeFilterExpr::ExecRangeVisitorImplForData() {
    typedef std::
        conditional_t<std::is_same_v<T, std::string_view>, std::string, T>
            IndexInnerType;
    using Index = index::ScalarIndex<IndexInnerType>;
    typedef std::conditional_t<std::is_integral_v<IndexInnerType> &&
                                   !std::is_same_v<bool, T>,
                               int64_t,
                               IndexInnerType>
        HighPrecisionType;
    auto real_batch_size = GetNextBatchSize();
    if (real_batch_size == 0) {
        return nullptr;
    }

    HighPrecisionType val1;
    HighPrecisionType val2;
    bool lower_inclusive = false;
    bool upper_inclusive = false;
    if (auto res = PreCheckOverflow<T>(
            real_batch_size, val1, val2, lower_inclusive, upper_inclusive)) {
        return res;
    }
    auto res_vec =
        std::make_shared<FlatVector>(DataType::BOOL, real_batch_size);
    bool* res = (bool*)res_vec->GetRawData();
    auto execute_sub_batch = [lower_inclusive, upper_inclusive](
                                 const T* data,
                                 const int size,
                                 bool* res,
                                 HighPrecisionType val1,
                                 HighPrecisionType val2) {
        if (lower_inclusive && upper_inclusive) {
            BinaryRangeElementFunc<T, true, true> func;
            func(val1, val2, data, size, res);
        } else if (lower_inclusive && !upper_inclusive) {
            BinaryRangeElementFunc<T, true, false> func;
            func(val1, val2, data, size, res);
        } else if (!lower_inclusive && upper_inclusive) {
            BinaryRangeElementFunc<T, false, true> func;
            func(val1, val2, data, size, res);
        } else {
            BinaryRangeElementFunc<T, false, false> func;
            func(val1, val2, data, size, res);
        }
    };
    ProcessDataChunks<T>(execute_sub_batch, res, val1, val2);
    return res_vec;
}

}  //namespace exec
}  // namespace milvus
