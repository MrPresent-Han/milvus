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

#include "BinaryArithOpEvalRangeExpr.h"

namespace milvus {
namespace exec {

void
PhyBinaryArithOpEvalRangeExpr::Eval(EvalCtx& context, VectorPtr& result) {
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
PhyBinaryArithOpEvalRangeExpr::ExecRangeVisitorImpl() {
    if (is_index_mode_) {
        return ExecRangeVisitorImplForIndex<T>();
    } else {
        return ExecRangeVisitorImplForData<T>();
    }
}

template <typename T>
VectorPtr
PhyBinaryArithOpEvalRangeExpr::ExecRangeVisitorImplForIndex() {
    using Index = index::ScalarIndex<T>;
    typedef std::conditional_t<std::is_integral_v<T> &&
                                   !std::is_same_v<bool, T>,
                               int64_t,
                               T>
        HighPrecisionType;
    auto real_batch_size = GetNextBatchSize();
    if (real_batch_size == 0) {
        return nullptr;
    }
    auto value = GetValueFromProto<HighPrecisionType>(expr_->value_);
    auto right_operand =
        GetValueFromProto<HighPrecisionType>(expr_->right_operand_);
    auto op_type = expr_->op_type_;
    auto arith_type = expr_->arith_op_type_;
    auto sub_batch_size = size_per_chunk_;

    auto execute_sub_batch = [op_type, arith_type, sub_batch_size](
                                 Index* index_ptr,
                                 HighPrecisionType value,
                                 HighPrecisionType right_operand) {
        FixedVector<bool> res;
        switch (op_type) {
            case proto::plan::OpType::Equal: {
                switch (arith_type) {
                    case proto::plan::ArithOpType::Add: {
                        ArithOpIndexFunc<T,
                                         proto::plan::OpType::Equal,
                                         proto::plan::ArithOpType::Add>
                            func;
                        res = std::move(func(
                            index_ptr, sub_batch_size, value, right_operand));
                        break;
                    }
                    case proto::plan::ArithOpType::Sub: {
                        ArithOpIndexFunc<T,
                                         proto::plan::OpType::Equal,
                                         proto::plan::ArithOpType::Sub>
                            func;
                        res = std::move(func(
                            index_ptr, sub_batch_size, value, right_operand));
                        break;
                    }
                    case proto::plan::ArithOpType::Mul: {
                        ArithOpIndexFunc<T,
                                         proto::plan::OpType::Equal,
                                         proto::plan::ArithOpType::Mul>
                            func;
                        res = std::move(func(
                            index_ptr, sub_batch_size, value, right_operand));
                        break;
                    }
                    case proto::plan::ArithOpType::Div: {
                        ArithOpIndexFunc<T,
                                         proto::plan::OpType::Equal,
                                         proto::plan::ArithOpType::Div>
                            func;
                        res = std::move(func(
                            index_ptr, sub_batch_size, value, right_operand));
                        break;
                    }
                    case proto::plan::ArithOpType::Mod: {
                        ArithOpIndexFunc<T,
                                         proto::plan::OpType::Equal,
                                         proto::plan::ArithOpType::Mod>
                            func;
                        res = std::move(func(
                            index_ptr, sub_batch_size, value, right_operand));
                        break;
                    }
                    default:
                        PanicInfo(
                            fmt::format("unsupported arith type for binary "
                                        "arithmetic eval expr: {}",
                                        int(arith_type)));
                }
                break;
            }
            case proto::plan::OpType::NotEqual: {
                switch (arith_type) {
                    case proto::plan::ArithOpType::Add: {
                        ArithOpIndexFunc<T,
                                         proto::plan::OpType::NotEqual,
                                         proto::plan::ArithOpType::Add>
                            func;
                        res = std::move(func(
                            index_ptr, sub_batch_size, value, right_operand));
                        break;
                    }
                    case proto::plan::ArithOpType::Sub: {
                        ArithOpIndexFunc<T,
                                         proto::plan::OpType::NotEqual,
                                         proto::plan::ArithOpType::Sub>
                            func;
                        res = std::move(func(
                            index_ptr, sub_batch_size, value, right_operand));
                        break;
                    }
                    case proto::plan::ArithOpType::Mul: {
                        ArithOpIndexFunc<T,
                                         proto::plan::OpType::NotEqual,
                                         proto::plan::ArithOpType::Mul>
                            func;
                        res = std::move(func(
                            index_ptr, sub_batch_size, value, right_operand));
                        break;
                    }
                    case proto::plan::ArithOpType::Div: {
                        ArithOpIndexFunc<T,
                                         proto::plan::OpType::NotEqual,
                                         proto::plan::ArithOpType::Div>
                            func;
                        res = std::move(func(
                            index_ptr, sub_batch_size, value, right_operand));
                        break;
                    }
                    case proto::plan::ArithOpType::Mod: {
                        ArithOpIndexFunc<T,
                                         proto::plan::OpType::NotEqual,
                                         proto::plan::ArithOpType::Mod>
                            func;
                        res = std::move(func(
                            index_ptr, sub_batch_size, value, right_operand));
                        break;
                    }
                    default:
                        PanicInfo(
                            fmt::format("unsupported arith type for binary "
                                        "arithmetic eval expr: {}",
                                        int(arith_type)));
                }
                break;
            }
            default:
                PanicInfo(
                    fmt::format("unsupported operator type for binary "
                                "arithmetic eval expr: {}",
                                int(op_type)));
        }
        return res;
    };
    auto res = ProcessIndexChunks<T>(execute_sub_batch, value, right_operand);
    AssertInfo(res.size() == real_batch_size,
               fmt::format("internal error: expr processed rows {} not equal "
                           "expect batch size {}",
                           res.size(),
                           real_batch_size));
    return std::make_shared<FlatVector>(std::move(res));
}

template <typename T>
VectorPtr
PhyBinaryArithOpEvalRangeExpr::ExecRangeVisitorImplForData() {
    typedef std::conditional_t<std::is_integral_v<T> &&
                                   !std::is_same_v<bool, T>,
                               int64_t,
                               T>
        HighPrecisionType;
    auto real_batch_size = GetNextBatchSize();
    if (real_batch_size == 0) {
        return nullptr;
    }

    auto value = GetValueFromProto<HighPrecisionType>(expr_->value_);
    auto right_operand =
        GetValueFromProto<HighPrecisionType>(expr_->right_operand_);
    auto res_vec =
        std::make_shared<FlatVector>(DataType::BOOL, real_batch_size);
    bool* res = (bool*)res_vec->GetRawData();

    auto op_type = expr_->op_type_;
    auto arith_type = expr_->arith_op_type_;
    auto execute_sub_batch = [op_type, arith_type](
                                 const T* data,
                                 const int size,
                                 bool* res,
                                 HighPrecisionType value,
                                 HighPrecisionType right_operand) {
        switch (op_type) {
            case proto::plan::OpType::Equal: {
                switch (arith_type) {
                    case proto::plan::ArithOpType::Add: {
                        ArithOpElementFunc<T,
                                           proto::plan::OpType::Equal,
                                           proto::plan::ArithOpType::Add>
                            func;
                        func(data, size, value, right_operand, res);
                        break;
                    }
                    case proto::plan::ArithOpType::Sub: {
                        ArithOpElementFunc<T,
                                           proto::plan::OpType::Equal,
                                           proto::plan::ArithOpType::Sub>
                            func;
                        func(data, size, value, right_operand, res);
                        break;
                    }
                    case proto::plan::ArithOpType::Mul: {
                        ArithOpElementFunc<T,
                                           proto::plan::OpType::Equal,
                                           proto::plan::ArithOpType::Mul>
                            func;
                        func(data, size, value, right_operand, res);
                        break;
                    }
                    case proto::plan::ArithOpType::Div: {
                        ArithOpElementFunc<T,
                                           proto::plan::OpType::Equal,
                                           proto::plan::ArithOpType::Div>
                            func;
                        func(data, size, value, right_operand, res);
                        break;
                    }
                    case proto::plan::ArithOpType::Mod: {
                        ArithOpElementFunc<T,
                                           proto::plan::OpType::Equal,
                                           proto::plan::ArithOpType::Mod>
                            func;
                        func(data, size, value, right_operand, res);
                        break;
                    }
                    default:
                        PanicInfo(
                            fmt::format("unsupported arith type for binary "
                                        "arithmetic eval expr: {}",
                                        int(arith_type)));
                }
                break;
            }
            case proto::plan::OpType::NotEqual: {
                switch (arith_type) {
                    case proto::plan::ArithOpType::Add: {
                        ArithOpElementFunc<T,
                                           proto::plan::OpType::NotEqual,
                                           proto::plan::ArithOpType::Add>
                            func;
                        func(data, size, value, right_operand, res);
                        break;
                    }
                    case proto::plan::ArithOpType::Sub: {
                        ArithOpElementFunc<T,
                                           proto::plan::OpType::NotEqual,
                                           proto::plan::ArithOpType::Sub>
                            func;
                        func(data, size, value, right_operand, res);
                        break;
                    }
                    case proto::plan::ArithOpType::Mul: {
                        ArithOpElementFunc<T,
                                           proto::plan::OpType::NotEqual,
                                           proto::plan::ArithOpType::Mul>
                            func;
                        func(data, size, value, right_operand, res);
                        break;
                    }
                    case proto::plan::ArithOpType::Div: {
                        ArithOpElementFunc<T,
                                           proto::plan::OpType::NotEqual,
                                           proto::plan::ArithOpType::Div>
                            func;
                        func(data, size, value, right_operand, res);
                        break;
                    }
                    case proto::plan::ArithOpType::Mod: {
                        ArithOpElementFunc<T,
                                           proto::plan::OpType::NotEqual,
                                           proto::plan::ArithOpType::Mod>
                            func;
                        func(data, size, value, right_operand, res);
                        break;
                    }
                    default:
                        PanicInfo(
                            fmt::format("unsupported arith type for binary "
                                        "arithmetic eval expr: {}",
                                        int(arith_type)));
                }
                break;
            }
            default:
                PanicInfo(
                    fmt::format("unsupported operator type for binary "
                                "arithmetic eval expr: {}",
                                int(op_type)));
        }
    };
    ProcessDataChunks<T>(execute_sub_batch, res, value, right_operand);
    return res_vec;
}

}  //namespace exec
}  // namespace milvus
