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

#include "TermExpr.h"

namespace milvus {
namespace exec {

void
PhyTermFilterExpr::Eval(EvalCtx& context, VectorPtr& result) {
    switch (expr_->column_.data_type_) {
        case DataType::BOOL: {
            result = ExecVisitorImpl<bool>();
            break;
        }
        case DataType::INT8: {
            result = ExecVisitorImpl<int8_t>();
            break;
        }
        case DataType::INT16: {
            result = ExecVisitorImpl<int16_t>();
            break;
        }
        case DataType::INT32: {
            result = ExecVisitorImpl<int32_t>();
            break;
        }
        case DataType::INT64: {
            result = ExecVisitorImpl<int64_t>();
            break;
        }
        case DataType::FLOAT: {
            result = ExecVisitorImpl<float>();
            break;
        }
        case DataType::DOUBLE: {
            result = ExecVisitorImpl<double>();
            break;
        }
        case DataType::VARCHAR: {
            if (segment_->type() == SegmentType::Growing) {
                result = ExecVisitorImpl<std::string>();
            } else {
                result = ExecVisitorImpl<std::string_view>();
            }
            break;
        }
        case DataType::JSON: {
            auto type = expr_->vals_[0].val_case();
            switch (type) {
                case proto::plan::GenericValue::ValCase::kBoolVal:
                    result = ExecVisitorImplTemplateJson<bool>();
                    break;
                case proto::plan::GenericValue::ValCase::kInt64Val:
                    result = ExecVisitorImplTemplateJson<int64_t>();
                    break;
                case proto::plan::GenericValue::ValCase::kFloatVal:
                    result = ExecVisitorImplTemplateJson<double>();
                    break;
                case proto::plan::GenericValue::ValCase::kStringVal:
                    result = ExecVisitorImplTemplateJson<std::string>();
                    break;
                case proto::plan::GenericValue::ValCase::VAL_NOT_SET:
                    result = ExecVisitorImplTemplateJson<bool>();
                    break;
                default:
                    PanicInfo(fmt::format("unknown data type: {}", int(type)));
            }
        }
        default:
            PanicInfo(fmt::format("unsupported data type: {}",
                                  expr_->column_.data_type_));
    }
}

template <typename ValueType>
VectorPtr
PhyTermFilterExpr::ExecVisitorImplTemplateJson() {
    if (expr_->is_in_field_) {
        return ExecTermJsonVariableInField<ValueType>();
    } else {
        return ExecTermJsonFieldInVariable<ValueType>();
    }
}

template <typename ValueType>
VectorPtr
PhyTermFilterExpr::ExecTermJsonVariableInField() {
    using GetType = std::conditional_t<std::is_same_v<ValueType, std::string>,
                                       std::string_view,
                                       ValueType>;
    auto real_batch_size = GetNextBatchSize();
    if (real_batch_size == 0) {
        return nullptr;
    }
    auto res_vec =
        std::make_shared<FlatVector>(DataType::BOOL, real_batch_size);
    bool* res = (bool*)res_vec->GetRawData();
    AssertInfo(expr_->vals_.size() == 1,
               "element length in json array must be one");
    ValueType val = GetValueFromProto<ValueType>(expr_->vals_[0]);
    auto pointer = milvus::Json::pointer(expr_->column_.nested_path_);

    auto execute_sub_batch = [](const Json* data,
                                const int size,
                                bool* res,
                                const std::string pointer,
                                const ValueType& target_val) {
        auto executor = [&](size_t i) {
            auto doc = data[i].doc();
            auto array = doc.at_pointer(pointer).get_array();
            if (array.error())
                return false;
            for (auto it = array.begin(); it != array.end(); ++it) {
                auto val = (*it).template get<GetType>();
                if (val.error()) {
                    return false;
                }
                if (val.value() == target_val) {
                    return true;
                }
            }
            return false;
        };
        for (size_t i = 0; i < size; ++i) {
            res[i] = executor(i);
        }
    };
    ProcessDataChunks<milvus::Json>(execute_sub_batch, res, pointer, val);
    return res_vec;
}

template <typename ValueType>
VectorPtr
PhyTermFilterExpr::ExecTermJsonFieldInVariable() {
    using GetType = std::conditional_t<std::is_same_v<ValueType, std::string>,
                                       std::string_view,
                                       ValueType>;
    auto real_batch_size = GetNextBatchSize();
    if (real_batch_size == 0) {
        return nullptr;
    }
    auto res_vec =
        std::make_shared<FlatVector>(DataType::BOOL, real_batch_size);
    bool* res = (bool*)res_vec->GetRawData();
    auto pointer = milvus::Json::pointer(expr_->column_.nested_path_);
    std::unordered_set<ValueType> term_set;
    for (const auto& element : expr_->vals_) {
        term_set.insert(GetValueFromProto<ValueType>(element));
    }

    auto execute_sub_batch = [](const Json* data,
                                const int size,
                                bool* res,
                                const std::string pointer,
                                const std::unordered_set<ValueType>& terms) {
        auto executor = [&](size_t i) {
            auto x = data[i].template at<GetType>(pointer);
            if (x.error()) {
                if constexpr (std::is_same_v<GetType, std::int64_t>) {
                    auto x = data[i].template at<double>(pointer);
                    if (x.error()) {
                        return false;
                    }

                    auto value = x.value();
                    // if the term set is {1}, and the value is 1.1, we should not return true.
                    return std::floor(value) == value &&
                           terms.find(ValueType(value)) != terms.end();
                }
                return false;
            }
            return terms.find(ValueType(x.value())) != terms.end();
        };
        for (size_t i = 0; i < size; ++i) {
            res[i] = executor(i);
        }
    };
    ProcessDataChunks<milvus::Json>(execute_sub_batch, res, pointer, term_set);
    return res_vec;
}

template <typename T>
VectorPtr
PhyTermFilterExpr::ExecVisitorImpl() {
    if (is_index_mode_) {
        return ExecVisitorImplForIndex<T>();
    } else {
        return ExecVisitorImplForData<T>();
    }
}

template <typename T>
VectorPtr
PhyTermFilterExpr::ExecVisitorImplForIndex() {
    typedef std::
        conditional_t<std::is_same_v<T, std::string_view>, std::string, T>
            IndexInnerType;
    using Index = index::ScalarIndex<IndexInnerType>;
    auto real_batch_size = GetNextBatchSize();
    if (real_batch_size == 0) {
        return nullptr;
    }

    std::vector<IndexInnerType> vals;
    for (auto& val : expr_->vals_) {
        vals.emplace_back(GetValueFromProto<IndexInnerType>(val));
    }
    auto execute_sub_batch = [](Index* index_ptr,
                                const std::vector<IndexInnerType>& vals) {
        TermIndexFunc<T> func;
        return func(index_ptr, vals.size(), vals.data());
    };
    auto res = ProcessIndexChunks<T>(execute_sub_batch, vals);
    AssertInfo(res.size() == real_batch_size,
               fmt::format("internal error: expr processed rows {} not equal "
                           "expect batch size {}",
                           res.size(),
                           real_batch_size));
    return std::make_shared<FlatVector>(std::move(res));
}

template <>
VectorPtr
PhyTermFilterExpr::ExecVisitorImplForIndex<bool>() {
    using Index = index::ScalarIndex<bool>;

    if (current_index_chunk_ == num_index_chunk_) {
        return nullptr;
    }

    std::vector<uint8_t> vals;
    for (auto& val : expr_->vals_) {
        vals.emplace_back(GetValueFromProto<bool>(val) ? 1 : 0);
    }
    auto execute_sub_batch = [](Index* index_ptr,
                                const std::vector<uint8_t>& vals) {
        TermIndexFunc<bool> func;
        return std::move(func(index_ptr, vals.size(), (bool*)vals.data()));
    };
    auto res = ProcessIndexChunks<bool>(execute_sub_batch, vals);
    return std::make_shared<FlatVector>(std::move(res));
}

template <typename T>
VectorPtr
PhyTermFilterExpr::ExecVisitorImplForData() {
    auto real_batch_size = GetNextBatchSize();
    if (real_batch_size == 0) {
        return nullptr;
    }
    auto res_vec =
        std::make_shared<FlatVector>(DataType::BOOL, real_batch_size);
    bool* res = (bool*)res_vec->GetRawData();
    std::vector<T> vals;
    for (auto& val : expr_->vals_) {
        vals.emplace_back(GetValueFromProto<T>(val));
    }
    std::unordered_set<T> vals_set(vals.begin(), vals.end());
    auto execute_sub_batch = [](const T* data,
                                const int size,
                                bool* res,
                                const std::unordered_set<T>& vals) {
        TermElementFuncSet<T> func;
        for (size_t i = 0; i < size; ++i) {
            res[i] = func(vals, data[i]);
        }
    };
    ProcessDataChunks<T>(execute_sub_batch, res, vals_set);
    return res_vec;
}

}  //namespace exec
}  // namespace milvus
