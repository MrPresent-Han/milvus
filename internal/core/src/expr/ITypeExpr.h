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
#include <vector>

#include <fmt/core.h>

#include "common/Schema.h"
#include "common/Types.h"
#include "pb/plan.pb.h"

namespace milvus {
namespace expr {

struct ColumnInfo {
    FieldId field_id_;
    DataType data_type_;
    std::vector<std::string> nested_path_;

    ColumnInfo(const proto::plan::ColumnInfo& column_info)
        : field_id_(column_info.field_id()),
          data_type_(static_cast<DataType>(column_info.data_type())),
          nested_path_(column_info.nested_path().begin(),
                       column_info.nested_path().end()) {
    }

    ColumnInfo(FieldId field_id,
               DataType data_type,
               std::vector<std::string> nested_path = {})
        : field_id_(field_id),
          data_type_(data_type),
          nested_path_(std::move(nested_path)) {
    }

    bool
    operator==(const ColumnInfo& other) {
        if (field_id_ != other.field_id_) {
            return false;
        }

        if (data_type_ != other.data_type_) {
            return false;
        }

        for (int i = 0; i < nested_path_.size(); ++i) {
            if (nested_path_[i] != other.nested_path_[i]) {
                return false;
            }
        }

        return true;
    }

    std::string
    ToString() const {
        std::stringstream ss;
        ss << "FieldId:" << field_id_.get() << " data_type:" << int(data_type_)
           << " nested_path:" << milvus::join(nested_path_, ",");
        return ss.str();
    }
};

/** 
 * @brief Base class for all exprs
 * a strongly-typed expression, such as literal, function call, etc...
 */
class ITypeExpr {
 public:
    explicit ITypeExpr(DataType type) : type_(type), inputs_{} {
    }

    ITypeExpr(DataType type,
              std::vector<std::shared_ptr<const ITypeExpr>> inputs)
        : type_(type), inputs_{std::move(inputs)} {
    }

    virtual ~ITypeExpr() = default;

    const std::vector<std::shared_ptr<const ITypeExpr>>&
    inputs() const {
        return inputs_;
    }

    DataType
    type() const {
        return type_;
    }

    virtual std::string
    ToString() const = 0;

    const std::vector<std::shared_ptr<const ITypeExpr>>&
    inputs() {
        return inputs_;
    }

 protected:
    DataType type_;
    std::vector<std::shared_ptr<const ITypeExpr>> inputs_;
};

using TypedExprPtr = std::shared_ptr<const ITypeExpr>;

class InputTypeExpr : public ITypeExpr {
 public:
    InputTypeExpr(DataType type) : ITypeExpr(type) {
    }

    std::string
    ToString() const override {
        return "ROW";
    }
};

using InputTypeExprPtr = std::shared_ptr<const InputTypeExpr>;

class CallTypeExpr : public ITypeExpr {
 public:
    CallTypeExpr(DataType type,
                 const std::vector<TypedExprPtr>& inputs,
                 std::string fun_name)
        : ITypeExpr{type, std::move(inputs)} {
    }

    virtual ~CallTypeExpr() = default;

    virtual const std::string&
    name() const {
        return name_;
    }

    std::string
    ToString() const override {
        std::string str{};
        str += name();
        str += "(";
        for (size_t i = 0; i < inputs_.size(); ++i) {
            if (i != 0) {
                str += ",";
            }
            str += inputs_[i]->ToString();
        }
        str += ")";
        return str;
    }

 private:
    std::string name_;
};

using CallTypeExprPtr = std::shared_ptr<const CallTypeExpr>;

class FieldAccessTypeExpr : public ITypeExpr {
 public:
    FieldAccessTypeExpr(DataType type, const std::string& name)
        : ITypeExpr{type}, name_(name), is_input_column_(true) {
    }

    FieldAccessTypeExpr(DataType type,
                        const TypedExprPtr& input,
                        const std::string& name)
        : ITypeExpr{type, {std::move(input)}}, name_(name) {
        is_input_column_ =
            dynamic_cast<const InputTypeExpr*>(inputs_[0].get()) != nullptr;
    }

    bool
    is_input_column() const {
        return is_input_column_;
    }

    std::string
    ToString() const override {
        if (inputs_.empty()) {
            return fmt::format("{}", name_);
        }

        return fmt::format("{}[{}]", inputs_[0]->ToString(), name_);
    }

 private:
    std::string name_;
    bool is_input_column_;
};

using FieldAccessTypeExprPtr = std::shared_ptr<const FieldAccessTypeExpr>;

/** 
 * @brief Base class for all milvus filter exprs, output type must be BOOL
 * a strongly-typed expression, such as literal, function call, etc...
 */
class ITypeFilterExpr : public ITypeExpr {
 public:
    ITypeFilterExpr() : ITypeExpr(DataType::BOOL) {
    }

    ITypeFilterExpr(std::vector<std::shared_ptr<const ITypeExpr>> inputs)
        : ITypeExpr(DataType::BOOL, std::move(inputs)) {
    }

    virtual ~ITypeFilterExpr() = default;
};

class UnaryRangeFilterExpr : public ITypeFilterExpr {
 public:
    explicit UnaryRangeFilterExpr(const ColumnInfo& column,
                                  proto::plan::OpType op_type,
                                  const proto::plan::GenericValue& val)
        : ITypeFilterExpr(), column_(column), op_type_(op_type), val_(val) {
    }

    std::string
    ToString() const override {
        std::string value;
        val_.SerializeToString(&value);
        std::stringstream ss;
        ss << "columnInfo:" << column_.ToString()
           << " op_type:" << milvus::proto::plan::OpType_Name(op_type_)
           << " val:" << value;
        return ss.str();
    }

 public:
    const ColumnInfo column_;
    const proto::plan::OpType op_type_;
    const proto::plan::GenericValue val_;
};

class AlwaysTrueExpr : public ITypeFilterExpr {
 public:
    explicit AlwaysTrueExpr() {
    }

    std::string
    ToString() const override {
        return "AlwaysTrue expr";
    }
};

class ExistsExpr : public ITypeFilterExpr {
 public:
    explicit ExistsExpr(const ColumnInfo& column)
        : ITypeFilterExpr(), column_(column) {
    }

    std::string
    ToString() const override {
        return "Exists expr";
    }

    const ColumnInfo column_;
};

class LogicalUnaryExpr : public ITypeFilterExpr {
 public:
    enum class OpType { Invalid = 0, LogicalNot = 1 };

    explicit LogicalUnaryExpr(const OpType op_type, TypedExprPtr& child)
        : op_type_(op_type) {
        inputs_.emplace_back(child);
    }

    std::string
    ToString() const override {
        return "";
    }

    const OpType op_type_;
};

class TermFilterExpr : public ITypeFilterExpr {
 public:
    explicit TermFilterExpr(const ColumnInfo& column,
                            const std::vector<proto::plan::GenericValue>& vals,
                            bool is_in_field)
        : ITypeFilterExpr(),
          column_(column),
          vals_(vals),
          is_in_field_(is_in_field) {
    }

    std::string
    ToString() const override {
        std::string values;
        for (size_t i = 0; i < vals_.size(); ++i) {
            std::string value;
            vals_[i].SerializeToString(&value);
            values += value;
        }
        std::stringstream ss;
        ss << "columnInfo:" << column_.ToString() << " val:" << values
           << " is_in_field:" << is_in_field_;
        return ss.str();
    }

 public:
    const ColumnInfo column_;
    const std::vector<proto::plan::GenericValue> vals_;
    const bool is_in_field_;
};

class LogicalBinaryExpr : public ITypeFilterExpr {
 public:
    enum class OpType { Invalid = 0, And = 1, Or = 2, Xor = 3, Minus = 4 };

    explicit LogicalBinaryExpr(OpType op_type,
                               const TypedExprPtr& left,
                               const TypedExprPtr& right)
        : ITypeFilterExpr(), op_type_(op_type) {
        inputs_.emplace_back(left);
        inputs_.emplace_back(right);
    }

    std::string
    GetOpTypeString(OpType op) const {
        switch (op) {
            case OpType::Invalid:
                return "Invalid";
            case OpType::And:
                return "And";
            case OpType::Or:
                return "Or";
            case OpType::Xor:
                return "Xor";
            case OpType::Minus:
                return "Minus";
            default:
                return "Unknown";  // Handle the default case if necessary
        }
    }

    std::string
    ToString() const override {
        return "";
    }

    std::string
    name() const {
        return GetOpTypeString(op_type_);
    }

 public:
    const OpType op_type_;
};

class BinaryRangeFilterExpr : public ITypeFilterExpr {
 public:
    BinaryRangeFilterExpr(const ColumnInfo& column,
                          const proto::plan::GenericValue& lower_value,
                          const proto::plan::GenericValue& upper_value,
                          bool lower_inclusive,
                          bool upper_inclusive)
        : ITypeFilterExpr(),
          column_(column),
          lower_val_(lower_value),
          upper_val_(upper_value),
          lower_inclusive_(lower_inclusive),
          upper_inclusive_(upper_inclusive) {
    }

    std::string
    ToString() const override {
        return "";
    }

    const ColumnInfo column_;
    const proto::plan::GenericValue lower_val_;
    const proto::plan::GenericValue upper_val_;
    const bool lower_inclusive_;
    const bool upper_inclusive_;
};

class BinaryArithOpEvalRangeExpr : public ITypeFilterExpr {
 public:
    BinaryArithOpEvalRangeExpr(const ColumnInfo& column,
                               const proto::plan::OpType op_type,
                               const proto::plan::ArithOpType arith_op_type,
                               const proto::plan::GenericValue right_operand,
                               const proto::plan::GenericValue value)
        : column_(column),
          op_type_(op_type),
          arith_op_type_(arith_op_type),
          right_operand_(right_operand),
          value_(value) {
    }

    std::string
    ToString() const override {
        return "";
    }

 public:
    const ColumnInfo column_;
    const proto::plan::OpType op_type_;
    const proto::plan::ArithOpType arith_op_type_;
    const proto::plan::GenericValue right_operand_;
    const proto::plan::GenericValue value_;
};

class CompareExpr : public ITypeFilterExpr {
 public:
    CompareExpr(const FieldId& left_field,
                const FieldId& right_field,
                DataType left_data_type,
                DataType right_data_type,
                proto::plan::OpType op_type)
        : left_field_id_(left_field),
          right_field_id_(right_field),
          left_data_type_(left_data_type),
          right_data_type_(right_data_type),
          op_type_(op_type) {
    }

    std::string
    ToString() const override {
        return "";
    }

 public:
    const FieldId left_field_id_;
    const FieldId right_field_id_;
    DataType left_data_type_;
    DataType right_data_type_;
    proto::plan::OpType op_type_;
};

class JsonContainsExpr : public ITypeFilterExpr {
 public:
    JsonContainsExpr(ColumnInfo column,
                     ContainsType op,
                     const bool same_type,
                     const std::vector<proto::plan::GenericValue>& vals)
        : column_(column),
          op_(op),
          same_type_(same_type),
          vals_(std::move(vals)) {
    }

    std::string
    ToString() const override {
        return "";
    }

 public:
    const ColumnInfo column_;
    ContainsType op_;
    bool same_type_;
    const std::vector<proto::plan::GenericValue>& vals_;
};
}  // namespace expr
}  // namespace milvus