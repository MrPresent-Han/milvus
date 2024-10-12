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

#include "ProjectNode.h"
#include "exec/expression/Utils.h"

namespace milvus{
namespace exec {
PhyProjectNode::PhyProjectNode(int32_t operator_id,
                               milvus::exec::DriverContext *ctx,
                               const std::shared_ptr<const plan::ProjectNode> &projectNode):
        Operator(ctx, projectNode->output_type(), operator_id, projectNode->id(), "Project"),
        fields_to_project_(projectNode->FieldsToProject()){
    auto exec_context = operator_context_->get_exec_context();
    segment_ = exec_context->get_query_context()->get_segment();
}

void PhyProjectNode::AddInput(milvus::RowVectorPtr &input) {
   input_ = std::move(input);
}

RowVectorPtr
PhyProjectNode::GetOutput() {
    if (is_finished_ ||input_ == nullptr) {
        LOG_INFO("hc==skip running project node");
        return nullptr;
    }
    LOG_INFO("hc==start running project node");
    auto col_input = GetColumnVector(input_);
    // raw data view
    TargetBitmapView raw_data_view(col_input->GetRawData(), col_input->size());
    auto result_pair = segment_->find_first(-1, raw_data_view);
    auto selected_offsets = result_pair.first;
    auto selected_count = selected_offsets.size();
    // valid data view
    TargetBitmapView valid_data_view(col_input->GetValidRawData(), col_input->size());
    LOG_INFO("hc==project_selected_count:{}, col_input_size:{}", selected_count, col_input->size());
    auto row_type = OutputType();
    std::vector<VectorPtr> column_vectors;
    for (int i = 0; i < fields_to_project_.size(); i++) {
        auto column_type = row_type->column_type(i);
        auto field_id = fields_to_project_.at(i);
        LOG_INFO("hc==start to project column_type:{}, field_id:{}, selected_count:{}", column_type, field_id.get(),
                 selected_count);
        auto field_data = projectFieldData(field_id, column_type, selected_offsets.data(), selected_count);
        LOG_INFO("hc==finish project column{}, length:{}", i, field_data->Length());
        auto column_vector = std::make_shared<ColumnVector>(std::move(field_data), std::move(valid_data_view));
        column_vectors.emplace_back(column_vector);
        LOG_INFO("hc==finish project column{}, length:{}", i, column_vector->size());
        /*for(int j = 0; j < selected_count; j++) {
            auto* val = column_vector->RawValueAt(j, GetDataTypeSize(column_type));
            if (column_type == DataType::INT32) {
                LOG_INFO("hc==projected_i:{} val:{}", j, *static_cast<int32_t*>(val));
            }
            if (column_type == DataType::INT16) {
                LOG_INFO("hc==projected_i:{} val:{}", j, *static_cast<int16_t*>(val));
            }
        }*/
    }
    is_finished_ = true;
    auto row_vector = std::make_shared<RowVector>(std::move(column_vectors));
    LOG_INFO("hc==finish project columns:");
    return row_vector;
}

FieldDataPtr
PhyProjectNode::projectFieldData(milvus::FieldId fieldId,
                                 milvus::DataType dataType,
                                 const int64_t *seg_offsets,
                                 int64_t count) const {
    FieldDataPtr ret = nullptr;
    switch(dataType) {
        case milvus::DataType::BOOL: {
            FixedVector<bool> vec(count);
            segment_->bulk_subscript(fieldId, dataType, seg_offsets, count, vec.data());
            ret = std::make_shared<FieldDataImpl<bool, true>>(1, dataType, false, std::move(vec));
            break;
        }
        case milvus::DataType::INT8: {
            FixedVector<int8_t> vec(count);
            segment_->bulk_subscript(fieldId, dataType, seg_offsets, count, vec.data());
            ret = std::make_shared<FieldDataImpl<int8_t, true>>(1, dataType, false, std::move(vec));
            break;
        }
        case milvus::DataType::INT16: {
            FixedVector<int16_t> vec(count);
            segment_->bulk_subscript(fieldId, dataType, seg_offsets, count, vec.data());
            ret = std::make_shared<FieldDataImpl<int16_t, true>>(1, dataType, false, std::move(vec));
            break;
        }
        case milvus::DataType::INT32: {
            FixedVector<int32_t> vec(count);
            segment_->bulk_subscript(fieldId, dataType, seg_offsets, count, vec.data());
            ret = std::make_shared<FieldDataImpl<int32_t, true>>(1, dataType, false, std::move(vec));
            break;
        }
        case milvus::DataType::INT64: {
            FixedVector<int64_t> vec(count);
            segment_->bulk_subscript(fieldId, dataType, seg_offsets, count, vec.data());
            ret = std::make_shared<FieldDataImpl<int64_t, true>>(1, dataType, false, std::move(vec));
            break;
        }
        case milvus::DataType::FLOAT: {
            FixedVector<float> vec(count);
            segment_->bulk_subscript(fieldId, dataType, seg_offsets, count, vec.data());
            ret = std::make_shared<FieldDataImpl<float, true>>(1, dataType, false, std::move(vec));
            break;
        }
        case milvus::DataType::DOUBLE: {
            FixedVector<double> vec(count);
            segment_->bulk_subscript(fieldId, dataType, seg_offsets, count, vec.data());
            ret = std::make_shared<FieldDataImpl<double, true>>(1, dataType, false, std::move(vec));
            break;
        }
        case milvus::DataType::STRING:
        case milvus::DataType::VARCHAR: {
            FixedVector<std::string> vec(count);
            segment_->bulk_subscript(fieldId, dataType, seg_offsets, count, vec.data());
            ret = std::make_shared<FieldDataImpl<std::string, true>>(1, dataType, false, std::move(vec));
            break;
        }
        default: {
            PanicInfo(DataTypeInvalid,
                      fmt::format("unsupported data type {}",
                                  dataType));
        }
    }
    return ret;
}

};
};