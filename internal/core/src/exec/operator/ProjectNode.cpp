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
#include "ProjectNode.h"
#include "exec/expression/Utils.h"

namespace milvus{
namespace exec {
PhyProjectNode::PhyProjectNode(int32_t operator_id,
                               milvus::exec::DriverContext *ctx,
                               RowTypePtr row_type,
                               const std::shared_ptr<const plan::ProjectNode> &projectNode):
        Operator(ctx, row_type, operator_id, projectNode->id(), "Project"),
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
        return nullptr;
    }
    auto col_input = GetColumnVector(input_);
    TargetBitmapView bitset_view(col_input->GetRawData(), col_input->size());
    auto result_pair = segment_->find_first(0, bitset_view);
    auto selected_offsets = result_pair.first;
    is_finished_ = true;
    for (auto field_id: )
}

};
};