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
#include "OperatorUtils.h"

namespace milvus{
namespace exec {
void deselectRowsWithNulls(const std::vector<std::unique_ptr<VectorHasher>>& hashers,
                           const RowVectorPtr& input,
                           TargetBitmap& activeRows){
    for(auto i = 0; i < hashers.size(); i++){
        auto column_idx = hashers[i]->ChannelIndex();
        ColumnVectorPtr column_ptr = std::dynamic_pointer_cast<ColumnVector>(input->child(column_idx));
        AssertInfo(column_ptr!=nullptr, "Failed to get column vector from row vector input");
        int64_t length = column_ptr->size();
        TargetBitmapView valid_bits_view(column_ptr->GetValidRawData(), length);
        activeRows&=valid_bits_view;
    }
}
}
}
