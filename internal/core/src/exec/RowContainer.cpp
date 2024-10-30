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

#include "RowContainer.h"

namespace milvus {
namespace exec {
RowContainer::RowContainer(const std::vector<DataType> &keyTypes,
                           bool nullableKeys,
                           bool hasNormalizedKeys):
                           keyTypes_(keyTypes),
                           nullableKeys_(nullableKeys),
                           hasNormalizedKeys_(hasNormalizedKeys){
    int32_t offset = 0;
    int32_t nullOffset = 0;
    bool isVariableWidth = false;
    for(auto& type: keyTypes_){
        offset +=
    }
}
}
}
