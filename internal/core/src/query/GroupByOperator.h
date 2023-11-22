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

#include "common/QueryInfo.h"
#include "knowhere/index_node.h"
#include "segcore/SegmentInterface.h"

namespace milvus{
namespace query{
    knowhere::DataSetPtr GroupBy(
            const std::vector<std::shared_ptr<knowhere::IndexNode::iterator>>& iterators,
            const SearchInfo& searchInfo,
            std::vector<GroupByValueType>& group_by_values,
            const segcore::SegmentInternalInterface& segment);

    template <typename T>
    void
    GroupIteratorsByType(const std::vector<std::shared_ptr<knowhere::IndexNode::iterator>> &iterators,
                         const FieldId &field_id,
                         int64_t topK,
                         const segcore::SegmentInternalInterface& segment,
                         std::vector<GroupByValueType> &group_by_values);

    template <typename T>
    void
    GroupIteratorResult(const std::shared_ptr<knowhere::IndexNode::iterator>& iterator,
                        const FieldId& field_id,
                        int64_t topK,
                        const segcore::SegmentInternalInterface& segment,
                        std::vector<GroupByValueType>& group_by_values,
                        std::vector<int64_t>& offsets,
                        std::vector<float>& distances);

}
}
