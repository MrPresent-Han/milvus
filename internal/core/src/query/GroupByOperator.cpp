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
#include "GroupByOperator.h"
#include "common/Consts.h"

namespace milvus{
namespace query{

knowhere::DataSetPtr
GroupBy(
        const std::vector<std::shared_ptr<knowhere::IndexNode::iterator>>& iterators,
        const SearchInfo& search_info,
        std::vector<GroupByValueType>& group_by_values,
        const segcore::SegmentInternalInterface& segment) {
    //1. get meta
    FieldId group_by_field_id = search_info.group_by_field_id_.value();
    auto data_type = segment.GetFieldDataType(group_by_field_id);


    switch(data_type){
        case DataType::BOOL:{
            GroupIteratorsByType<bool>(iterators, group_by_field_id, search_info.topk_, segment, group_by_values);
            break;
        }
        case DataType::INT8:{
            GroupIteratorsByType<int8_t>(iterators, group_by_field_id, search_info.topk_, segment, group_by_values);
            break;
        }
        case DataType::INT16:{
            GroupIteratorsByType<int16_t>(iterators, group_by_field_id, search_info.topk_, segment, group_by_values);
            break;
        }
        default:{
            PanicInfo(DataTypeInvalid,
                      fmt::format("unsupported data type {} for group by operator", data_type));
        }
    }
    return nullptr;
}

template <typename T>
void
GroupIteratorsByType(const std::vector<std::shared_ptr<knowhere::IndexNode::iterator>> &iterators,
                     const FieldId &field_id,
                     int64_t topK,
                     const segcore::SegmentInternalInterface& segment,
                     std::vector<GroupByValueType> &group_by_values){
    std::vector<int64_t> offsets;
    std::vector<float> distances;
    for(auto& iterator: iterators){
        GroupIteratorResult<T>(iterator, field_id, topK, segment, group_by_values, offsets, distances);
    }
}

template<typename T>
void
GroupIteratorResult(const std::shared_ptr<knowhere::IndexNode::iterator>& iterator,
                    const FieldId& field_id,
                    int64_t topK,
                    const segcore::SegmentInternalInterface& segment,
                    std::vector<GroupByValueType>& group_by_values,
                    std::vector<int64_t>& offsets,
                    std::vector<float>& distances){
    

}


}
}
