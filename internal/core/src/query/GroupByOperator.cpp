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
GroupByOperator::GroupBy(
        const std::vector<std::shared_ptr<knowhere::IndexNode::iterator>>& iterators,
        const knowhere::Json& search_conf,
        std::vector<GroupByValueType>& group_by_values) {
    //1. get meta
    auto topK = search_conf[knowhere::meta::TOPK];
    auto group_by_field = search_conf[GROUP_BY_FIELD];
    auto field_id = FieldId(group_by_field);

    //2. prepare data
    //fetch_data_ptr()
    return nullptr;
}

}
}
