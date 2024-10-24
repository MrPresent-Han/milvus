// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License

#include "common/Types.h"
#include "plan/PlanNode.h"

namespace milvus{
namespace exec{
class Aggregate {
protected:
    explicit Aggregate(DataType result_type): result_type_(result_type){}
private:
    const DataType result_type_;

public:
    DataType resultType() const {
        return result_type_;
    }

    static std::unique_ptr<Aggregate> create(
            const std::string& name,
            plan::AggregationNode::Step step,
            const std::vector<DataType>& argTypes,
            DataType resultType);
};

bool isRawInput(milvus::plan::AggregationNode::Step step) {
    return step == milvus::plan::AggregationNode::Step::kPartial ||
           step == milvus::plan::AggregationNode::Step::kSingle;
}

bool isPartialOutput(milvus::plan::AggregationNode::Step step) {
    return step == milvus::plan::AggregationNode::Step::kPartial ||
           step == milvus::plan::AggregationNode::Step::kIntermediate;
}

}
}
