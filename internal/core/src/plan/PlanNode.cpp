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
#include "PlanNode.h"

namespace milvus {
namespace plan {

RowTypePtr getAggregationOutputType(const std::vector<expr::FieldAccessTypeExprPtr>& groupingKeys,
                                    const std::vector<std::string>& aggregateNames,
                                    const std::vector<AggregationNode::Aggregate>& aggregates) {
    std::vector<std::string> names;
    std::vector<milvus::DataType> types;
    for (auto& key : groupingKeys) {
        types.emplace_back(key->type());
        names.emplace_back(key->name());
    }

    for (int i = 0; i < aggregateNames.size(); i++) {
        names.emplace_back(aggregateNames[i]);
        types.emplace_back(aggregates[i].call_->type());
    }

    return std::make_shared<RowType>(std::move(names), std::move(types));
}

AggregationNode::AggregationNode(const milvus::plan::PlanNodeId &id,
                                 Step step,
                                 std::vector<expr::FieldAccessTypeExprPtr> &&groupingKeys,
                                 std::vector<std::string> &&aggNames,
                                 std::vector<Aggregate> &&aggregates,
                                 RowType &&output_type,
                                 std::vector<PlanNodePtr> sources):
                                 PlanNode(id),
                                 step_(step),
                                 groupingKeys_(std::move(groupingKeys)),
                                 aggregateNames_(std::move(aggNames)),
                                 aggregates_(std::move(aggregates)),
                                 sources_(std::move(sources)),
                                 output_type_(getAggregationOutputType(groupingKeys_, aggregateNames_, aggregates_)),
                                 ignoreNullKeys_(true){}

const char* AggregationNode::stepName(milvus::plan::AggregationNode::Step step) {
    switch (step) {
        case milvus::plan::AggregationNode::Step::kPartial:
            return "PARTIAL";
        case milvus::plan::AggregationNode::Step::kFinal:
            return "FINAL";
        case milvus::plan::AggregationNode::Step::kIntermediate:
            return "INTERMEDIATE";
        case milvus::plan::AggregationNode::Step::kSingle:
            return "SINGLE";
        default:
            return "UNKNOWN";
    }
}

}
}