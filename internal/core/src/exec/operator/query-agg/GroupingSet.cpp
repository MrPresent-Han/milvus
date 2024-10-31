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

#include "GroupingSet.h"

namespace milvus{
namespace exec{
GroupingSet::~GroupingSet(){}

void GroupingSet::addInput(const milvus::RowVector &input, bool mayPushDown) {
    auto numRows = input.size();
    numInputRows_ += numRows;
}

void GroupingSet::addGlobalAggregationInput(const milvus::RowVector &input, bool mayPushDown) {

}

std::vector<Accumulator> GroupingSet::accumulators(bool /*excludeToIntermediate*/) {
    std::vector<Accumulator> accumulators;
    accumulators.reserve(aggregates_.size());
    for(auto& aggregate: aggregates_) {
        // add accumalator for each aggregate
        // accumulators.emplace_back(Accumulator{aggregate-});
    }
    return accumulators;
}


void GroupingSet::addInputForActiveRows(const RowVectorPtr& input, bool mayPushdown) {
    AssertInfo(!isGlobal_, "Global aggregations should not reach add input for acitve rows");
    if (!hash_table_) {

    }
}

void initializeAggregates(const std::vector<AggregateInfo>& aggregates, RowContainer& rows) {
    const auto numKeys = rows.KeyTypes().size();
    int i = 0;
    for (auto& aggregate : aggregates) {
        auto& function = aggregate.function_;
        const auto& rowColumn = rows.columnAt(numKeys + i);
    }
}

void GroupingSet::createHashTable(){
    if (ignoreNullKeys_) {
        hash_table_ = std::make_unique<HashTable<true>>(std::move(hashers_));
    } else {
        hash_table_ = std::make_unique<HashTable<false>>(std::move(hashers_));
    }

    auto& rows_container = *(hash_table_->rows());

}  

}
}