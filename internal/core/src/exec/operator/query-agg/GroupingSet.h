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
#include "common/Types.h"
#include "VectorHasher.h"
#include "AggregateInfo.h"

namespace milvus {
namespace exec {

class GroupingSet {
  public:
    GroupingSet(const RowTypePtr& input_type,
                std::vector<std::unique_ptr<VectorHasher>>&& hashers,
                std::vector<AggregateInfo>&& aggregates,
                bool ignoreNullKeys,
                bool isRawInput):
                hashers_(std::move(hashers)),
                isGlobal_(hashers_.empty()),
                isRawInput_(isRawInput),
                aggregates_(std::move(aggregates)),
                ignoreNullKeys_(ignoreNullKeys){}

    ~GroupingSet();

    void addInput(const RowVector& input, bool mayPushDown);

private:
    const bool isGlobal_;
    const bool isRawInput_;
    const bool ignoreNullKeys_;
    std::vector<std::unique_ptr<VectorHasher>> hashers_;
    std::vector<AggregateInfo> aggregates_;
};

}
}
