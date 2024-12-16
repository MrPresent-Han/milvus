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
#pragma once

#include "SimpleNumericAggregate.h"

namespace milvus {
namespace exec {
class CountAggregate: public SimpleNumericAggregate<bool, int64_t, int64_t> {
    using BaseAggregate = SimpleNumericAggregate<bool, int64_t, int64_t>;
public:
    explicit CountAggregate() : BaseAggregate(DataType::INT64){}

    int32_t accumulatorFixedWidthSize() const override {
        return sizeof(int64_t);
    }

    void extractValues(char** groups, int32_t numGroups, VectorPtr* result) override {
        BaseAggregate::doExtractValues(groups, numGroups, result, [&](char* group){
            return *value<int64_t>(group);
        });
    }

    void addRawInput(char** group, const TargetBitmapView& activeRows,
                     const std::vector<VectorPtr>& input) override {

    }

};

}
}
