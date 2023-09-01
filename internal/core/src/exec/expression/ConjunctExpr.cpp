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

#include "ConjunctExpr.h"

namespace milvus {
namespace exec {

DataType
PhyConjunctFilterExpr::ResolveType(const std::vector<DataType>& inputs) {
    AssertInfo(
        inputs.size() > 0,
        fmt::format(
            "Conjunct expressions expect at least one argument, received: {}",
            inputs.size()));

    for (const auto& type : inputs) {
        AssertInfo(
            type == DataType::BOOL,
            fmt::format("Conjunct expressions expect BOOLEAN, received: {}",
                        type));
    }
    return DataType::BOOL;
}

static bool
AllTrue(FlatVectorPtr& vec) {
    bool* data = static_cast<bool*>(vec->GetRawData());
    for (int i = 0; i < vec->size(); ++i) {
        if (!data[i]) {
            return false;
        }
    }
    return true;
}

static bool
AllFalse(FlatVectorPtr& vec) {
    bool* data = static_cast<bool*>(vec->GetRawData());
    for (int i = 0; i < vec->size(); ++i) {
        if (data[i]) {
            return false;
        }
    }
    return true;
}

int64_t
PhyConjunctFilterExpr::UpdateResult(FlatVectorPtr& input_result,
                                    EvalCtx& ctx,
                                    FlatVectorPtr& result) {
    if (is_and_) {
        if (AllFalse(input_result)) {
            result = input_result;
            return 0;
        } else {
            ConjunctElementFunc<true> func;
            func(input_result, result);
            return result->size();
        }
    } else {
        if (AllTrue(input_result)) {
            result = input_result;
            return 0;
        } else {
            ConjunctElementFunc<false> func;
            func(input_result, result);
            return result->size();
        }
    }
}

void
PhyConjunctFilterExpr::Eval(EvalCtx& context, VectorPtr& result) {
    for (int i = 0; i < inputs_.size(); ++i) {
        VectorPtr input_result;
        inputs_[i]->Eval(context, input_result);
        if (i == 0) {
            result = std::make_shared<FlatVector>(DataType::BOOL,
                                                  input_result->size());
        }
        auto input_flat_result =
            std::dynamic_pointer_cast<FlatVector>(input_result);
        auto flat_result = std::dynamic_pointer_cast<FlatVector>(result);
        auto active_rows =
            UpdateResult(input_flat_result, context, flat_result);
        if (active_rows == 0) {
            std::cout << "xxxxx1111xxxxx" << std::endl;
            return;
        }
    }
}

}  //namespace exec
}  // namespace milvus
