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

#include "VectorHasher.h"
namespace milvus{
namespace exec {
std::vector<std::unique_ptr<VectorHasher>> createVectorHashers(
        const RowTypePtr& rowType,
        const std::vector<expr::FieldAccessTypeExprPtr>& exprs) {
    std::vector<std::unique_ptr<VectorHasher>> hashers;
    hashers.reserve(exprs.size());
    for (const auto& expr: exprs) {
        auto column_idx = rowType->GetChildIndex(expr->name());
        hashers.emplace_back(VectorHasher::create(expr->type(), column_idx));
    }
    return hashers;
}
}
}
