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

#include "HashTable.h"
namespace milvus{
namespace exec {
template<bool nullableKeys>
HashTable<nullableKeys>::HashTable(
    std::vector<std::unique_ptr<VectorHasher>>&& hashers,
    const std::vector<Accumulator>& accumulators)
    : BaseHashTable(std::move(hashers)){
        std::vector<DataType> keyTypes;
        for (auto& hasher : hashers_) {
            keyTypes.push_back(hasher->ChannelDataType());
            if (!VectorHasher::typeSupportValueIds(hasher->ChannelDataType())) {
                hashMode_ = HashMode::kHash;
            }
        }
        rows_ = std::make_unique<RowContainer>(keyTypes, accumulators, nullableKeys, hashMode_ != HashMode::kHash);
    }
}
}