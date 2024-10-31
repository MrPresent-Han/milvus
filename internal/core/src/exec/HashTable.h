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
#include <vector>
#include <memory>

#include "VectorHasher.h"
#include "RowContainer.h"

namespace milvus{
namespace exec{
class BaseHashTable {
public:
#if XSIMD_WITH_SSE2
        using TagVector = xsimd::batch<uint8_t, xsimd::sse2>;
#elif XSIMD_WITH_NEON
        using TagVector = xsimd::batch<uint8_t, xsimd::neon>;
#endif

enum class HashMode {kHash, kArray, kNormalizedKey};

explicit BaseHashTable(std::vector<std::unique_ptr<VectorHasher>>&& hashers)
        :hashers_(std::move(hashers)){}

private:
  std::vector<std::unique_ptr<VectorHasher>> hashers_;
  std::unique_ptr<RowContainer> rows_;
};

template <bool ignoreNullKeys>
class HashTable : public BaseHashTable {
public:
    HashTable(
        std::vector<std::unique_ptr<VectorHasher>>&& hashers);

private:
  HashMode hashMode_ = HashMode::kArray;

};

struct HashLookup {
  explicit HashLookup(const std::vector<std::unique_ptr<VectorHasher>>& hashers): hashers_(hashers){}

  /// One entry per group-by
  const std::vector<std::unique_ptr<VectorHasher>>& hashers_;
};

}
}