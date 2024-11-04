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
#include "exec/operator/query-agg/RowContainer.h"
#include "xsimd/xsimd.hpp"
#include "common/BitUtil.h"

namespace milvus{
namespace exec{

struct HashLookup {
    explicit HashLookup(const std::vector<std::unique_ptr<VectorHasher>>& hashers): hashers_(hashers){}

    void reset(vector_size_t size){
        rows_.resize(size);
        hashes_.resize(size);
        hits_.resize(size);
        newGroups_.clear();
    }

    /// One entry per group-by
    const std::vector<std::unique_ptr<VectorHasher>>& hashers_;

    /// Set of row numbers of row to probe.
    std::vector<vector_size_t> rows_;

    /// Hashes or value IDs for rows in 'rows'. Not aligned with 'rows'. Index is
    /// the row number.
    std::vector<uint64_t> hashes_;

    /// Contains one entry for each row in 'rows'. Index is the row number.
    /// For groupProbe, a pointer to an existing or new row with matching grouping
    /// keys. For joinProbe, a pointer to the first row with matching keys or null
    /// if no match.
    std::vector<char*> hits_;

    /// For groupProbe, row numbers for which a new entry was inserted (didn't
    /// exist before the groupProbe). Empty for joinProbe.
    std::vector<vector_size_t> newGroups_;

    /// If using valueIds, list of concatenated valueIds. 1:1 with 'hashes'.
    /// Populated by groupProbe and joinProbe.
    std::vector<uint64_t> normalizedKeys_;
};

class BaseHashTable {
public:
#if XSIMD_WITH_SSE2
        using TagVector = xsimd::batch<uint8_t, xsimd::sse2>;
#elif XSIMD_WITH_NEON
        using TagVector = xsimd::batch<uint8_t, xsimd::neon>;
#endif
  using MaskType = uint64_t;

enum class HashMode {kHash, kArray, kNormalizedKey};

explicit BaseHashTable(std::vector<std::unique_ptr<VectorHasher>>&& hashers)
        :hashers_(std::move(hashers)){}

    RowContainer* rows() const {
        return rows_.get();
    }

/// Extracts a 7 bit tag from a hash number. The high bit is always set.
static uint8_t hashTag(uint64_t hash) {
    // This is likely all 0 for small key types (<= 32 bits).  Not an issue
    // because small types have a range that makes them normalized key cases.
    // If there are multiple small type keys, they are mixed which makes them a
    // 64 bit hash.  Normalized keys are mixed before being used as hash
    // numbers.
    return static_cast<uint8_t>(hash >> 38) | 0x80;
}

static TagVector
loadTags(uint8_t* tags, int64_t tagIndex) {
    auto src = tags + tagIndex;
#if XSIMD_WITH_SSE2
    return TagVector(_mm_loadu_si128(reinterpret_cast<__m128i const*>(src)));
#elif XSIMD_WITH_NEON
    return TagVector(vld1q_u8(src));
#endif
}


const std::vector<std::unique_ptr<VectorHasher>>& hashers() const {
    return hashers_;    
}

/// Returns the hash mode. This is needed for the caller to calculate
/// the hash numbers using the appropriate method of the
/// VectorHashers of 'this'.
virtual HashMode hashMode() const = 0;

virtual void setHashMode(HashMode mode, int32_t numNew) = 0;

/// Disables use of array or normalized key hash modes.
void forceGenericHashMode() {
  setHashMode(HashMode::kHash, 0);
}

/// Populates 'hashes' and 'rows' fields in 'lookup' in preparation for
/// 'groupProbe' call. Rehashes the table if necessary. Uses lookup.hashes to
/// decode grouping keys from 'input'. If 'ignoreNullKeys' is true, updates
/// 'rows' to remove entries with null grouping keys. After this call, 'rows'
/// may have no entries selected.
void prepareForGroupProbe(
    HashLookup& lookup,
    const RowVectorPtr& input,
    TargetBitmap &activeRows,
    bool nullableKeys
  );

/// Finds or creates a group for each key in 'lookup'. The keys are
/// returned in 'lookup.hits'.
virtual void
groupProbe(HashLookup& lookup) = 0;

protected:
  std::vector<std::unique_ptr<VectorHasher>> hashers_;
  std::unique_ptr<RowContainer> rows_;
  char** table_ = nullptr;
};

class ProbeState;

template <bool nullableKeys>
class HashTable : public BaseHashTable {
public:
    HashTable(
        std::vector<std::unique_ptr<VectorHasher>>&& hashers,
        const std::vector<Accumulator>& accumulators):
        BaseHashTable(std::move(hashers)){
        std::vector<DataType> keyTypes;
        for (auto& hasher : hashers_) {
            keyTypes.push_back(hasher->ChannelDataType());
            if (!VectorHasher::typeSupportValueIds(hasher->ChannelDataType())) {
                hashMode_ = HashMode::kHash;
            }
        }
        rows_ = std::make_unique<RowContainer>(keyTypes, accumulators, nullableKeys, hashMode_ != HashMode::kHash);
    };

    void setHashMode(HashMode mode, int32_t numNew) override;

    void groupProbe(HashLookup& lookup) override;

    class Bucket {
      private:
        static constexpr uint8_t kPointerSignificantBits = 48;
        static constexpr uint64_t kPointerMask = milvus::lowMask(kPointerSignificantBits);
        static constexpr int32_t kPointerSize = kPointerSignificantBits / 8;

        TagVector tags_;
        char pointers_[sizeof(TagVector) * kPointerSize];
        char padding_[16];
    };

    int64_t bucketOffset(uint64_t hash ) const {
        return hash & bucketOffsetMask_;
    }

private:
  HashMode hashMode_ = HashMode::kArray;
  int64_t bucketOffsetMask_{0};
  int64_t numBuckets_{0};

  HashMode hashMode() const override {
    return hashMode_;
  }
};

}
}