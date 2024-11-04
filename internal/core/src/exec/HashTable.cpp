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
#include <memory>
#include "common/SimdUtil.h"

namespace milvus{
namespace exec {

template<bool nullableKeys>
void HashTable<nullableKeys>::groupProbe(milvus::exec::HashLookup &lookup) {
    if (hashMode_ == HashMode::kArray) {
        //arrayGroupProbe(lookup);
        return;
    }
    //checkSize(lookup.rows.size(), false);
    if (hashMode_ == HashMode::kNormalizedKey) {
        return;
    }

}

template<bool nullableKeys>
void HashTable<nullableKeys>::setHashMode(HashMode mode, int32_t numNew) {
    if (mode == HashMode::kArray) {
        
    } else if (mode == HashMode::kHash) {

    } else if (mode == HashMode::kNormalizedKey) {

    }
}

void BaseHashTable::prepareForGroupProbe(HashLookup& lookup,
    const RowVectorPtr& input,
    TargetBitmap& activeRows,
    bool nullableKeys) {
    auto& hashers = lookup.hashers_;

    if (!nullableKeys) {
        // A null in any of the keys disables the row.
        // deselectRowsWithNulls(hashers, rows);
    }
    //lookup.reset(rows.end());

    const auto mode = hashMode();
    for (auto i = 0; i < hashers.size(); i++) {
        auto& hasher = hashers[i];
        auto column_idx = hasher->ChannelIndex();
        ColumnVectorPtr column_ptr = std::static_pointer_cast<ColumnVector>(input->child(column_idx));
        if (mode == BaseHashTable::HashMode::kHash) {
            hasher->hash(column_ptr, i > 0, lookup.hashes_);
        } else {
           //if (!hasher->computeValueIds(rows, lookup.hashes)) {//hc---computing hash code here?
            //    rehash = true;
            //}
        }
    }      
}

template class HashTable<true>;
template class HashTable<false>;

class ProbeState {
  public:
    enum class Operation {kProbe, kInsert, kErase};
    // Special tag for an erased entry. This counts as occupied for probe and as
    // empty for insert. If a tag word with empties gets an erase, we make the
    // erased tag empty. If the tag word getting the erase has no empties, the
    // erase is marked with a tombstone. A probe always stops with a tag word with
    // empties. Adding an empty to a tag word with no empties would break probes
    // that needed to skip this tag word. This is standard practice for open
    // addressing hash tables. F14 has more sophistication in this but we do not
    // need it here since erase is very rare except spilling and is not expected
    // to change the load factor by much in the expected uses.
    static constexpr uint8_t kTombstoneTag = 0x7f;
    static constexpr uint8_t kEmptyTag = 0x00;
    static constexpr int32_t kFullMask = 0xffff;

    int32_t row() const {
        return row_;
    }

    template <typename Table>
    inline void preProbe(const Table& table, uint64_t hash, int32_t row) {
        row_ = row;
        bucketOffset_ = table.bucketOffset(hash);
        const auto tag = BaseHashTable::hashTag(hash);
        wantedTags_ = BaseHashTable::TagVector::broadcast(tag);
        group_ = nullptr;
        indexInTags_ = kNotSet;
        __builtin_prefetch(
                reinterpret_cast<uint8_t*>(table.table_) + bucketOffset_);
    }

    template <Operation op = Operation::kInsert, typename Table>
    inline void firstProbe(const Table& table, int32_t firstKey) {
        tagsInTable_ = BaseHashTable::loadTags(reinterpret_cast<uint8_t*>(table.table_), bucketOffset_);
        hits_ = milvus::toBitMask(tagsInTable_ == wantedTags_);
        if (hits_) {

        }
    }



  private:
    static constexpr uint8_t kNotSet = 0xff;
    template <Operation op, typename Table>
    inline void loadNextHigt(Table& table, int32_t firstKey) {

    }



    char* group_;
    BaseHashTable::TagVector wantedTags_;
    BaseHashTable::TagVector tagsInTable_;
    int32_t row_;
    int64_t bucketOffset_;
    BaseHashTable::MaskType hits_;
    uint8_t indexInTags_ = kNotSet;

};

}
}