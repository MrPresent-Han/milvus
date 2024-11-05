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
        tagsInTable_ = BaseHashTable::loadTags(reinterpret_cast<uint8_t *>(table.table_), bucketOffset_);
        hits_ = milvus::toBitMask(tagsInTable_ == wantedTags_);
        if (hits_) {
            loadNextHit<op>(table, firstKey);
        }
    }

    template<Operation op, typename Compare, typename Insert, typename Table>
    inline char* fullProbe(Table& table, int32_t firstKey,
                           Compare compare, Insert insert,
                           int64_t& numTombstones,
                           bool extraCheck) {
        AssertInfo(op == Operation::kInsert, "Only support insert operation for group cases");
        auto* alreadyChecked = group_;
        if (extraCheck) {
            tagsInTable_ = table.loadTags(bucketOffset_);
            hits_ = milvus::toBitMask(tagsInTable_ == wantedTags_);
        }

        const int64_t startBucketOffset = bucketOffset_;
        int64_t insertBucketOffset = -1;
        const auto kEmptyGroup = BaseHashTable::TagVector::broadcast(0);
        const auto kTombstoneGroup = BaseHashTable::TagVector::broadcast(kTombstoneTag);
        for(int64_t numProbedBuckets = 0; numProbedBuckets < table.numBuckets(); ++numProbedBuckets) {
            while(hits_ > 0) {
                loadNextHit<op>(table, firstKey);
                if (!(extraCheck && group_ == alreadyChecked) && compare(group_, row_)) {
                    return group_;
                }
            }
        }

        uint16_t empty = milvus::toBitMask(tagsInTable_ == kEmptyGroup) & kFullMask;
        if (empty > 0) {
            if (op == ProbeState::Operation::kProbe) {
                return nullptr;
            }
            if (indexInTags_ != kNotSet) {
                // We came to the end of the probe without a hit. We replace the first
                // tombstone on the way.
                --numTombstones;
                return insert(row_, insertBucketOffset + indexInTags_);
            }
            auto pos = milvus::getAndClearLastSetBit(empty);
            return insert(row_, bucketOffset_ + pos);
        }
        if (op == Operation::kInsert && indexInTags_ == kNotSet) {
            // We passed through a full group.
            uint16_t tombstones =
                    milvus::toBitMask(tagsInTable_ == kTombstoneGroup) & kFullMask;
            if (tombstones > 0) {
                insertBucketOffset = bucketOffset_;
                indexInTags_ = milvus::getAndClearLastSetBit(tombstones);
            }
        }
        bucketOffset_ = table.nextBucketOffset(bucketOffset_);
        tagsInTable_ = table.loadTags(bucketOffset_);
        hits_ = milvus::toBitMask(tagsInTable_ == wantedTags_);
    }


  private:
    static constexpr uint8_t kNotSet = 0xff;
    template <Operation op, typename Table>
    inline void loadNextHit(Table& table, int32_t firstKey) {
        const int32_t hit = milvus::getAndClearLastSetBit(hits_);
        if (op == Operation::kErase) {
            indexInTags_ = hit;
        }
        group_ = table.row(bucketOffset_, hit);
        __builtin_prefetch(group_ + firstKey);
    }



    char* group_;
    BaseHashTable::TagVector wantedTags_;
    BaseHashTable::TagVector tagsInTable_;
    int32_t row_;
    int64_t bucketOffset_;
    BaseHashTable::MaskType hits_;
    uint8_t indexInTags_ = kNotSet;

};

template<bool nullableKeys>
bool HashTable<nullableKeys>::compareKeys(const char *group,
                                          milvus::exec::HashLookup &lookup,
                                          milvus::vector_size_t row) {
    int32_t numKeys = lookup.hashers_.size();
    int32_t i = 0;
    do {
        auto& hasher = lookup.hashers_[i];
        //if (!rows_->)
    } while(++i < numKeys);
    return true;
}

template <bool nullableKeys>
void HashTable<nullableKeys>::storeKeys(milvus::exec::HashLookup &lookup, milvus::vector_size_t row) {
    for (int32_t i = 0; i < hashers_.size(); i++) {
        auto& hasher = hashers_[i];
        //rows_->store(hasher->decodedVector(), row, lookup.hits[row], i); // NOLINT
    }
}

template<bool nullableKeys>
void HashTable<nullableKeys>::storeRowPointer(uint64_t index, uint64_t hash, char *row) {
    if (hashMode_==HashMode::kArray) {
        reinterpret_cast<char**>(table_)[index] = row;
        return;
    }
    const int64_t bktOffset = bucketOffset(index);
    auto* bucket = bucketAt(bktOffset);
    const auto slotIndex = index & (sizeof(TagVector) - 1);
    bucket->setTag(slotIndex, hashTag(hash));
    bucket->setPointer(slotIndex, row);
}

template <bool nullableKeys>
char* HashTable<nullableKeys>::insertEntry(milvus::exec::HashLookup &lookup, uint64_t index,
                                           milvus::vector_size_t row) {
    char* group = rows_->newRow();
    lookup.hits_[row] = group;
    storeKeys(lookup, row);
    storeRowPointer(index, lookup.hashes_[row], group);
    if (hashMode_ == HashMode::kNormalizedKey) {

    }
    numDistinct_++;
    lookup.newGroups_.push_back(row);
    return group;
}

template<bool nullableKeys>
template<bool isJoin, bool isNormalizedKey>
FOLLY_ALWAYS_INLINE void HashTable<nullableKeys>::fullProbe(HashLookup &lookup,
                                                            ProbeState &state,
                                                            bool extraCheck) {
    constexpr ProbeState::Operation op = isJoin ? ProbeState::Operation::kProbe : ProbeState::Operation::kInsert;
    if constexpr (isNormalizedKey) {

    }
    lookup.hits_[state.row()] = state.fullProbe<op>(*this,
                                                    0,
                                                    [&](char *group, int32_t row){ return compareKeys(group, lookup, row);},
                                                    [&](int32_t row, uint64_t index) {
                                                        return !isJoin? nullptr: insertEntry(lookup, index, row);
                                                    },
                                                    numTombstones_,
                                                    !isJoin && extraCheck);

}

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
    ProbeState state1;
    ProbeState state2;
    ProbeState state3;
    ProbeState state4;
    int32_t probeIdx = 0;
    int32_t numProbes = lookup.rows_.size();
    auto rows = lookup.rows_.data();
    for(; probeIdx + 4 <= numProbes; probeIdx+=4) {
        int32_t row = rows[probeIdx];
        state1.preProbe(*this, lookup.hashes_[row], row);
        row = rows[probeIdx + 1];
        state2.preProbe(*this, lookup.hashes_[row], row);
        row = rows[probeIdx + 2];
        state3.preProbe(*this, lookup.hashes_[row], row);
        row = rows[probeIdx + 3];
        state4.preProbe(*this, lookup.hashes_[row], row);

        state1.firstProbe<ProbeState::Operation::kInsert>(*this, 0);
        state2.firstProbe<ProbeState::Operation::kInsert>(*this, 0);
        state3.firstProbe<ProbeState::Operation::kInsert>(*this, 0);
        state4.firstProbe<ProbeState::Operation::kInsert>(*this, 0);

        fullProbe<false>(lookup, state1, false);
        fullProbe<false>(lookup, state2, false);
        fullProbe<false>(lookup, state3, false);
        fullProbe<false>(lookup, state4, false);
    }
    for(; probeIdx < numProbes; probeIdx++) {
        int32_t row = rows[probeIdx];
        state1.preProbe(*this, lookup.hashes_[row], row);
        state1.firstProbe(*this, 0);
        fullProbe<false>(lookup, state1, false);
    }
}

template <bool nullable>
void HashTable<nullable>::clear(bool freeTable) {

}

template class HashTable<true>;
template class HashTable<false>;
}
}
