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

#include "RowContainer.h"
#include "common/BitUtil.h"
#include "common/Vector.h"

namespace milvus {
namespace exec {

RowContainer::RowContainer(const std::vector<DataType> &keyTypes,
                           const std::vector<Accumulator>& accumulators,
                           bool nullableKeys,
                           bool hasNormalizedKeys):
                           keyTypes_(keyTypes),
                           accumulators_(accumulators),
                           nullableKeys_(nullableKeys),
                           hasNormalizedKeys_(hasNormalizedKeys){
    int32_t offset = 0;
    int32_t nullOffset = 0;
    bool isVariableWidth = false;
    for(auto& type: keyTypes_){
        offsets_.push_back(offset);
        offset += GetDataTypeSize(type, 1);
        nullOffsets_.push_back(nullOffset);
        if(nullableKeys_) {
            ++nullOffset;
        }
        isVariableWidth |= IsFixedSizeType(type);
    }
    // Make offset at least sizeof pointer so that there is space for a
    // free list next pointer below the bit at 'freeFlagOffset_'.
    offset = std::max<int32_t>(offset, sizeof(void*));
    const int32_t firstAggregateOffset = offset;
    if (!accumulators.empty()) {
        // This moves nullOffset to the start of the next byte.
        // This is to guarantee the null and initialized bits for an aggregate
        // always appear in the same byte.
        nullOffset = (nullOffset + 7) & -8;
    }
    for (const auto& accumulator: accumulators) {
         // Initialized bit.  Set when the accumulator is initialized.
        nullOffsets_.push_back(nullOffset);
        ++nullOffset;
        // Null bit.
        nullOffsets_.push_back(nullOffset);
        ++nullOffset;
        isVariableWidth |= !accumulator.isFixedSize();
        usesExternalMemory_ |= accumulator.usesExternalMemory();
        alignment_ = combineAlignments(accumulator.alignment(), alignment_);
    }


    // Free flag.
    nullOffsets_.push_back(nullOffset);
    freeFlagOffset_ = nullOffset + firstAggregateOffset * 8;
    ++nullOffset;
    // Add 1 to the last null offset to get the number of bits.
    flagBytes_ = milvus::nBytes(nullOffsets_.back() + 1);
    for (int32_t i = 0; i < nullOffsets_.size(); i++) {
        nullOffsets_[i] += firstAggregateOffset;
    }
    offset += flagBytes_;

    if (isVariableWidth) {
        rowSizeOffset_ = offset;
        offset += sizeof(uint32_t);
    }
    fixedRowSize_ = milvus::roundUp(offset, alignment_);

    // A distinct hash table has no aggregates and if the hash table has
    // no nulls, it may be that there are no null flags.
    if (!nullOffsets_.empty()) {
        // All flags like free and probed flags and null flags for keys and non-keys
        // start as 0. This is also used to mark aggregates as uninitialized on row
        // creation.
        initialNulls_.resize(flagBytes_, 0x0);
    }
    originalNormalizedKeySize_ = hasNormalizedKeys_? 
        milvus::roundUp(sizeof(normalized_key_t), alignment_):0;
    normalizedKeySize_ = originalNormalizedKeySize_;

    for (auto i = 0; i < offsets_.size(); i++){
        rowColumns_.emplace_back(offsets_[i], nullableKeys_?nullOffsets_[i]:RowColumn::kNotNullOffset);
    }
}

char *RowContainer::newRow() {
    return nullptr;
}

Accumulator::Accumulator(
        bool isFixedSize,
        int32_t fixedSize,
        bool useExternalMemory,
        int32_t alignment,
        DataType spillType,
        std::function<void(folly::Range<char**> groups, milvus::VectorPtr& result)>
            spillExtractFunction,
        std::function<void(folly::Range<char**> groups)> destroyFunction):
        isFixedSize_{isFixedSize},
        fixedSize_{fixedSize},
        usesExternalMemory_{useExternalMemory},
        alignment_{alignment},
        spillType_{spillType},
        spillExtractFunction_{std::move(spillExtractFunction)},
        destroyFunction_{std::move(destroyFunction)}{
            
        }
}
}
