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
#include <folly/Range.h>
#include "common/Types.h"
#include "common/Vector.h"
#include "Aggregate.h"

namespace milvus {
namespace exec {

class Accumulator {
public:
    Accumulator(
        bool isFixedSize,
        int32_t fixedSize,
        bool useExternalMemory,
        int32_t alignment,
        DataType spillType,
        std::function<void(folly::Range<char**> groups, VectorPtr& result)>
            spillExtractFunction,
        std::function<void(folly::Range<char**> groups)> destroyFunction);

    explicit Accumulator(Aggregate* aggregate, DataType spillType);

    bool isFixedSize() const {
        return isFixedSize_;
    }

    bool usesExternalMemory() const {
        return usesExternalMemory_;
    }        

    int32_t alignment() const {
        return alignment_;
    }

private:
    const bool isFixedSize_;
    const int32_t fixedSize_;
    const bool usesExternalMemory_;
    const int32_t alignment_;
    const DataType spillType_;
    std::function<void(folly::Range<char**> groups, VectorPtr& result)> spillExtractFunction_;
    std::function<void(folly::Range<char**> groups)> destroyFunction_;
};


/// Packed representation of offset, null byte offset and null mask for
/// a column inside a RowContainer.
class RowColumn {
public:
  /// Used as null offset for a non-null column.
  static constexpr int32_t kNotNullOffset = -1;

  RowColumn(int32_t offset, int32_t nullOffset): packedOffsets_(PackOffsets(offset, nullOffset)) {}

  int32_t offset() const {
      return packedOffsets_ >> 32;
  }

  int32_t nullByte() const {
    return static_cast<uint32_t>(packedOffsets_) >> 8;
  }

  uint8_t nullMask() const {
    return packedOffsets_ & 0xff;
  }

  int32_t initializedByte() const {
    return nullByte();
  }

  int32_t initializedMask() const {
    return nullMask() << 1;
  }


private:

   static uint64_t PackOffsets(int32_t offset, int32_t nullOffset) {
       if (nullOffset == kNotNullOffset) {
           // If the column is not nullable, The low word is 0, meaning
           // that a null check will AND 0 to the 0th byte of the row,
           // which is always false and always safe to do.
           return static_cast<uint64_t>(offset) << 32;
       }
       return (1UL << (nullOffset & 7)) | ((nullOffset & ~7UL) << 5) |
              static_cast<uint64_t>(offset) << 32;
  }

   const uint64_t packedOffsets_;
};

using normalized_key_t = uint64_t;

class RowContainer {
public:
    RowContainer(const std::vector<DataType>& keyTypes,
                 const std::vector<Accumulator>& accumulators,
                 bool nullableKeys,
                 bool hasNormalizedKeys);

    const std::vector<DataType>& KeyTypes() const {
        return keyTypes_;
    }

    const RowColumn& columnAt(int32_t column_idx) const {
        return rowColumns_[column_idx];
    }

    static int32_t combineAlignments(int32_t a, int32_t b){
        AssertInfo(__builtin_popcount(a) == 1, "Alignment can only be power of 2, but got{}", a);
        AssertInfo(__builtin_popcount(b) == 1, "Alignment can only be power of 2, but got{}", b);
        return std::max(a, b);
    }

    int32_t rowSizeOffset() const {
        return rowSizeOffset_;
    }

private:
    const std::vector<DataType> keyTypes_;
    const bool nullableKeys_;
    const bool hasNormalizedKeys_;
    std::vector<int32_t> offsets_;
    std::vector<int32_t> nullOffsets_;
    
    std::vector<RowColumn> rowColumns_;

    // How many bytes do the flags (null, free) occupy.
    int32_t fixedRowSize_;
    int32_t flagBytes_;

    // Bit position of free bit. 
    int32_t freeFlagOffset_ = 0;
    int32_t rowSizeOffset_ = 0;

    int alignment_ = 1;

    // Copied over the null bits of each row on initialization. Keys are
    // not null, aggregates are null.
    std::vector<uint8_t> initialNulls_;
    // Extra bytes to reserve before  each added row for a normalized key. Set to
    // 0 after deciding not to use normalized keys.    
    int originalNormalizedKeySize_;
    int normalizedKeySize_;

    std::vector<Accumulator> accumulators_;

    bool usesExternalMemory_{false};
};
}
}

