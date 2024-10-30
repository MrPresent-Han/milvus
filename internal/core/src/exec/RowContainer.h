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
#include "common/Types.h"

namespace milvus {
namespace exec {

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

class RowContainer {
public:
    RowContainer(const std::vector<DataType>& keyTypes,
                 bool nullableKeys,
                 bool hasNormalizedKeys);

private:
    const std::vector<DataType> keyTypes_;
    const bool nullableKeys_;
    const bool hasNormalizedKeys_;
    std::vector<int32_t> offsets_;
    std::vector<int32_t> nullOffsets_;
    std::vector<RowColumn> rowColumns_;
};
}
}

