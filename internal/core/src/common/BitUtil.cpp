/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "BitUtil.h"

namespace milvus::bits {

namespace {
// Naive implementation that does not rely on BMI2.
void
scatterBitsSimple(int32_t numSource,
                  int32_t numTarget,
                  const char* source,
                  const uint64_t* targetMask,
                  char* target) {
    int64_t from = numSource - 1;
    for (int64_t to = numTarget - 1; to >= 0; to--) {
        bool maskIsSet = bits::isBitSet(targetMask, to);
        bits::setBit(target, to, maskIsSet && bits::isBitSet(source, from));
        from -= maskIsSet ? 1 : 0;
    }
}

// Fetches 'numBits' bits of data, from data starting at lastBit -
// numbits (inclusive) and ending at lastBit (exclusive). 'lastBit' is
// updated to be the bit offset of the lowest returned bit. Successive
// calls will go through 'data' from high to low in consecutive chunks
// of up to 64 bits each.
uint64_t
getBitField(const char* data, int32_t numBits, int32_t& lastBit) {
    int32_t highByte = lastBit / 8;
    int32_t lowByte = (lastBit - numBits) / 8;
    int32_t lowBit = (lastBit - numBits) & 7;
    uint64_t bits =
        *reinterpret_cast<const uint64_t*>(data + lowByte) >> lowBit;
    if (numBits + lowBit > 64) {
        auto fromNextByte = numBits + lowBit - 64;
        uint8_t lastBits = *reinterpret_cast<const uint8_t*>(data + highByte) &
                           bits::lowMask(fromNextByte);
        bits |= static_cast<uint64_t>(lastBits) << (64 - lowBit);
    }
    lastBit -= numBits;
    return bits;
}

// Copy bits backward while the remaining data is still larger than size of T.
template <typename T>
inline void
copyBitsBackwardImpl(uint64_t* bits,
                     uint64_t sourceOffset,
                     uint64_t targetOffset,
                     int64_t& remaining) {
    constexpr int kBits = 8 * sizeof(T);
    for (; remaining >= kBits; remaining -= kBits) {
        T word =
            detail::loadBits<T>(bits, sourceOffset + remaining - kBits, kBits);
        detail::storeBits<T>(
            bits, targetOffset + remaining - kBits, word, kBits);
    }
}

}  // namespace

void
copyBitsBackward(uint64_t* bits,
                 uint64_t sourceOffset,
                 uint64_t targetOffset,
                 uint64_t numBits) {
    int64_t remaining = numBits;
    // Copy using the largest unit first and narrow down to smaller ones.
    copyBitsBackwardImpl<uint64_t>(bits, sourceOffset, targetOffset, remaining);
    copyBitsBackwardImpl<uint32_t>(bits, sourceOffset, targetOffset, remaining);
    copyBitsBackwardImpl<uint16_t>(bits, sourceOffset, targetOffset, remaining);
    copyBitsBackwardImpl<uint8_t>(bits, sourceOffset, targetOffset, remaining);
    if (remaining > 0) {
        uint8_t byte = detail::loadBits<uint8_t>(bits, sourceOffset, remaining);
        detail::storeBits<uint8_t>(bits, targetOffset, byte, remaining);
    }
}

void
toString(const void* bits, int offset, int size, char* out) {
    for (int i = 0; i < size; ++i) {
        out[i] =
            '0' + isBitSet(reinterpret_cast<const uint8_t*>(bits), offset + i);
    }
}

std::string
toString(const void* bits, int offset, int size) {
    std::string ans(size, '\0');
    toString(bits, offset, size, ans.data());
    return ans;
}

}  // namespace milvus::bits
