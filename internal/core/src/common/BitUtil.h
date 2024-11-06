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

#pragma once
#include <cstdint>

namespace milvus {
namespace bits {
template <typename T, typename U>
constexpr T roundUp(T value, U factor) {
    return (value + (factor - 1)) / factor * factor;
}

constexpr uint64_t nBytes(int32_t value) {
    return roundUp(value, 8) / 8;
}

constexpr inline uint64_t lowMask(int32_t bits){
    return (1UL << bits) - 1;
}

inline int32_t getAndClearLastSetBit(uint16_t& bits) {
    int32_t trailingZeros = __builtin_ctz(bits);
    bits &= bits - 1;
    return trailingZeros;
}

// This is the Hash128to64 function from Google's cityhash (available
// under the MIT License).  We use it to reduce multiple 64 bit hashes
// into a single hash.
#if defined(FOLLY_DISABLE_UNDEFINED_BEHAVIOR_SANITIZER)
        FOLLY_DISABLE_UNDEFINED_BEHAVIOR_SANITIZER("unsigned-integer-overflow")
#endif
inline uint64_t hashMix(const uint64_t upper, const uint64_t lower) noexcept{
    // Murmur-inspired hashing.
    const uint64_t kMul = 0x9ddfea08eb382d69ULL;
    uint64_t a = (lower ^ upper) * kMul;
    a ^= (a >> 47);
    uint64_t b = (upper ^ a) * kMul;
    b ^= (b >> 47);
    b *= kMul;
    return b;
}

/// Extract bits from integer 'a' at the corresponding bit locations specified
/// by 'mask' to contiguous low bits in return value; the remaining upper bits
/// in return value are set to zero.
template <typename T>
inline T extractBits(T a, T mask);

#ifdef __BMI2__
        template <>
inline uint32_t extractBits(uint32_t a, uint32_t mask) {
  return _pext_u32(a, mask);
}
template <>
inline uint64_t extractBits(uint64_t a, uint64_t mask) {
  return _pext_u64(a, mask);
}
#else
template <typename T>
T extractBits(T a, T mask) {
    constexpr int kBitsCount = 8 * sizeof(T);
    T dst = 0;
    for (int i = 0, k = 0; i < kBitsCount; ++i) {
        if (mask & 1) {
            dst |= ((a & 1) << k);
            ++k;
        }
        a >>= 1;
        mask >>= 1;
    }
    return dst;
}
#endif
}
}