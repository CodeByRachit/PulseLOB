#pragma once

#include <cstddef>

namespace pulselob::core {

// Alignment of numeric arrays. One 64-byte line holds 8 doubles, the padding granularity of every
// row in FixedMatrix / FixedVector.
inline constexpr std::size_t kCacheLineBytes = 64;

// Minimum separation between data written by different threads. 128 bytes covers Apple M-series
// cores (128-byte cache lines) and the adjacent-line prefetcher of x86 cores, which pulls lines in
// 128-byte pairs. std::hardware_destructive_interference_size is not used because its value is
// not stable across compilers and flags.
inline constexpr std::size_t kFalseSharingBytes = 128;

inline constexpr std::size_t kDoublesPerLine = kCacheLineBytes / sizeof(double);

constexpr std::size_t round_up(std::size_t n, std::size_t multiple) noexcept {
  return (n + multiple - 1) / multiple * multiple;
}

}  // namespace pulselob::core
