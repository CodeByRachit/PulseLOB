#pragma once

#include <cstdint>

namespace pulselob::core {

// Absolute timestamps are signed 64-bit nanoseconds since the Unix epoch. They are never converted
// to double: at ~1.7e18 ns they exceed 2^53 and would be quantized to 256 ns (math_spec 4.7).
using TimestampNs = std::int64_t;
using DurationNs = std::int64_t;

// Converts a time *difference* to seconds. Exact conversion to double for |dt| < 2^53 ns
// (about 104 days); division (not multiplication by 1e-9, which is inexact) keeps it correctly
// rounded.
constexpr double ns_to_seconds(DurationNs dt_ns) noexcept {
  return static_cast<double>(dt_ns) / 1e9;
}

}  // namespace pulselob::core
