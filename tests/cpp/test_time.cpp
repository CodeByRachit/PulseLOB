#include "pulselob/core/time.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <initializer_list>

namespace {

using pulselob::core::ns_to_seconds;

TEST(Time, DifferencesConvertExactlyAndAbsoluteStampsWouldNot) {
  // math_spec 4.7: absolute epoch nanoseconds lose precision as double; differences do not.
  const std::int64_t t0 = 1'709'251'200'081'000'123;
  const std::int64_t t1 = t0 + 7;
  EXPECT_EQ(ns_to_seconds(t1 - t0), 7e-9);
  EXPECT_NE(static_cast<double>(t1) - static_cast<double>(t0), 7.0)
      << "absolute timestamps are quantized to 256 ns in double";
  EXPECT_EQ(ns_to_seconds(1'000'000'000), 1.0);
  EXPECT_EQ(ns_to_seconds(0), 0.0);
}

TEST(Time, DivisionIsCorrectlyRounded) {
  for (std::int64_t dt : {std::int64_t{1}, std::int64_t{3}, std::int64_t{7}, std::int64_t{999'999'999},
                          std::int64_t{123'456'789'012}}) {
    const double expected = static_cast<double>(dt) / 1e9;
    EXPECT_EQ(ns_to_seconds(dt), expected);
    EXPECT_NEAR(ns_to_seconds(dt) * 1e9, static_cast<double>(dt), std::abs(static_cast<double>(dt)) * 1e-15);
  }
}

}  // namespace
