#include "pulselob/core/fixed_point.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <optional>

namespace {

using pulselob::core::e8_to_double;
using pulselob::core::parse_decimal_e8;

// Compile-time evaluation is part of the contract (constexpr parsing of constants).
static_assert(parse_decimal_e8("1") == 100'000'000);
static_assert(parse_decimal_e8("0.00000001") == 1);

TEST(FixedPoint, ParsesBinanceSampleStrings) {
  // Strings taken verbatim from the data.binance.vision samples in docs/architecture.md 7.4.
  EXPECT_EQ(parse_decimal_e8("0.05464000"), 5'464'000);    // spot ETHBTC price
  EXPECT_EQ(parse_decimal_e8("0.00250000"), 250'000);      // spot qty
  EXPECT_EQ(parse_decimal_e8("0.14044"), 14'044'000);      // futures TRXUSDT price
  EXPECT_EQ(parse_decimal_e8("2941.0"), 294'100'000'000);  // futures qty
  EXPECT_EQ(parse_decimal_e8("15173.00000000"), 1'517'300'000'000);
  EXPECT_EQ(parse_decimal_e8("413.03404"), 41'303'404'000);
}

TEST(FixedPoint, AcceptsSignsAndPartialForms) {
  EXPECT_EQ(parse_decimal_e8("-1.5"), -150'000'000);
  EXPECT_EQ(parse_decimal_e8("+2"), 200'000'000);
  EXPECT_EQ(parse_decimal_e8(".5"), 50'000'000);
  EXPECT_EQ(parse_decimal_e8("3."), 300'000'000);
  EXPECT_EQ(parse_decimal_e8("0"), 0);
  EXPECT_EQ(parse_decimal_e8("0.054640000000"), 5'464'000);  // trailing zeros past 8 decimals
}

TEST(FixedPoint, RejectsInexactAndMalformedInput) {
  EXPECT_EQ(parse_decimal_e8("0.000000001"), std::nullopt);  // 9th decimal nonzero
  EXPECT_EQ(parse_decimal_e8(""), std::nullopt);
  EXPECT_EQ(parse_decimal_e8("-"), std::nullopt);
  EXPECT_EQ(parse_decimal_e8("."), std::nullopt);
  EXPECT_EQ(parse_decimal_e8("1e5"), std::nullopt);
  EXPECT_EQ(parse_decimal_e8("1.2.3"), std::nullopt);
  EXPECT_EQ(parse_decimal_e8(" 1"), std::nullopt);
  EXPECT_EQ(parse_decimal_e8("nan"), std::nullopt);
}

TEST(FixedPoint, OverflowBoundary) {
  // int64 max = 9223372036854775807 e8 units = 92233720368.54775807
  EXPECT_EQ(parse_decimal_e8("92233720368.54775807"), std::numeric_limits<std::int64_t>::max());
  EXPECT_EQ(parse_decimal_e8("92233720368.54775808"), std::nullopt);
  EXPECT_EQ(parse_decimal_e8("92233720369"), std::nullopt);
  EXPECT_EQ(parse_decimal_e8("99999999999999999999999"), std::nullopt);
}

TEST(FixedPoint, ToDoubleIsCorrectlyRounded) {
  EXPECT_EQ(e8_to_double(5'464'000), 0.05464);
  EXPECT_EQ(e8_to_double(-150'000'000), -1.5);
  EXPECT_EQ(e8_to_double(6'900'012'345'678), 69000.12345678);
}

}  // namespace
