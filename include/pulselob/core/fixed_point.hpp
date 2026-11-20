#pragma once

#include <cstdint>
#include <limits>
#include <optional>
#include <string_view>

namespace pulselob::core {

// Prices and quantities are stored as signed 64-bit integers in units of 1e-8 ("e8").
// Every price and quantity Binance publishes has at most 8 decimals, so the encoding is exact,
// and the range (9.2e10 currency units) covers every listed instrument.
inline constexpr std::int64_t kE8 = 100'000'000;
inline constexpr int kE8Decimals = 8;

// Parses a plain decimal string ("[+-]digits[.digits]") into e8 units. Returns nullopt for
// malformed input, overflow, or a nonzero digit beyond the 8th decimal (which e8 cannot represent
// exactly). Trailing zeros beyond the 8th decimal are accepted ("0.054640000").
constexpr std::optional<std::int64_t> parse_decimal_e8(std::string_view s) noexcept {
  if (s.empty()) {
    return std::nullopt;
  }
  std::size_t i = 0;
  bool negative = false;
  if (s[0] == '+' || s[0] == '-') {
    negative = s[0] == '-';
    i = 1;
  }
  constexpr std::uint64_t kMax = static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
  std::uint64_t value = 0;
  std::size_t digits = 0;
  for (; i < s.size() && s[i] != '.'; ++i) {
    const char c = s[i];
    if (c < '0' || c > '9') {
      return std::nullopt;
    }
    const auto d = static_cast<std::uint64_t>(c - '0');
    if (value > (kMax - d) / 10) {
      return std::nullopt;
    }
    value = value * 10 + d;
    ++digits;
  }
  int decimals = 0;
  if (i < s.size()) {  // at '.'
    ++i;
    for (; i < s.size(); ++i) {
      const char c = s[i];
      if (c < '0' || c > '9') {
        return std::nullopt;
      }
      ++digits;
      if (decimals == kE8Decimals) {
        if (c != '0') {
          return std::nullopt;
        }
        continue;
      }
      const auto d = static_cast<std::uint64_t>(c - '0');
      if (value > (kMax - d) / 10) {
        return std::nullopt;
      }
      value = value * 10 + d;
      ++decimals;
    }
  }
  if (digits == 0) {
    return std::nullopt;
  }
  for (; decimals < kE8Decimals; ++decimals) {
    if (value > kMax / 10) {
      return std::nullopt;
    }
    value *= 10;
  }
  const auto magnitude = static_cast<std::int64_t>(value);
  return negative ? -magnitude : magnitude;
}

// Model-side conversion. Exact for |v| < 2^53 e8 units (9.0e7 currency units).
constexpr double e8_to_double(std::int64_t v) noexcept {
  return static_cast<double>(v) / 1e8;
}

}  // namespace pulselob::core
