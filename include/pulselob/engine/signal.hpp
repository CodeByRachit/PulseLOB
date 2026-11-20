#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "pulselob/core/platform.hpp"

namespace pulselob::engine {

namespace signal_flags {
inline constexpr std::uint32_t kWarmingUp = 1U << 0;
inline constexpr std::uint32_t kAskPulled = 1U << 1;
inline constexpr std::uint32_t kBidPulled = 1U << 2;
inline constexpr std::uint32_t kKillSwitch = 1U << 3;
inline constexpr std::uint32_t kFastForgetting = 1U << 4;
inline constexpr std::uint32_t kProjectionActive = 1U << 5;
inline constexpr std::uint32_t kNoQuoteBook = 1U << 6;      // quoting instrument has no book yet
inline constexpr std::uint32_t kStaleParameters = 1U << 7;  // split mode: served parameters lag the stream
}  // namespace signal_flags

namespace detail {
template <std::size_t N>
struct SignalPad {
  std::array<std::uint8_t, N> bytes{};
};
template <>
struct SignalPad<0> {};
}  // namespace detail

// Output record, one per input event (docs/architecture.md Section 7.6). 128 bytes for D = 4.
template <std::size_t D>
struct Signal {
  static constexpr std::size_t kRawBytes = 16 + 8 * D + 72;
  static constexpr std::size_t kBytes = core::round_up(kRawBytes, 64);

  std::int64_t t_ns = 0;           //  0  engine clock of the event that produced it
  std::uint32_t flags = 0;         //  8  signal_flags
  std::uint16_t status = 0;        // 12  low byte: engine Status; high byte: feed::EventCheck
  std::uint16_t reserved = 0;      // 14
  std::array<double, D> lambda{};  // 16  served intensities at t+
  double tau_buy = 0.0;
  double tau_sell = 0.0;
  double criticality = 0.0;        // rho~
  double reservation_shift = 0.0;  // Delta^
  double adverse_ask = 0.0;        // epsilon^+
  double adverse_bid = 0.0;        // epsilon^-
  double sigma2 = 0.0;
  std::int64_t bid_px_e8 = 0;  // rounded down to tick; 0 = not quoted
  std::int64_t ask_px_e8 = 0;  // rounded up to tick; 0 = not quoted
  [[no_unique_address]] detail::SignalPad<kBytes - kRawBytes> pad{};
};

static_assert(sizeof(Signal<4>) == 128, "Signal<4> must be 128 bytes (architecture Section 7.6)");
static_assert(offsetof(Signal<4>, lambda) == 16);
static_assert(offsetof(Signal<4>, tau_buy) == 48);
static_assert(offsetof(Signal<4>, sigma2) == 96);
static_assert(offsetof(Signal<4>, bid_px_e8) == 104);
static_assert(offsetof(Signal<4>, ask_px_e8) == 112);
static_assert(std::is_trivially_copyable_v<Signal<4>>);
static_assert(sizeof(Signal<8>) == Signal<8>::kBytes);

}  // namespace pulselob::engine
