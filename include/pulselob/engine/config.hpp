#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "pulselob/control/quoting.hpp"
#include "pulselob/core/time.hpp"
#include "pulselob/feed/event_mapper.hpp"
#include "pulselob/hawkes/config.hpp"
#include "pulselob/hawkes/forecast.hpp"
#include "pulselob/hawkes/marks.hpp"

namespace pulselob::engine {

// Complete engine configuration (docs/architecture.md Section 4.1).
template <class Shape>
struct EngineConfig {
  static constexpr std::size_t D = Shape::D;
  hawkes::ModelConfig<Shape> model{};
  hawkes::MarkConfig marks{};
  hawkes::ForecastConfig<D> forecast{};
  control::ControlConfig<D> control{};
  feed::ChannelMap<D> channels{};
  std::uint16_t quote_instrument = 1;           // instrument quoted (its book gives the mid)
  core::DurationNs ooo_tolerance_ns = 100'000;  // epsilon_ooo (Section 4.5)
  core::DurationNs stale_ns = 2'000'000'000;    // T_stale per venue (Section 8.6)
  double reject_rate_eta = 0.01;                // EWMA weight of the input rejection rate
  double reject_rate_max = 0.05;                // kill-switch threshold on that rate
};

// nullptr if valid, otherwise the first violated condition. Checks every component and their
// consistency (for example that the channel directions agree between the map and the forecaster).
template <class Shape>
const char* validate(const EngineConfig<Shape>& c) noexcept {
  if (const char* r = hawkes::validate(c.model)) {
    return r;
  }
  if (const char* r = hawkes::validate(c.marks)) {
    return r;
  }
  if (const char* r = hawkes::validate(c.forecast)) {
    return r;
  }
  if (const char* r = control::validate(c.control, c.forecast.horizon_s)) {
    return r;
  }
  if (const char* r = feed::validate(c.channels)) {
    return r;
  }
  for (std::size_t i = 0; i < Shape::D; ++i) {
    if (c.channels.direction[i] != c.forecast.direction[i]) {
      return "channel directions differ between the channel map and the forecast configuration";
    }
  }
  if (c.quote_instrument >= feed::kMaxInstruments || !c.channels.instruments[c.quote_instrument].present) {
    return "quote instrument missing from the instrument table";
  }
  if (c.ooo_tolerance_ns < 0 || c.stale_ns <= 0) {
    return "out-of-order tolerance must be >= 0 and the staleness limit positive";
  }
  if (!(c.reject_rate_eta > 0.0 && c.reject_rate_eta <= 1.0 && std::isfinite(c.reject_rate_max) &&
        c.reject_rate_max > 0.0)) {
    return "reject-rate parameters out of range";
  }
  return nullptr;
}

// Engine-level counters (docs/architecture.md Section 8); the model's own guards are in
// hawkes::ModelCounters.
struct EngineCounters {
  std::uint64_t inputs = 0;
  std::uint64_t trades = 0;
  std::uint64_t books = 0;
  std::uint64_t heartbeats = 0;
  std::uint64_t statuses = 0;
  std::array<std::uint64_t, 16> invalid{};  // rejected by feed::EventCheck code
  std::uint64_t ooo_clamped = 0;
  std::uint64_t ooo_rejected = 0;
  std::uint64_t unmapped = 0;
  std::uint64_t missing_book_marks = 0;  // trade marked with iota = 0 because its book was missing
  std::uint64_t nonfinite_inputs = 0;    // non-finite inventory or a rejected model step
  std::uint64_t feed_faults = 0;         // Status events (gap, disconnect, ring overflow)
  std::uint64_t ring_overflows = 0;      // split mode: estimator ring full
};

}  // namespace pulselob::engine
