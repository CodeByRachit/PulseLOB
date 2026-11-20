#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "pulselob/feed/market_event.hpp"
#include "pulselob/synth/hawkes_simulator.hpp"

namespace pulselob::synth {

// Converts channel events into MarketEvents for a two-instrument market: channels 0/1 are
// aggressive buys/sells on instrument 0 (the leading venue, a perpetual), channels 2/3 on
// instrument 1 (the quoting venue, spot). A random-walk mid, sweeps, sizes, a top-of-book update
// after every trade on the traded instrument, and a heartbeat whenever heartbeat_ns pass without
// an event.
struct StreamOptions {
  std::int64_t t0_ns = 1'700'000'000'000'000'000;
  std::array<std::int64_t, 2> tick_e8 = {10'000'000, 1'000'000};                // 0.1 perp, 0.01 spot
  std::array<std::int64_t, 2> mid_e8 = {6'000'000'000'000, 6'000'000'000'000};  // 60000.0
  std::int64_t heartbeat_ns = 100'000'000;
  bool books = true;
  bool heartbeats = true;
  // Constant marks: unit sizes, no sweeps and balanced books, so the engine's mark g is exactly 1
  // and the simulated process is exactly the model the engine estimates (math_spec Section 2.3).
  // Otherwise sizes are log-normal and about 15% of trades sweep 1-3 levels.
  bool constant_marks = false;
  // Deterministic one-way latency per instrument, added to ts_local_ns (the engine clock); the
  // output is then stably re-sorted by ts_local_ns.
  std::array<std::int64_t, 2> latency_ns = {0, 0};
};

inline std::vector<feed::MarketEvent> to_market_events(const std::vector<ChannelEvent>& ev, Rng& rng,
                                                       const StreamOptions& opt = {}) {
  std::vector<feed::MarketEvent> out;
  out.reserve(ev.size() * 2 + 16);
  std::array<std::int64_t, 2> mid = opt.mid_e8;
  std::int64_t last_ns = opt.t0_ns;
  std::uint64_t seq = 1;
  auto push_book = [&](std::size_t inst, std::int64_t t) {
    feed::MarketEvent b{};
    b.ts_exchange_ns = t;
    b.ts_local_ns = t + opt.latency_ns[inst];
    b.px0_e8 = mid[inst] - opt.tick_e8[inst];  // best bid
    b.px1_e8 = mid[inst] + opt.tick_e8[inst];  // best ask
    if (opt.constant_marks) {
      b.qty0_e8 = 100'000'000;
      b.qty1_e8 = 100'000'000;
    } else {
      b.qty0_e8 = static_cast<std::int64_t>((0.2 + 2.0 * rng.uniform()) * 1e8);
      b.qty1_e8 = static_cast<std::int64_t>((0.2 + 2.0 * rng.uniform()) * 1e8);
    }
    b.seq = seq++;
    b.instrument = static_cast<std::uint16_t>(inst);
    b.kind = feed::EventKind::BookTop;
    b.flags = feed::event_flags::make(feed::Side::None, feed::TsResolution::Nanosecond,
                                      feed::event_flags::kSynthetic);
    out.push_back(b);
  };
  if (opt.books) {
    push_book(0, opt.t0_ns);
    push_book(1, opt.t0_ns);
  }
  for (const ChannelEvent& c : ev) {
    const std::int64_t t = opt.t0_ns + static_cast<std::int64_t>(std::llround(c.t_s * 1e9));
    while (opt.heartbeats && t - last_ns > opt.heartbeat_ns) {
      last_ns += opt.heartbeat_ns;
      feed::MarketEvent h{};
      h.ts_exchange_ns = last_ns;
      h.ts_local_ns = last_ns;
      h.kind = feed::EventKind::Heartbeat;
      h.flags = feed::event_flags::kSynthetic;
      out.push_back(h);
    }
    const std::size_t inst = c.channel < 2 ? 0 : 1;
    const bool buy = (c.channel % 2) == 0;
    const std::int64_t tick = opt.tick_e8[inst];
    std::int64_t levels = 0;
    double qty = 1.0;
    if (!opt.constant_marks) {
      levels = rng.uniform() < 0.85 ? 0 : 1 + static_cast<std::int64_t>(rng.uniform() * 3.0);
    }
    feed::MarketEvent e{};
    e.ts_exchange_ns = t;
    e.ts_local_ns = t + opt.latency_ns[inst];
    const std::int64_t first = buy ? mid[inst] + tick : mid[inst] - tick;
    const std::int64_t last = buy ? first + levels * tick : first - levels * tick;
    e.px1_e8 = first;
    e.px0_e8 = last;
    if (!opt.constant_marks) {
      qty = std::exp(-3.0 + 1.2 * rng.normal());
    }
    e.qty0_e8 = std::max<std::int64_t>(1, static_cast<std::int64_t>(qty * 1e8));
    e.qty1_e8 = static_cast<std::int64_t>(qty * static_cast<double>(first));
    e.seq = seq++;
    e.count = 1 + static_cast<std::uint32_t>(levels);
    e.instrument = static_cast<std::uint16_t>(inst);
    e.kind = feed::EventKind::Trade;
    e.flags = feed::event_flags::make(buy ? feed::Side::Buy : feed::Side::Sell,
                                      feed::TsResolution::Nanosecond, feed::event_flags::kSynthetic);
    out.push_back(e);
    if (!opt.constant_marks) {
      mid[inst] += (buy ? 1 : -1) * levels * tick +
                   (rng.uniform() < 0.5 ? -tick : tick) * (rng.uniform() < 0.3 ? 1 : 0);
    }
    if (opt.books) {
      push_book(inst, t);
    }
    last_ns = t;
  }
  if (opt.latency_ns[0] != 0 || opt.latency_ns[1] != 0) {
    std::stable_sort(out.begin(), out.end(), [](const feed::MarketEvent& a, const feed::MarketEvent& b) {
      return a.ts_local_ns < b.ts_local_ns;
    });
  }
  return out;
}

}  // namespace pulselob::synth
