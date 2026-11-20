#pragma once

// Test and benchmark support (not part of the public API): a deterministic Ogata-thinning simulator
// of the marked Hawkes model, and a converter from channel events to a realistic MarketEvent stream
// (trades on a perp and a spot instrument, top-of-book updates, heartbeats).

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "pulselob/core/fixed_matrix.hpp"
#include "pulselob/feed/market_event.hpp"

namespace pulselob::testkit {

// Portable random numbers: std::mt19937_64 is fully specified by the standard, unlike the standard
// distributions, so the same seed gives the same stream on every platform.
class Rng {
 public:
  explicit Rng(std::uint64_t seed) : gen_(seed) {}
  double uniform() { return static_cast<double>(gen_() >> 11) * 0x1.0p-53; }              // [0, 1)
  double uniform_pos() { return (static_cast<double>(gen_() >> 11) + 0.5) * 0x1.0p-53; }  // (0, 1)
  double exponential(double rate) { return -std::log(uniform_pos()) / rate; }
  double normal() {
    const double u1 = uniform_pos();
    const double u2 = uniform();
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
  }
  std::uint64_t next() { return gen_(); }

 private:
  std::mt19937_64 gen_;
};

struct ChannelEvent {
  double t_s;
  int channel;
  double g;
};

// Linear multivariate Hawkes process with the kernel bank of Shape, intensities
// lambda_i = mu_i + sum_{j,q} alpha_{ijq} x^q_j, simulated by Ogata thinning with the exact state
// recursion. The upper bound between events uses e^{-b u}(R1 + u R0) <= R1 + R0 / (e b) for
// Erlang-2 bases. Parameters can switch at a given time (regime-shift experiments).
template <class Shape>
class HawkesSimulator {
 public:
  static constexpr std::size_t D = Shape::D;
  static constexpr std::size_t Q = Shape::Q;
  static constexpr std::size_t P = Shape::P;
  using Theta = core::FixedMatrix<D, P>;

  HawkesSimulator(const std::array<double, Q>& beta, const Theta& theta) : beta_(beta), theta_(theta) {}

  void schedule_switch(double t_switch_s, const Theta& theta2) {
    t_switch_ = t_switch_s;
    theta2_ = theta2;
    has_switch_ = true;
  }

  std::vector<ChannelEvent> simulate(double horizon_s, Rng& rng, std::size_t max_events = 50'000'000) {
    std::vector<ChannelEvent> out;
    std::array<std::array<double, 2>, Q * D> R{};  // R[(j*Q+q)] = {R0, R1}
    double t = 0.0;
    const Theta* th = &theta_;
    while (out.size() < max_events) {
      if (has_switch_ && th == &theta_ && t >= t_switch_) {
        th = &theta2_;
      }
      // Upper bound on the total intensity from t onward (valid until the next event).
      double bound = 0.0;
      for (std::size_t i = 0; i < D; ++i) {
        double s = (*th)(i, 0);
        for (std::size_t j = 0; j < D; ++j) {
          for (std::size_t q = 0; q < Q; ++q) {
            const auto& r = R[j * Q + q];
            const double b = beta_[q];
            const double xb =
                Shape::orders[q] == 1 ? b * r[0] : b * b * (r[1] + r[0] / (2.718281828459045 * b));
            s += (*th)(i, Shape::coord(j, q)) * xb;
          }
        }
        bound += s;
      }
      double w = rng.exponential(bound);
      // Do not step over a parameter switch with the old bound.
      if (has_switch_ && th == &theta_ && t + w >= t_switch_) {
        decay(R, t_switch_ - t);
        t = t_switch_;
        continue;
      }
      if (t + w > horizon_s) {
        break;
      }
      decay(R, w);
      t += w;
      std::array<double, D> lam{};
      double total = 0.0;
      for (std::size_t i = 0; i < D; ++i) {
        double s = (*th)(i, 0);
        for (std::size_t j = 0; j < D; ++j) {
          for (std::size_t q = 0; q < Q; ++q) {
            const auto& r = R[j * Q + q];
            const double x = Shape::orders[q] == 1 ? beta_[q] * r[0] : beta_[q] * beta_[q] * r[1];
            s += (*th)(i, Shape::coord(j, q)) * x;
          }
        }
        lam[i] = s;
        total += s;
      }
      const double u = rng.uniform() * bound;
      if (u >= total) {
        continue;  // rejected candidate
      }
      std::size_t ch = 0;
      double acc = lam[0];
      while (u >= acc && ch + 1 < D) {
        ++ch;
        acc += lam[ch];
      }
      out.push_back({t, static_cast<int>(ch), 1.0});
      for (std::size_t q = 0; q < Q; ++q) {
        R[ch * Q + q][0] += 1.0;
      }
    }
    return out;
  }

 private:
  void decay(std::array<std::array<double, 2>, Q * D>& R, double dt) const {
    for (std::size_t j = 0; j < D; ++j) {
      for (std::size_t q = 0; q < Q; ++q) {
        auto& r = R[j * Q + q];
        const double e = std::exp(-beta_[q] * dt);
        if (Shape::orders[q] == 2) {
          r[1] = e * (r[1] + dt * r[0]);
        }
        r[0] *= e;
      }
    }
  }

  std::array<double, Q> beta_;
  Theta theta_;
  Theta theta2_{};
  double t_switch_ = 0.0;
  bool has_switch_ = false;
};

// Converts channel events (channels 0/1 = perp buy/sell on instrument 0, 2/3 = spot buy/sell on
// instrument 1) into MarketEvents with a random-walk mid, sweeps, log-normal sizes, a top-of-book
// update after every trade on the traded instrument, and a heartbeat whenever 100 ms pass without
// an event. Timestamps start at t0_ns.
struct StreamOptions {
  std::int64_t t0_ns = 1'700'000'000'000'000'000;
  std::array<std::int64_t, 2> tick_e8 = {10'000'000, 1'000'000};                // 0.1 perp, 0.01 spot
  std::array<std::int64_t, 2> mid_e8 = {6'000'000'000'000, 6'000'000'000'000};  // 60000.0
  std::int64_t heartbeat_ns = 100'000'000;
  bool books = true;
  bool heartbeats = true;
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
    b.ts_local_ns = t;
    b.px0_e8 = mid[inst] - opt.tick_e8[inst];  // best bid
    b.px1_e8 = mid[inst] + opt.tick_e8[inst];  // best ask
    b.qty0_e8 = static_cast<std::int64_t>((0.2 + 2.0 * rng.uniform()) * 1e8);
    b.qty1_e8 = static_cast<std::int64_t>((0.2 + 2.0 * rng.uniform()) * 1e8);
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
      out.push_back(h);
    }
    const std::size_t inst = c.channel < 2 ? 0 : 1;
    const bool buy = (c.channel % 2) == 0;
    const std::int64_t tick = opt.tick_e8[inst];
    const std::int64_t levels = rng.uniform() < 0.85 ? 0 : 1 + static_cast<std::int64_t>(rng.uniform() * 3.0);
    feed::MarketEvent e{};
    e.ts_exchange_ns = t;
    e.ts_local_ns = t;
    const std::int64_t first = buy ? mid[inst] + tick : mid[inst] - tick;
    const std::int64_t last = buy ? first + levels * tick : first - levels * tick;
    e.px1_e8 = first;
    e.px0_e8 = last;
    const double qty = std::exp(-3.0 + 1.2 * rng.normal());
    e.qty0_e8 = std::max<std::int64_t>(1, static_cast<std::int64_t>(qty * 1e8));
    e.qty1_e8 = static_cast<std::int64_t>(qty * static_cast<double>(first));
    e.seq = seq++;
    e.count = 1 + static_cast<std::uint32_t>(levels);
    e.instrument = static_cast<std::uint16_t>(inst);
    e.kind = feed::EventKind::Trade;
    e.flags = feed::event_flags::make(buy ? feed::Side::Buy : feed::Side::Sell,
                                      feed::TsResolution::Nanosecond, feed::event_flags::kSynthetic);
    out.push_back(e);
    mid[inst] +=
        (buy ? 1 : -1) * levels * tick + (rng.uniform() < 0.5 ? -tick : tick) * (rng.uniform() < 0.3 ? 1 : 0);
    if (opt.books) {
      push_book(inst, t);
    }
    last_ns = t;
  }
  return out;
}

}  // namespace pulselob::testkit
