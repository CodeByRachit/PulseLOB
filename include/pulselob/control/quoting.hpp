#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "pulselob/core/time.hpp"
#include "pulselob/hawkes/forecast.hpp"

namespace pulselob::control {

// Parameters of the Hawkes-adjusted Avellaneda-Stoikov law (math_spec Section 8). Prices and impacts
// are in quote currency; gamma and k in 1/currency; inventory in units of the quoted asset.
template <std::size_t D>
struct ControlConfig {
  double gamma = 0.1;                     // CARA risk aversion
  double k = 1.5;                         // fill-probability decay e^{-k delta}
  double horizon_inv_s = 1.0;             // T_h, inventory-risk horizon
  double sigma_w2 = 0.0;                  // diffusive variance rate sigma_W^2 (currency^2 / s)
  double delta_max = 1.0;                 // largest useful quoting distance
  double q_max = 10.0;                    // inventory limit
  std::array<double, D> jump_benign{};    // J^b_i
  std::array<double, D> jump_informed{};  // J^I_i
  std::size_t buy_channel = 0;            // V+: aggressive buys on the quoting venue (hit our ask)
  std::size_t sell_channel = 0;           // V-: aggressive sells on the quoting venue (hit our bid)
  double hysteresis = 0.1;                // Delta_tau
  double cooldown_s = 0.25;               // c_min
  double requote_s = 0.005;               // c_rq
  double latency_rt_s = 0.01;             // L_rt (the forecast horizon must cover it)
};

template <std::size_t D>
const char* validate(const ControlConfig<D>& c, double horizon_s) noexcept {
  auto pos = [](double x) { return std::isfinite(x) && x > 0.0; };
  if (!pos(c.gamma) || !pos(c.k) || !pos(c.horizon_inv_s) || !pos(c.delta_max) || !pos(c.q_max)) {
    return "gamma, k, T_h, delta_max and q_max must be positive";
  }
  if (!(std::isfinite(c.sigma_w2) && c.sigma_w2 >= 0.0)) {
    return "sigma_W^2 must be non-negative";
  }
  for (std::size_t i = 0; i < D; ++i) {
    if (!(std::isfinite(c.jump_benign[i]) && c.jump_benign[i] >= 0.0 && std::isfinite(c.jump_informed[i]) &&
          c.jump_informed[i] >= 0.0)) {
      return "jump sizes must be finite and non-negative";
    }
  }
  if (c.buy_channel >= D || c.sell_channel >= D || c.buy_channel == c.sell_channel) {
    return "quoting channels must be two distinct valid channels";
  }
  if (!(std::isfinite(c.hysteresis) && c.hysteresis >= 0.0 && std::isfinite(c.cooldown_s) &&
        c.cooldown_s >= 0.0 && std::isfinite(c.requote_s) && c.requote_s >= 0.0)) {
    return "hysteresis, cooldown and requote interval must be non-negative";
  }
  if (!(std::isfinite(c.latency_rt_s) && c.latency_rt_s >= 0.0 && horizon_s >= c.latency_rt_s)) {
    return "forecast horizon H must be at least the cancel round-trip latency (Section 7.3)";
  }
  return nullptr;
}

// Theorem 19 plus Section 8.3 on one side's terms, before rounding and state machines.
struct QuoteTerms {
  double reservation_shift = 0.0;  // Delta^
  double sigma2 = 0.0;
  double adverse_ask = 0.0;    // epsilon^+
  double adverse_bid = 0.0;    // epsilon^-
  double breakeven_ask = 0.0;  // c^a
  double breakeven_bid = 0.0;  // c^b
  double dist_ask = 0.0;       // min(delta^a*, delta_max)
  double dist_bid = 0.0;
  double theta_ask = 1.0;  // theta^a_on in [0, 1]
  double theta_bid = 1.0;
};

template <std::size_t D>
class QuoteLaw {
 public:
  void init(const ControlConfig<D>& c, const hawkes::ForecastConfig<D>& f) noexcept {
    cfg_ = c;
    dir_ = f.direction;
    inv_h_ = 1.0 / f.horizon_s;
    spread_ = std::log1p(c.gamma / c.k) / c.gamma;
  }

  void compute(const hawkes::Forecast<D>& fc, double inventory, QuoteTerms& out) const noexcept {
    double shift = 0.0;
    double var_jump = 0.0;
    for (std::size_t i = 0; i < D; ++i) {
      const double tau = dir_[i] > 0 ? fc.tau_buy : (dir_[i] < 0 ? fc.tau_sell : 0.0);
      const double jb = cfg_.jump_benign[i];
      const double ji = cfg_.jump_informed[i];
      const double jbar = jb + tau * ji;
      shift += static_cast<double>(dir_[i]) * jbar * fc.x_inf[i];
      // E[J^2] under (A2): J = J^b + 1{informed} J^I with P(informed) = tau.
      const double ej2 = jb * jb + tau * ((jb + ji) * (jb + ji) - jb * jb);
      var_jump += fc.m0[i] * ej2;
    }
    const double sigma2 = cfg_.sigma_w2 + inv_h_ * var_jump;
    const double risk = 0.5 * cfg_.gamma * sigma2 * cfg_.horizon_inv_s;
    const std::size_t a = cfg_.buy_channel;
    const std::size_t b = cfg_.sell_channel;
    const double eps_a = cfg_.jump_benign[a] + fc.tau_buy * cfg_.jump_informed[a];
    const double eps_b = cfg_.jump_benign[b] + fc.tau_sell * cfg_.jump_informed[b];
    const double base_a = shift - risk * (2.0 * inventory - 1.0);   // -(theta_{q-1} - theta_q)
    const double base_b = -shift + risk * (2.0 * inventory + 1.0);  // -(theta_{q+1} - theta_q)
    out.reservation_shift = shift;
    out.sigma2 = sigma2;
    out.adverse_ask = eps_a;
    out.adverse_bid = eps_b;
    out.breakeven_ask = eps_a + base_a;
    out.breakeven_bid = eps_b + base_b;
    out.dist_ask = std::fmin(out.breakeven_ask + spread_, cfg_.delta_max);
    out.dist_bid = std::fmin(out.breakeven_bid + spread_, cfg_.delta_max);
    out.theta_ask = threshold(cfg_.jump_benign[a] + base_a, cfg_.jump_informed[a]);
    out.theta_bid = threshold(cfg_.jump_benign[b] + base_b, cfg_.jump_informed[b]);
  }

  const ControlConfig<D>& config() const noexcept { return cfg_; }
  double spread_term() const noexcept { return spread_; }

 private:
  // theta_on = (delta_max - c(tau = 0)) / J^I, clipped to [0, 1] (Section 8.3).
  double threshold(double c_at_zero, double ji) const noexcept {
    if (!(ji > 0.0)) {
      return c_at_zero >= cfg_.delta_max ? 0.0 : 1.0;
    }
    return std::fmin(std::fmax((cfg_.delta_max - c_at_zero) / ji, 0.0), 1.0);
  }

  ControlConfig<D> cfg_{};
  std::array<int, D> dir_{};
  double inv_h_ = 20.0;
  double spread_ = 0.0;
};

// Schmitt trigger with cooldown for one side (Section 8.4).
struct SideTrigger {
  bool pulled = false;
  core::TimestampNs t_pull = 0;
  std::uint64_t cycles = 0;

  bool update(core::TimestampNs t, double tau, double theta_on, double hysteresis,
              core::DurationNs cooldown_ns) noexcept {
    if (!pulled) {
      if (tau >= theta_on) {
        pulled = true;
        t_pull = t;
        ++cycles;
      }
    } else if (tau <= theta_on - hysteresis && t - t_pull >= cooldown_ns) {
      pulled = false;
    }
    return pulled;
  }
};

// Global kill switch (Section 8.6): trips while any condition holds and clears c_min after the last
// time a condition held.
struct KillSwitch {
  bool active = true;  // the engine starts killed until every venue has been seen
  core::TimestampNs t_last_condition = 0;
  std::uint64_t trips = 0;

  bool update(core::TimestampNs t, bool condition, core::DurationNs cooldown_ns) noexcept {
    if (condition) {
      if (!active) {
        ++trips;
      }
      active = true;
      t_last_condition = t;
    } else if (active && t - t_last_condition >= cooldown_ns) {
      active = false;
    }
    return active;
  }
};

// Requote throttle for one side (Theorem 20(3)): a working price changes only if it moves and at
// least c_rq has passed since the last change; cancels (desired price 0) and placements after a
// cancel are immediate.
struct Throttle {
  std::int64_t working_e8 = 0;
  core::TimestampNs t_change = 0;
  std::uint64_t requotes = 0;

  std::int64_t update(core::TimestampNs t, std::int64_t desired_e8,
                      core::DurationNs min_interval_ns) noexcept {
    if (desired_e8 == 0 || working_e8 == 0) {
      if (desired_e8 != working_e8) {
        working_e8 = desired_e8;
        t_change = t;
      }
    } else if (desired_e8 != working_e8 && t - t_change >= min_interval_ns) {
      working_e8 = desired_e8;
      t_change = t;
      ++requotes;
    }
    return working_e8;
  }
};

// Rounds a price to the tick grid: asks up, bids down (Section 8.5). A tolerance of 1e-9 tick keeps
// prices that are on the grid up to rounding from moving by a whole tick.
inline std::int64_t round_ask_e8(double price, std::int64_t tick_e8) noexcept {
  const double ticks = price * 1e8 / static_cast<double>(tick_e8);
  return static_cast<std::int64_t>(std::ceil(ticks - 1e-9)) * tick_e8;
}
inline std::int64_t round_bid_e8(double price, std::int64_t tick_e8) noexcept {
  const double ticks = price * 1e8 / static_cast<double>(tick_e8);
  return static_cast<std::int64_t>(std::floor(ticks + 1e-9)) * tick_e8;
}

}  // namespace pulselob::control
