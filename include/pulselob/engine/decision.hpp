#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "pulselob/control/quoting.hpp"
#include "pulselob/core/fixed_matrix.hpp"
#include "pulselob/core/time.hpp"
#include "pulselob/engine/config.hpp"
#include "pulselob/engine/signal.hpp"
#include "pulselob/feed/event_mapper.hpp"
#include "pulselob/hawkes/forecast.hpp"
#include "pulselob/hawkes/kernel.hpp"
#include "pulselob/hawkes/stability.hpp"

namespace pulselob::engine {

// Step 10 of the per-event algorithm: forecasts, toxicity, quotes, pull triggers, kill switch and
// throttling, from a post-event kernel state and a set of served parameters. Shared by the inline
// engine and the hot thread of the split engine.
template <class Shape>
class DecisionLayer {
 public:
  static constexpr std::size_t D = Shape::D;
  using Layout = hawkes::BankLayout<Shape>;

  void init(const EngineConfig<Shape>& cfg, const hawkes::KernelBank<Shape>& bank) noexcept {
    forecaster_.init(bank, cfg.forecast);
    law_.init(cfg.control, cfg.forecast);
    quote_instrument_ = cfg.quote_instrument;
    tick_e8_ = cfg.channels.instruments[cfg.quote_instrument].tick_e8;
    rho_crit_ = cfg.model.rho_crit;
    warmup_ = cfg.model.warmup_events;
    stale_ns_ = cfg.stale_ns;
    reject_eta_ = cfg.reject_rate_eta;
    reject_max_ = cfg.reject_rate_max;
    cooldown_ns_ = static_cast<core::DurationNs>(std::llround(cfg.control.cooldown_s * 1e9));
    requote_ns_ = static_cast<core::DurationNs>(std::llround(cfg.control.requote_s * 1e9));
    hysteresis_ = cfg.control.hysteresis;
    q_max_ = cfg.control.q_max;
    venues_used_ = {};
    for (const auto& inst : cfg.channels.instruments) {
      if (inst.present) {
        venues_used_[inst.venue] = true;
      }
    }
    venue_seen_ = {};
    venue_last_ = {};
    ask_ = {};
    bid_ = {};
    kill_ = {};
    ask_throttle_ = {};
    bid_throttle_ = {};
    reject_rate_ = 0.0;
  }

  void on_venue(std::size_t venue, core::TimestampNs t) noexcept {
    venue_seen_[venue] = true;
    venue_last_[venue] = t;
  }

  // Updates the EWMA input-rejection rate (one update per input event).
  void on_input(bool rejected) noexcept {
    reject_rate_ += reject_eta_ * ((rejected ? 1.0 : 0.0) - reject_rate_);
  }

  void decide(core::TimestampNs t, const hawkes::KernelBank<Shape>& bank,
              const core::FixedMatrix<Layout::S, D>& Rpost, const hawkes::Served<Shape>& served,
              const feed::EventMapper<D>& mapper, double inventory, bool fault, Signal<D>& sig) noexcept {
    forecaster_.compute(bank, Rpost, served, fc_);
    law_.compute(fc_, inventory, terms_);
    const feed::BookState& book = mapper.book(quote_instrument_);

    bool stale = false;
    for (std::size_t v = 0; v < feed::kMaxVenues; ++v) {
      if (venues_used_[v] && (!venue_seen_[v] || t - venue_last_[v] > stale_ns_)) {
        stale = true;
      }
    }
    const bool condition = fault || stale || served.criticality >= rho_crit_ || reject_rate_ > reject_max_;
    const bool killed = kill_.update(t, condition, cooldown_ns_);
    const bool ask_pulled = ask_.update(t, fc_.tau_buy, terms_.theta_ask, hysteresis_, cooldown_ns_);
    const bool bid_pulled = bid_.update(t, fc_.tau_sell, terms_.theta_bid, hysteresis_, cooldown_ns_);

    std::int64_t want_ask = 0;
    std::int64_t want_bid = 0;
    if (book.valid && !killed) {
      const double mid = book.mid();
      if (!ask_pulled && inventory > -q_max_) {
        want_ask = control::round_ask_e8(mid + terms_.dist_ask, tick_e8_);
        const std::int64_t floor_ask = book.bid_px_e8 + tick_e8_;  // post-only
        want_ask = want_ask < floor_ask ? floor_ask : want_ask;
      }
      if (!bid_pulled && inventory < q_max_) {
        want_bid = control::round_bid_e8(mid - terms_.dist_bid, tick_e8_);
        const std::int64_t cap_bid = book.ask_px_e8 - tick_e8_;  // post-only
        want_bid = want_bid > cap_bid ? cap_bid : want_bid;
        want_bid = want_bid > 0 ? want_bid : 0;
      }
    }
    sig.ask_px_e8 = ask_throttle_.update(t, want_ask, requote_ns_);
    sig.bid_px_e8 = bid_throttle_.update(t, want_bid, requote_ns_);

    sig.t_ns = t;
    for (std::size_t i = 0; i < D; ++i) {
      sig.lambda[i] = fc_.lambda[i];
    }
    sig.tau_buy = fc_.tau_buy;
    sig.tau_sell = fc_.tau_sell;
    sig.criticality = served.criticality;
    sig.reservation_shift = terms_.reservation_shift;
    sig.adverse_ask = terms_.adverse_ask;
    sig.adverse_bid = terms_.adverse_bid;
    sig.sigma2 = terms_.sigma2;
    std::uint32_t flags = 0;
    flags |= served.n_eff < warmup_ ? signal_flags::kWarmingUp : 0U;
    flags |= ask_pulled ? signal_flags::kAskPulled : 0U;
    flags |= bid_pulled ? signal_flags::kBidPulled : 0U;
    flags |= killed ? signal_flags::kKillSwitch : 0U;
    flags |=
        (served.flags & hawkes::Served<Shape>::kFlagFastForgetting) != 0 ? signal_flags::kFastForgetting : 0U;
    flags |= served.projected_rows != 0 ? signal_flags::kProjectionActive : 0U;
    flags |= book.valid ? 0U : signal_flags::kNoQuoteBook;
    sig.flags = flags;
  }

  const hawkes::Forecast<D>& forecast() const noexcept { return fc_; }
  const control::QuoteTerms& terms() const noexcept { return terms_; }
  const control::SideTrigger& ask_trigger() const noexcept { return ask_; }
  const control::SideTrigger& bid_trigger() const noexcept { return bid_; }
  const control::KillSwitch& kill_switch() const noexcept { return kill_; }
  const control::Throttle& ask_throttle() const noexcept { return ask_throttle_; }
  const control::Throttle& bid_throttle() const noexcept { return bid_throttle_; }
  double reject_rate() const noexcept { return reject_rate_; }

 private:
  hawkes::Forecaster<Shape> forecaster_{};
  control::QuoteLaw<D> law_{};
  hawkes::Forecast<D> fc_{};
  control::QuoteTerms terms_{};
  control::SideTrigger ask_{};
  control::SideTrigger bid_{};
  control::KillSwitch kill_{};
  control::Throttle ask_throttle_{};
  control::Throttle bid_throttle_{};
  std::array<bool, feed::kMaxVenues> venues_used_{};
  std::array<bool, feed::kMaxVenues> venue_seen_{};
  std::array<core::TimestampNs, feed::kMaxVenues> venue_last_{};
  std::uint16_t quote_instrument_ = 0;
  std::int64_t tick_e8_ = 1;
  double rho_crit_ = 1.0;
  double warmup_ = 0.0;
  double reject_eta_ = 0.01;
  double reject_max_ = 0.05;
  double reject_rate_ = 0.0;
  double hysteresis_ = 0.1;
  double q_max_ = 10.0;
  core::DurationNs stale_ns_ = 0;
  core::DurationNs cooldown_ns_ = 0;
  core::DurationNs requote_ns_ = 0;
};

}  // namespace pulselob::engine
