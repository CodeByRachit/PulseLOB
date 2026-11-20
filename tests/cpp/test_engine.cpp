#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <type_traits>
#include <vector>

#include "hawkes_sim.hpp"
#include "pulselob/core/fp_env.hpp"
#include "pulselob/engine/engine.hpp"
#include "report.hpp"
#include "state_compare.hpp"
#include "test_config.hpp"

namespace {

using namespace pulselob;
using testkit::kD;
using testkit::Shape;
using CholEngine = engine::Engine<Shape, hawkes::CholeskyEstimator<Shape>>;
using PotterEngine = engine::Engine<Shape, hawkes::PotterEstimator<Shape>>;
using Sig = engine::Signal<kD>;

static_assert(std::is_trivially_copyable_v<CholEngine>, "a snapshot must be a plain copy");
static_assert(std::is_trivially_copyable_v<PotterEngine>);
// The whole engine (state, scratch, configuration and output) stays inside a 24 KB L1 budget
// (docs/architecture.md Section 5.3: 48 KB L1D on recent x86, 128 KB on the M4 performance cores).
static_assert(sizeof(CholEngine) <= 24 * 1024);
static_assert(sizeof(PotterEngine) <= 28 * 1024);

std::vector<feed::MarketEvent> make_stream(std::uint64_t seed, double horizon_s) {
  testkit::Rng rng(seed);
  testkit::HawkesSimulator<Shape> sim(testkit::default_beta(), testkit::ground_truth_theta(0.3));
  const auto ev = sim.simulate(horizon_s, rng);
  return testkit::to_market_events(ev, rng);
}

template <class E>
std::vector<Sig> run(E& eng, const std::vector<feed::MarketEvent>& ev) {
  std::vector<Sig> out;
  out.reserve(ev.size());
  for (const auto& e : ev) {
    eng.on_event(e);
    out.push_back(eng.signal());
  }
  return out;
}

bool same_bytes(const std::vector<Sig>& a, const std::vector<Sig>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(Sig)) == 0;
}

TEST(Engine, ConfigValidationCatchesViolations) {
  auto cfg = testkit::default_engine_config();
  EXPECT_EQ(engine::validate(cfg), nullptr);
  auto bad = cfg;
  bad.model.half_life_fast_s = 0.5;  // kappa_fast = 1.39 > beta_min = 1: (I2)
  EXPECT_NE(engine::validate(bad), nullptr);
  bad = cfg;
  bad.model.rho_max = 0.995;  // rho_max * chi >= 1: (S4)
  EXPECT_NE(engine::validate(bad), nullptr);
  bad = cfg;
  bad.forecast.horizon_s = 0.001;  // H below the cancel latency
  EXPECT_NE(engine::validate(bad), nullptr);
  bad = cfg;
  bad.channels.set_channel(0, feed::Side::Sell, 0);  // channel 0 mapped twice, channel 1 never
  EXPECT_NE(engine::validate(bad), nullptr);
}

TEST(Engine, ReplayIsBitwiseDeterministic) {
  const auto cfg = testkit::default_engine_config();
  const auto ev = make_stream(61, 60.0);
  auto a = std::make_unique<CholEngine>(cfg);
  auto b = std::make_unique<CholEngine>(cfg);
  EXPECT_TRUE(same_bytes(run(*a, ev), run(*b, ev)));
}

TEST(Engine, SnapshotRestoreContinuesIdentically) {
  const auto cfg = testkit::default_engine_config();
  const auto ev = make_stream(67, 60.0);
  const std::size_t half = ev.size() / 2;
  auto full = std::make_unique<CholEngine>(cfg);
  const auto ref = run(*full, ev);
  auto first = std::make_unique<CholEngine>(cfg);
  for (std::size_t k = 0; k < half; ++k) {
    first->on_event(ev[k]);
  }
  auto restored = std::make_unique<CholEngine>(*first);  // snapshot = copy
  std::vector<Sig> tail;
  for (std::size_t k = half; k < ev.size(); ++k) {
    restored->on_event(ev[k]);
    tail.push_back(restored->signal());
  }
  EXPECT_EQ(0, std::memcmp(tail.data(), ref.data() + half, tail.size() * sizeof(Sig)));
}

TEST(Engine, FlushToZeroDoesNotChangeResults) {
  const auto cfg = testkit::default_engine_config();
  const auto ev = make_stream(71, 60.0);
  auto plain = std::make_unique<CholEngine>(cfg);
  const auto a = run(*plain, ev);
  std::vector<Sig> b;
  {
    const core::FlushToZeroGuard ftz;
    auto flushed = std::make_unique<CholEngine>(cfg);
    b = run(*flushed, ev);
  }
  EXPECT_TRUE(same_bytes(a, b));
}

TEST(Engine, BackendsProduceTheSameSignals) {
  const auto cfg = testkit::default_engine_config();
  const auto ev = make_stream(73, 120.0);
  auto c = std::make_unique<CholEngine>(cfg);
  auto p = std::make_unique<PotterEngine>(cfg);
  const auto sc = run(*c, ev);
  const auto sp = run(*p, ev);
  double worst = 0.0;
  std::size_t quote_diffs = 0;
  for (std::size_t k = 0; k < sc.size(); ++k) {
    worst = std::fmax(worst, std::fabs(sc[k].tau_buy - sp[k].tau_buy));
    worst = std::fmax(worst, std::fabs(sc[k].tau_sell - sp[k].tau_sell));
    worst = std::fmax(worst, std::fabs(sc[k].reservation_shift - sp[k].reservation_shift) /
                                 (1.0 + std::fabs(sc[k].reservation_shift)));
    for (std::size_t i = 0; i < kD; ++i) {
      worst = std::fmax(worst, std::fabs(sc[k].lambda[i] - sp[k].lambda[i]) / sc[k].lambda[i]);
    }
    quote_diffs += (sc[k].bid_px_e8 != sp[k].bid_px_e8 || sc[k].ask_px_e8 != sp[k].ask_px_e8) ? 1U : 0U;
  }
  RecordProperty("worst_signal_difference", testkit::sci(worst));
  EXPECT_LT(worst, 1e-9);
  EXPECT_EQ(quote_diffs, 0U);
}

template <class E>
bool model_unchanged(const E& before, const E& after) {
  bool books = true;
  for (std::size_t k = 0; k < feed::kMaxInstruments; ++k) {
    const auto& a = before.mapper().book(k);
    const auto& b = after.mapper().book(k);
    books = books && a.valid == b.valid && a.t == b.t && a.bid_px_e8 == b.bid_px_e8 &&
            a.ask_px_e8 == b.ask_px_e8 && a.bid_qty_e8 == b.bid_qty_e8 && a.ask_qty_e8 == b.ask_qty_e8;
  }
  return testkit::same_model_state(before.model(), after.model()) &&
         testkit::same_bits(before.marks().log_median(), after.marks().log_median()) &&
         testkit::same_bits(before.marks().g_bar(), after.marks().g_bar()) && books &&
         before.clock() == after.clock();
}

TEST(Engine, RejectionsLeaveTheModelUntouched) {
  const auto cfg = testkit::default_engine_config();
  const auto ev = make_stream(79, 10.0);
  auto eng = std::make_unique<CholEngine>(cfg);
  for (std::size_t k = 0; k < ev.size() / 2; ++k) {
    eng->on_event(ev[k]);
  }
  const auto snap = std::make_unique<CholEngine>(*eng);
  // Invalid record.
  feed::MarketEvent bad = ev[ev.size() / 2];
  bad.kind = feed::EventKind::Trade;
  bad.flags = feed::event_flags::make(feed::Side::Buy, feed::TsResolution::Nanosecond);
  bad.qty0_e8 = -5;
  EXPECT_EQ(eng->on_event(bad), engine::Status::RejectedInvalid);
  EXPECT_TRUE(model_unchanged(*snap, *eng));
  EXPECT_EQ(eng->counters().invalid[static_cast<std::size_t>(feed::EventCheck::NonPositiveQuantity)], 1U);
  // Too far out of order.
  feed::MarketEvent late = ev[ev.size() / 2 - 10];
  late.ts_local_ns = eng->clock() - cfg.ooo_tolerance_ns - 1;
  EXPECT_EQ(eng->on_event(late), engine::Status::RejectedOutOfOrder);
  EXPECT_TRUE(model_unchanged(*snap, *eng));
  // Within tolerance: clamped to the engine clock.
  feed::MarketEvent jitter = late;
  jitter.ts_local_ns = eng->clock() - cfg.ooo_tolerance_ns / 2;
  const std::int64_t clock = eng->clock();
  EXPECT_EQ(eng->on_event(jitter), engine::Status::Clamped);
  EXPECT_EQ(eng->clock(), clock);
  EXPECT_EQ(eng->signal().t_ns, clock);
}

// Proposition 4: permuting cross-channel events inside a tie batch does not change the model.
TEST(Engine, TieBatchIsOrderInvariant) {
  const auto cfg = testkit::default_engine_config();
  auto ev = make_stream(83, 20.0);
  std::vector<feed::MarketEvent> a;
  std::vector<feed::MarketEvent> b;
  for (std::size_t k = 0; k < ev.size(); ++k) {
    a.push_back(ev[k]);
    b.push_back(ev[k]);
    if (k % 200 == 100) {
      // Append a batch of four trades on four different channels at the same timestamp.
      std::array<feed::MarketEvent, 4> batch{};
      for (std::size_t c = 0; c < 4; ++c) {
        feed::MarketEvent t = ev[k];
        t.kind = feed::EventKind::Trade;
        t.instrument = static_cast<std::uint16_t>(c / 2);
        const bool buy = c % 2 == 0;
        const std::int64_t tick = cfg.channels.instruments[c / 2].tick_e8;
        const std::int64_t mid = 6'000'000'000'000;
        t.px1_e8 = buy ? mid + tick : mid - tick;
        t.px0_e8 = t.px1_e8;
        t.qty0_e8 = 10'000'000 * static_cast<std::int64_t>(c + 1);
        t.qty1_e8 = 1;
        t.count = 1;
        t.flags =
            feed::event_flags::make(buy ? feed::Side::Buy : feed::Side::Sell, feed::TsResolution::Nanosecond);
        batch[c] = t;
      }
      for (std::size_t c = 0; c < 4; ++c) {
        a.push_back(batch[c]);
        b.push_back(batch[3 - c]);
      }
    }
  }
  auto ea = std::make_unique<CholEngine>(cfg);
  auto eb = std::make_unique<CholEngine>(cfg);
  for (const auto& e : a) {
    ea->on_event(e);
  }
  for (const auto& e : b) {
    eb->on_event(e);
  }
  const auto& ta = ea->model().estimator().theta();
  const auto& tb = eb->model().estimator().theta();
  for (std::size_t i = 0; i < kD; ++i) {
    for (std::size_t p = 0; p < Shape::P; ++p) {
      EXPECT_NEAR(ta(i, p), tb(i, p), 1e-12 * (1.0 + std::fabs(ta(i, p))));
    }
  }
  EXPECT_EQ(ea->model().counters().alarms, eb->model().counters().alarms);
}

TEST(Engine, SignalsAreWellFormed) {
  const auto cfg = testkit::default_engine_config();
  const auto ev = make_stream(89, 60.0);
  auto eng = std::make_unique<CholEngine>(cfg);
  const auto sig = run(*eng, ev);
  EXPECT_NE(sig.front().flags & engine::signal_flags::kWarmingUp, 0U);
  std::size_t quoted = 0;
  const std::int64_t tick = cfg.channels.instruments[1].tick_e8;
  for (std::size_t k = 0; k < sig.size(); ++k) {
    const Sig& s = sig[k];
    ASSERT_TRUE(std::isfinite(s.tau_buy) && std::isfinite(s.tau_sell) && std::isfinite(s.sigma2));
    EXPECT_GE(s.tau_buy, 0.0);
    EXPECT_LT(s.tau_buy, 1.0);
    EXPECT_EQ(s.tau_buy * s.tau_sell, 0.0);
    for (std::size_t i = 0; i < kD; ++i) {
      EXPECT_GE(s.lambda[i], cfg.model.mu_min);
    }
    if (s.bid_px_e8 != 0 && s.ask_px_e8 != 0) {
      ++quoted;
      EXPECT_LT(s.bid_px_e8, s.ask_px_e8);
      EXPECT_EQ(s.bid_px_e8 % tick, 0);
      EXPECT_EQ(s.ask_px_e8 % tick, 0);
    }
  }
  EXPECT_GT(quoted, sig.size() / 10);  // sides are pulled often: delta_max is tight in the test config
  EXPECT_EQ(eng->model().estimator().factorization_failures(), 0U);
}

TEST(Engine, KillSwitchCauses) {
  const auto cfg = testkit::default_engine_config();
  auto ev = make_stream(97, 10.0);
  auto eng = std::make_unique<CholEngine>(cfg);
  // Before any spot event the spot venue is unseen: killed.
  feed::MarketEvent first = ev.front();  // perp book
  eng->on_event(first);
  EXPECT_NE(eng->signal().flags & engine::signal_flags::kKillSwitch, 0U);
  for (std::size_t k = 1; k < ev.size(); ++k) {
    eng->on_event(ev[k]);
  }
  const std::int64_t t_end = eng->clock();
  EXPECT_EQ(eng->signal().flags & engine::signal_flags::kKillSwitch, 0U);
  // Feed status event trips it.
  feed::MarketEvent st{};
  st.ts_local_ns = t_end + 1'000;
  st.kind = feed::EventKind::Status;
  st.count = static_cast<std::uint32_t>(feed::StatusCode::FeedGap);
  eng->on_event(st);
  EXPECT_NE(eng->signal().flags & engine::signal_flags::kKillSwitch, 0U);
  // A non-finite inventory is ignored and trips it at the next event.
  eng->set_inventory(std::nan(""));
  EXPECT_EQ(eng->inventory(), 0.0);
  EXPECT_EQ(eng->counters().nonfinite_inputs, 1U);
  // Stale feed: a heartbeat 3 s later with no market data.
  EXPECT_EQ(eng->on_heartbeat(t_end + 3'000'000'000), engine::Status::Accepted);
  EXPECT_NE(eng->signal().flags & engine::signal_flags::kKillSwitch, 0U);
}

TEST(Engine, InventoryLimitsRemoveTheRiskIncreasingSide) {
  const auto cfg = testkit::default_engine_config();
  const auto ev = make_stream(101, 20.0);
  auto eng = std::make_unique<CholEngine>(cfg);
  for (const auto& e : ev) {
    eng->on_event(e);
  }
  eng->set_inventory(cfg.control.q_max);
  eng->on_heartbeat(eng->clock() + 1'000'000);
  EXPECT_EQ(eng->signal().bid_px_e8, 0);
  eng->set_inventory(-cfg.control.q_max);
  eng->on_heartbeat(eng->clock() + 1'000'000);
  EXPECT_EQ(eng->signal().ask_px_e8, 0);
}

}  // namespace
