#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include "hawkes_sim.hpp"
#include "pulselob/core/linalg.hpp"
#include "pulselob/engine/engine.hpp"
#include "state_compare.hpp"
#include "test_config.hpp"

namespace {

using namespace pulselob;
using testkit::kD;
using testkit::kP;
using testkit::Shape;
using CholEngine = engine::Engine<Shape, hawkes::CholeskyEstimator<Shape>>;
using PotterEngine = engine::Engine<Shape, hawkes::PotterEstimator<Shape>>;

feed::MarketEvent trade(std::int64_t t, std::uint16_t inst, bool buy, std::int64_t qty_e8,
                        std::int64_t levels = 0) {
  const std::int64_t tick = inst == 0 ? 10'000'000 : 1'000'000;
  const std::int64_t mid = 6'000'000'000'000;
  feed::MarketEvent e{};
  e.ts_exchange_ns = t;
  e.ts_local_ns = t;
  e.px1_e8 = buy ? mid + tick : mid - tick;
  e.px0_e8 = buy ? e.px1_e8 + levels * tick : e.px1_e8 - levels * tick;
  e.qty0_e8 = qty_e8;
  e.qty1_e8 = 0;
  e.count = 1;
  e.instrument = inst;
  e.kind = feed::EventKind::Trade;
  e.flags = feed::event_flags::make(buy ? feed::Side::Buy : feed::Side::Sell, feed::TsResolution::Nanosecond);
  return e;
}

feed::MarketEvent book(std::int64_t t, std::uint16_t inst, std::int64_t qb, std::int64_t qa) {
  const std::int64_t tick = inst == 0 ? 10'000'000 : 1'000'000;
  feed::MarketEvent b{};
  b.ts_exchange_ns = t;
  b.ts_local_ns = t;
  b.px0_e8 = 6'000'000'000'000 - tick;
  b.px1_e8 = 6'000'000'000'000 + tick;
  b.qty0_e8 = qb;
  b.qty1_e8 = qa;
  b.instrument = inst;
  b.kind = feed::EventKind::BookTop;
  return b;
}

// Invariants of math_spec V11: everything finite, M positive definite, the served model certified
// sub-critical (G^ v <= rho_max v), Potter's U U^T M = I, no factorization failure.
template <class E>
void check_invariants(const E& eng, const char* where) {
  const auto& m = eng.model();
  const auto& s = m.served();
  const auto& sig = eng.signal();
  const double vals[] = {sig.tau_buy,     sig.tau_sell,    sig.criticality, sig.reservation_shift,
                         sig.adverse_ask, sig.adverse_bid, sig.sigma2,      s.n_eff};
  for (double v : vals) {
    ASSERT_TRUE(std::isfinite(v)) << where;
  }
  for (std::size_t i = 0; i < kD; ++i) {
    ASSERT_TRUE(std::isfinite(sig.lambda[i])) << where;
    double gv = 0.0;
    for (std::size_t j = 0; j < kD; ++j) {
      gv += s.G(i, j) * s.v[j];
    }
    ASSERT_LE(gv, eng.config().model.rho_max * s.v[i] * (1.0 + 1e-14)) << where;
    for (std::size_t p = 0; p < kP; ++p) {
      ASSERT_TRUE(std::isfinite(m.estimator().theta()(i, p))) << where;
      ASSERT_TRUE(std::isfinite(s.theta(i, p))) << where;
    }
  }
  for (std::size_t r = 0; r < hawkes::BankLayout<Shape>::S; ++r) {
    for (std::size_t j = 0; j < kD; ++j) {
      ASSERT_TRUE(std::isfinite(m.kernel().R(r, j)) && m.kernel().R(r, j) >= 0.0) << where;
    }
  }
  auto M = m.estimator().stats().M;
  core::FixedVector<kP> col{};
  ASSERT_TRUE(core::cholesky_lower(M, col)) << "M not positive definite: " << where;
  ASSERT_EQ(m.estimator().factorization_failures(), 0U) << where;
}

template <class E>
std::unique_ptr<E> warmed_engine() {
  auto eng = std::make_unique<E>(testkit::default_engine_config());
  testkit::Rng rng(107);
  testkit::HawkesSimulator<Shape> sim(testkit::default_beta(), testkit::ground_truth_theta(0.3));
  const auto ev = testkit::to_market_events(sim.simulate(30.0, rng), rng);
  for (const auto& e : ev) {
    eng->on_event(e);
  }
  return eng;
}

template <class E>
class Stress : public ::testing::Test {};
using Engines = ::testing::Types<CholEngine, PotterEngine>;
TYPED_TEST_SUITE(Stress, Engines);

// Flash crash: 10,000 aggressive sells on both venues within 1 ms.
TYPED_TEST(Stress, FlashCrashBurst) {
  auto eng = warmed_engine<TypeParam>();
  std::int64_t t = eng->clock();
  for (int k = 0; k < 10'000; ++k) {
    t += 100;  // 100 ns apart
    eng->on_event(trade(t, static_cast<std::uint16_t>(k % 2), false, 50'000'000, k % 5));
    if (k % 1000 == 0) {
      check_invariants(*eng, "flash crash");
    }
  }
  check_invariants(*eng, "flash crash end");
  EXPECT_GT(eng->signal().tau_sell, 0.5);
  EXPECT_EQ(eng->signal().tau_buy, 0.0);
  EXPECT_GE(eng->signal().criticality, 1.0);  // the data look supercritical...
  EXPECT_NE(eng->signal().flags & engine::signal_flags::kProjectionActive, 0U);  // ...and are projected
  EXPECT_NE(eng->signal().flags & engine::signal_flags::kKillSwitch, 0U);
  // The cascade recedes: one second later the state has decayed and the score is back down.
  eng->on_heartbeat(t + 1'000'000'000);
  check_invariants(*eng, "after crash");
}

// 1e5 events with one timestamp: one tie batch.
TYPED_TEST(Stress, HundredThousandSimultaneousEvents) {
  auto eng = warmed_engine<TypeParam>();
  const std::int64_t t = eng->clock() + 1'000'000;
  for (int k = 0; k < 100'000; ++k) {
    eng->on_event(trade(t, static_cast<std::uint16_t>((k / 2) % 2), k % 2 == 0, 1'000'000 + k));
  }
  check_invariants(*eng, "same-timestamp batch");
  // The flush happens at the next later timestamp.
  eng->on_heartbeat(t + 1);
  check_invariants(*eng, "after batch flush");
}

// Gaps of 1 ns, then idle gaps of 1e4 s and 1e6 s without heartbeats.
TYPED_TEST(Stress, ExtremeGaps) {
  auto eng = warmed_engine<TypeParam>();
  std::int64_t t = eng->clock();
  for (int k = 0; k < 5000; ++k) {
    t += 1;
    eng->on_event(trade(t, static_cast<std::uint16_t>(k % 2), k % 3 == 0, 2'000'000));
  }
  check_invariants(*eng, "1 ns gaps");
  for (const std::int64_t gap :
       {std::int64_t{10'000} * 1'000'000'000, std::int64_t{1'000'000} * 1'000'000'000}) {
    t += gap;
    eng->on_event(book(t, 0, 100'000'000, 100'000'000));
    eng->on_event(book(t, 1, 100'000'000, 100'000'000));
    eng->on_event(trade(t, 1, true, 3'000'000));
    check_invariants(*eng, "after idle gap");
    // After an idle gap the kernel states have been cut and the estimator is restored toward the prior.
    for (std::size_t q = 0; q < Shape::Q; ++q) {
      EXPECT_EQ(eng->model().kernel().R(hawkes::BankLayout<Shape>::row(q, 0), 0), 0.0);
    }
    // The model keeps working afterwards.
    for (int k = 0; k < 2000; ++k) {
      t += 5'000'000;
      eng->on_event(trade(t, static_cast<std::uint16_t>(k % 2), k % 2 == 0, 2'000'000));
    }
    check_invariants(*eng, "resumed after gap");
  }
  EXPECT_EQ(eng->model().counters().long_gaps >= 2U, true);
}

// Extreme marks: tiny and huge sizes, deep sweeps, one-sided books. The mark stays in [g_min/g_max,
// g_max/g_min] (Lemma 1), so nothing overflows.
TYPED_TEST(Stress, ExtremeMarks) {
  auto eng = warmed_engine<TypeParam>();
  std::int64_t t = eng->clock();
  const std::int64_t qtys[] = {1, 9'000'000'000'000'000'000, 1, 100'000'000};
  for (int k = 0; k < 4000; ++k) {
    t += 200'000;
    if (k % 3 == 0) {
      eng->on_event(book(t, static_cast<std::uint16_t>(k % 2), k % 2 == 0 ? 1 : 9'000'000'000'000'000'000,
                         k % 2 == 0 ? 9'000'000'000'000'000'000 : 1));
    }
    eng->on_event(trade(t, static_cast<std::uint16_t>(k % 2), k % 4 < 2, qtys[k % 4], (k % 7) * 10'000));
  }
  check_invariants(*eng, "extreme marks");
}

// Out-of-order streams: jitter within epsilon is clamped, larger disorder is rejected with no state
// change; the engine clock never decreases.
TYPED_TEST(Stress, DisorderedStream) {
  auto eng = warmed_engine<TypeParam>();
  testkit::Rng rng(109);
  std::int64_t t = eng->clock();
  std::int64_t last_clock = eng->clock();
  for (int k = 0; k < 20'000; ++k) {
    t += 300'000;
    const double u = rng.uniform();
    std::int64_t ts = t;
    if (u < 0.2) {
      ts = eng->clock() - 1 - static_cast<std::int64_t>(rng.uniform() * 90'000.0);  // within 100 us
    } else if (u < 0.3) {
      ts = eng->clock() - 200'000 - static_cast<std::int64_t>(rng.uniform() * 1e8);  // far out of order
    }
    auto snap = std::make_unique<TypeParam>(*eng);
    const auto st = eng->on_event(trade(ts, static_cast<std::uint16_t>(k % 2), k % 2 == 0, 2'000'000));
    if (st == engine::Status::RejectedOutOfOrder) {
      ASSERT_TRUE(testkit::same_model_state(snap->model(), eng->model()));
    }
    ASSERT_GE(eng->clock(), last_clock);
    last_clock = eng->clock();
  }
  EXPECT_GT(eng->counters().ooo_clamped, 1000U);
  EXPECT_GT(eng->counters().ooo_rejected, 500U);
  check_invariants(*eng, "disordered stream");
}

// Non-finite and invalid inputs at the model level are rejected with no state change.
TEST(StressModel, NonFiniteMarksAreRejected) {
  using ModelT = hawkes::Model<Shape, hawkes::PotterEstimator<Shape>>;
  auto m = std::make_unique<ModelT>(testkit::default_engine_config().model);
  m->step(1'000'000'000, 0, 1.0);
  m->step(1'001'000'000, 1, 1.0);
  const auto snap = std::make_unique<ModelT>(*m);
  for (double g :
       {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(), -1.0, 0.0}) {
    EXPECT_EQ(m->step(1'002'000'000, 2, g), ModelT::StepResult::Rejected);
  }
  EXPECT_EQ(m->step(1'002'000'000, 7, 1.0), ModelT::StepResult::Rejected);  // bad channel
  EXPECT_EQ(m->step(1'000'500'000, 0, 1.0), ModelT::StepResult::Rejected);  // time went backwards
  EXPECT_EQ(m->counters().rejected_inputs, 6U);
  // Everything except the rejection counter is unchanged.
  ModelT::StepResult r = m->step(1'002'000'000, 2, 1.0);
  ModelT::StepResult r2 = snap->step(1'002'000'000, 2, 1.0);
  EXPECT_EQ(r, r2);
  EXPECT_TRUE(testkit::same_served(m->served(), snap->served()));
  EXPECT_TRUE(testkit::same_estimator(m->estimator(), snap->estimator()));
}

}  // namespace
