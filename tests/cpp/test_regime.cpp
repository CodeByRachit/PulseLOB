#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>

#include "hawkes_sim.hpp"
#include "pulselob/hawkes/model.hpp"
#include "report.hpp"
#include "test_config.hpp"

namespace {

using namespace pulselob;
using testkit::kD;
using testkit::Shape;
using ModelT = hawkes::Model<Shape, hawkes::CholeskyEstimator<Shape>>;

struct Path {
  std::vector<double> t;
  std::vector<double> cross;  // served G^(spot buy <- perp buy)
  std::vector<double> alarm_times;
};

constexpr double kShift = 300.0;
constexpr double kEnd = 600.0;
constexpr double kA1 = 0.10;
constexpr double kA2 = 0.50;

std::vector<testkit::ChannelEvent> simulate_shift(std::uint64_t seed) {
  testkit::Rng rng(seed);
  testkit::HawkesSimulator<Shape> sim(testkit::default_beta(), testkit::ground_truth_theta(kA1));
  sim.schedule_switch(kShift, testkit::ground_truth_theta(kA2));
  return sim.simulate(kEnd, rng);
}

Path run(const std::vector<testkit::ChannelEvent>& ev, bool detector) {
  auto cfg = testkit::default_engine_config().model;
  cfg.cusum_enabled = detector;
  // The detector mechanism is checked at the sensitive threshold of the D = 2 experiment (14). The
  // engine default is the conservative 30, which rarely fires on a shift this moderate.
  cfg.cusum_threshold = 14.0;
  auto m = std::make_unique<ModelT>(cfg);
  Path out;
  std::uint64_t alarms = 0;
  double next_sample = 1.0;
  std::int64_t last = 0;
  for (const auto& e : ev) {
    const std::int64_t t = 1'000'000'000 + static_cast<std::int64_t>(std::llround(e.t_s * 1e9));
    while (last != 0 && t - last > 100'000'000) {  // heartbeats in quiet stretches
      last += 100'000'000;
      m->step(last, -1, 0.0);
    }
    m->step(t, e.channel, 1.0);
    last = t;
    if (m->counters().alarms != alarms) {
      alarms = m->counters().alarms;
      out.alarm_times.push_back(e.t_s);
    }
    if (e.t_s >= next_sample) {
      double g = 0.0;
      for (std::size_t q = 0; q < Shape::Q; ++q) {
        g += m->served().theta(2, Shape::coord(0, q));
      }
      out.t.push_back(e.t_s);
      out.cross.push_back(g);
      next_sample += 1.0;
    }
  }
  return out;
}

double mean_over(const Path& p, double a, double b) {
  double s = 0.0;
  int n = 0;
  for (std::size_t k = 0; k < p.t.size(); ++k) {
    if (p.t[k] >= a && p.t[k] < b) {
      s += p.cross[k];
      ++n;
    }
  }
  return s / n;
}

double lag_to_midpoint(const Path& p) {
  const double mid = 0.5 * (kA1 + kA2);
  for (std::size_t k = 0; k < p.t.size(); ++k) {
    if (p.t[k] >= kShift && p.cross[k] >= mid) {
      return p.t[k] - kShift;
    }
  }
  return 1e9;
}

// V10: non-stationary recovery A1 -> A2 (math_spec Sections 5.5 and 5.8, now with D = 4 and the
// full kernel bank). The coupling is spread over a 5 ms Erlang-2 hump; the estimate checked is the
// branching entry G(spot buy <- perp buy) = sum_q alpha^q.
TEST(Regime, DetectorFiresAndEstimatorConvergesToNewRegime) {
  const auto ev = simulate_shift(2024);
  const Path with = run(ev, true);
  const Path without = run(ev, false);
  const double before = mean_over(with, 200.0, kShift);
  const double after = mean_over(with, 500.0, kEnd);
  std::size_t alarms_before = 0;
  std::size_t alarms_after_shift = 0;
  for (double ta : with.alarm_times) {
    alarms_before += ta < kShift ? 1U : 0U;
    alarms_after_shift += (ta >= kShift && ta < kShift + 60.0) ? 1U : 0U;
  }
  const double lag_with = lag_to_midpoint(with);
  const double lag_without = lag_to_midpoint(without);
  RecordProperty("events", testkit::sci(ev.size()));
  RecordProperty("cross_before", testkit::sci(before));
  RecordProperty("cross_after", testkit::sci(after));
  RecordProperty("lag_with_detector_s", testkit::sci(lag_with));
  RecordProperty("lag_without_detector_s", testkit::sci(lag_without));
  RecordProperty("alarms_before_shift", testkit::sci(alarms_before));
  RecordProperty("alarms_within_60s_after_shift", testkit::sci(alarms_after_shift));
  std::printf(
      "[V10] events=%zu cross before=%.4f after=%.4f lag with=%.1fs without=%.1fs alarms pre-shift=%zu "
      "post-shift(60s)=%zu\n",
      ev.size(), before, after, lag_with, lag_without, alarms_before, alarms_after_shift);
  EXPECT_NEAR(before, kA1, 0.06);
  EXPECT_NEAR(after, kA2, 0.08);
  EXPECT_GE(alarms_after_shift, 1U);
  EXPECT_LT(lag_with, lag_without);
  EXPECT_LT(lag_without, 120.0);
}

// In-control behavior of the detector: alarms per hour on a stationary stream (no regime change)
// for several thresholds. Not a pass/fail calibration (that is Phase 5, by run-length simulation);
// recorded so the default b_cs = 14, inherited from the D = 2 experiment, can be revisited for D = 4
// (2D = 8 CUSUMs, plug-in residuals).
TEST(Regime, InControlFalseAlarmRate) {
  const double horizon = 1800.0;
  testkit::Rng rng(77);
  testkit::HawkesSimulator<Shape> sim(testkit::default_beta(), testkit::ground_truth_theta(kA1));
  const auto ev = sim.simulate(horizon, rng);
  double prev_rate = 1e300;
  for (double b : {14.0, 20.0, 30.0}) {
    auto cfg = testkit::default_engine_config().model;
    cfg.cusum_threshold = b;
    auto m = std::make_unique<ModelT>(cfg);
    std::int64_t last = 0;
    for (const auto& e : ev) {
      const std::int64_t t = 1'000'000'000 + static_cast<std::int64_t>(std::llround(e.t_s * 1e9));
      while (last != 0 && t - last > 100'000'000) {
        last += 100'000'000;
        m->step(last, -1, 0.0);
      }
      m->step(t, e.channel, 1.0);
      last = t;
    }
    const double per_hour = static_cast<double>(m->counters().alarms) / (horizon / 3600.0);
    std::printf("[ARL] b_cs=%.0f: %.1f false alarms per hour (%zu events)\n", b, per_hour, ev.size());
    RecordProperty("alarms_per_hour_b" + testkit::sci(static_cast<int>(b)), testkit::sci(per_hour));
    EXPECT_LE(per_hour, prev_rate);
    prev_rate = per_hour;
  }
}

}  // namespace
