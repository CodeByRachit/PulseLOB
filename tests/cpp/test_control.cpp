#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>

#include "hawkes_sim.hpp"
#include "pulselob/control/quoting.hpp"
#include "test_config.hpp"

namespace {

using namespace pulselob;
using testkit::kD;

hawkes::Forecast<kD> random_forecast(testkit::Rng& rng) {
  hawkes::Forecast<kD> f{};
  for (std::size_t i = 0; i < kD; ++i) {
    f.m0[i] = 2.0 * rng.uniform();
    f.x_inf[i] = rng.normal();
  }
  if (rng.uniform() < 0.5) {
    f.tau_buy = rng.uniform() * 0.99;
  } else {
    f.tau_sell = rng.uniform() * 0.99;
  }
  return f;
}

// Theorem 19: p^a, p^b against the closed form written out term by term.
TEST(QuoteLaw, MatchesTheorem19) {
  const auto cfg = testkit::default_engine_config();
  control::QuoteLaw<kD> law;
  law.init(cfg.control, cfg.forecast);
  const auto& c = cfg.control;
  testkit::Rng rng(41);
  for (int trial = 0; trial < 5000; ++trial) {
    const auto f = random_forecast(rng);
    const double q = 20.0 * rng.uniform() - 10.0;
    control::QuoteTerms t{};
    law.compute(f, q, t);
    double shift = 0.0;
    double var = 0.0;
    for (std::size_t i = 0; i < kD; ++i) {
      const int s = cfg.forecast.direction[i];
      const double tau = s > 0 ? f.tau_buy : f.tau_sell;
      shift += s * (c.jump_benign[i] + tau * c.jump_informed[i]) * f.x_inf[i];
      const double jb = c.jump_benign[i];
      const double ji = c.jump_informed[i];
      var += f.m0[i] * ((1.0 - tau) * jb * jb + tau * (jb + ji) * (jb + ji));
    }
    const double sigma2 = c.sigma_w2 + var / cfg.forecast.horizon_s;
    const double g = c.gamma;
    const double Th = c.horizon_inv_s;
    const double eps_a = c.jump_benign[2] + f.tau_buy * c.jump_informed[2];
    const double eps_b = c.jump_benign[3] + f.tau_sell * c.jump_informed[3];
    const double spread = std::log(1.0 + g / c.k) / g;
    const double pa = shift - q * g * sigma2 * Th + 0.5 * g * sigma2 * Th + spread + eps_a;  // minus mid
    const double pb = shift - q * g * sigma2 * Th - 0.5 * g * sigma2 * Th - spread - eps_b;
    EXPECT_NEAR(t.reservation_shift, shift, 1e-12 * (1.0 + std::fabs(shift)));
    EXPECT_NEAR(t.sigma2, sigma2, 1e-12 * sigma2);
    EXPECT_NEAR(std::fmin(pa, c.delta_max), t.dist_ask, 1e-10 * (1.0 + std::fabs(pa)));
    EXPECT_NEAR(std::fmin(-pb, c.delta_max), t.dist_bid, 1e-10 * (1.0 + std::fabs(pb)));
  }
}

// Section 8.3: pulling the ask (c^a >= delta_max) is exactly tau^+ >= theta^a_on.
TEST(QuoteLaw, PullThresholdIsTheBreakEvenCondition) {
  const auto cfg = testkit::default_engine_config();
  control::QuoteLaw<kD> law;
  law.init(cfg.control, cfg.forecast);
  testkit::Rng rng(43);
  std::size_t pulls = 0;
  std::size_t interior = 0;
  for (int trial = 0; trial < 20000; ++trial) {
    auto f = random_forecast(rng);
    f.tau_sell = 0.0;
    f.tau_buy = rng.uniform() * 0.999;
    const double q = 6.0 * rng.uniform() - 3.0;
    control::QuoteTerms t{};
    law.compute(f, q, t);
    const bool breakeven_pull = t.breakeven_ask >= cfg.control.delta_max;
    const bool threshold_pull = f.tau_buy >= t.theta_ask;
    if (std::fabs(t.breakeven_ask - cfg.control.delta_max) > 1e-12) {
      EXPECT_EQ(breakeven_pull, threshold_pull) << "tau=" << f.tau_buy << " theta=" << t.theta_ask;
    }
    pulls += breakeven_pull ? 1U : 0U;
    interior += (t.theta_ask > 0.0 && t.theta_ask < 1.0) ? 1U : 0U;
  }
  EXPECT_GT(pulls, 100U);
  EXPECT_GT(interior, 100U);
}

// Theorem 20 (1) and (2): pull cycles are bounded by T / c_min + 1 and by TV(tau) / Delta_tau + 1.
TEST(SideTrigger, CycleCountObeysTheorem20) {
  testkit::Rng rng(47);
  const double theta = 0.4;
  const double hyst = 0.1;
  const core::DurationNs cooldown = 250'000'000;
  control::SideTrigger trig;
  double tau = 0.0;
  double tv = 0.0;
  const std::int64_t dt = 1'000'000;   // 1 ms steps
  const std::int64_t steps = 200'000;  // 200 s
  for (std::int64_t k = 0; k < steps; ++k) {
    const double next =
        std::fmin(std::fmax(tau + 0.08 * rng.normal(), 0.0), 0.99);  // a violently noisy signal
    tv += std::fabs(next - tau);
    tau = next;
    trig.update(k * dt, tau, theta, hyst, cooldown);
  }
  const double T = static_cast<double>(steps * dt) * 1e-9;
  EXPECT_LE(static_cast<double>(trig.cycles), T / 0.25 + 1.0);
  EXPECT_LE(static_cast<double>(trig.cycles), tv / hyst + 1.0);
  EXPECT_GT(trig.cycles, 100U);
}

// Theorem 20 (3): requotes per side are at most T / c_rq + 1.
TEST(Throttle, RequoteCountObeysTheorem20) {
  testkit::Rng rng(53);
  control::Throttle th;
  const core::DurationNs crq = 5'000'000;
  const std::int64_t steps = 100'000;
  for (std::int64_t k = 0; k < steps; ++k) {
    const std::int64_t want = 6'000'000'000'000 + static_cast<std::int64_t>(rng.next() % 5) * 1'000'000;
    th.update(k * 100'000, want, crq);  // a new desired price every 0.1 ms
  }
  const double T = static_cast<double>(steps * 100'000) * 1e-9;
  EXPECT_LE(static_cast<double>(th.requotes), T / 0.005 + 1.0);
  EXPECT_GT(th.requotes, 1000U);
}

TEST(Rounding, AsksRoundUpBidsRoundDownToTick) {
  const std::int64_t tick = 1'000'000;  // 0.01
  EXPECT_EQ(control::round_ask_e8(100.001, tick), 10'001'000'000);
  EXPECT_EQ(control::round_bid_e8(100.009, tick), 10'000'000'000);
  // On-grid prices stay put despite binary rounding.
  EXPECT_EQ(control::round_ask_e8(60000.01, tick), 6'000'001'000'000);
  EXPECT_EQ(control::round_bid_e8(60000.01, tick), 6'000'001'000'000);
  EXPECT_EQ(control::round_ask_e8(0.07, tick), 7'000'000);
  EXPECT_EQ(control::round_bid_e8(0.07, tick), 7'000'000);
}

TEST(KillSwitch, TripsAndClearsAfterCooldown) {
  control::KillSwitch k;
  const core::DurationNs cd = 250'000'000;
  EXPECT_TRUE(k.active);  // starts killed
  EXPECT_FALSE(k.update(10'000'000'000, false, cd));
  EXPECT_TRUE(k.update(11'000'000'000, true, cd));
  EXPECT_EQ(k.trips, 1U);
  EXPECT_TRUE(k.update(11'100'000'000, false, cd));   // cooling down
  EXPECT_FALSE(k.update(11'250'000'000, false, cd));  // cleared after c_min
}

}  // namespace
