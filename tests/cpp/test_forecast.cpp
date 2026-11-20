#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "hawkes_sim.hpp"
#include "pulselob/hawkes/forecast.hpp"
#include "pulselob/hawkes/model.hpp"
#include "quadrature.hpp"
#include "report.hpp"
#include "test_config.hpp"

namespace {

using namespace pulselob;
using testkit::kD;
using testkit::kP;
using testkit::Shape;
using L = hawkes::BankLayout<Shape>;
using ModelT = hawkes::Model<Shape, hawkes::CholeskyEstimator<Shape>>;

double survival(std::size_t r, double beta, double s) {
  return r == 1 ? std::exp(-beta * s) : (1.0 + beta * s) * std::exp(-beta * s);
}

struct Ev {
  std::int64_t t;
  int ch;
  double g;
};

// V9: m^(0), m-bar^(0), y^inf against direct sums over the history and X^inf against a dense solve.
TEST(Forecast, MatchesDirectSumsAndDenseSolve) {
  const auto ecfg = testkit::default_engine_config();
  auto model = std::make_unique<ModelT>(ecfg.model);
  hawkes::ForecastConfig<kD> fcfg = ecfg.forecast;
  hawkes::Forecaster<Shape> fc;
  fc.init(model->bank(), fcfg);
  hawkes::ForecastConfig<kD> fbig = fcfg;
  fbig.neumann_terms = 5000;  // 0.97^5000 ~ 1e-66: converged (validate() caps K at 64 for the engine)
  hawkes::Forecaster<Shape> fc_big;
  fc_big.init(model->bank(), fbig);
  testkit::Rng rng(29);
  testkit::HawkesSimulator<Shape> sim(testkit::default_beta(), testkit::ground_truth_theta(0.3));
  const auto sev = sim.simulate(20.0, rng);
  std::vector<Ev> ev;
  for (const auto& e : sev) {
    ev.push_back({1'000'000'000 + static_cast<std::int64_t>(std::llround(e.t_s * 1e9)), e.channel,
                  0.5 + rng.uniform()});
  }
  const double H = fcfg.horizon_s;
  const auto beta = testkit::default_beta();
  const testkit::GaussLegendre30 gl;
  double worst_m0 = 0.0;
  double worst_bar = 0.0;
  double worst_y = 0.0;
  double worst_x = 0.0;
  for (std::size_t k = 0; k < ev.size(); ++k) {
    model->step(ev[k].t, ev[k].ch, ev[k].g);
    if (k % 97 != 0) {
      continue;
    }
    core::FixedMatrix<L::S, kD> rpost{};
    model->kernel().post_event(rpost);
    hawkes::Forecast<kD> f{};
    fc.compute(model->bank(), rpost, model->served(), f);
    const auto& s = model->served();
    for (std::size_t i = 0; i < kD; ++i) {
      double m0 = s.theta(i, 0) * H;
      double mb = s.theta(i, 0) * H;
      double y = 0.0;
      for (std::size_t j = 0; j < kD; ++j) {
        for (std::size_t q = 0; q < Shape::Q; ++q) {
          const std::size_t r = Shape::orders[q];
          double mass_h = 0.0;
          double mass_inf = 0.0;
          for (std::size_t l = 0; l <= k; ++l) {
            if (static_cast<std::size_t>(ev[l].ch) != j) {
              continue;
            }
            const double a = core::ns_to_seconds(ev[k].t - ev[l].t);
            mass_h += ev[l].g * (survival(r, beta[q], a) - survival(r, beta[q], a + H));
            mass_inf += ev[l].g * survival(r, beta[q], a);
          }
          // Stationary: lambda_j int_0^H S_q(u) du (Campbell means), by quadrature.
          const double kbar =
              gl.integrate_decaying([&](double u) { return survival(r, beta[q], u); }, H, beta[q]);
          const double alpha = s.theta(i, Shape::coord(j, q));
          m0 += alpha * mass_h;
          mb += alpha * s.lambda_inf[j] * kbar;
          y += alpha * (mass_inf - s.lambda_inf[j] * static_cast<double>(r) / beta[q]);
        }
      }
      worst_m0 = std::fmax(worst_m0, std::fabs(f.m0[i] - m0) / m0);
      worst_bar = std::fmax(worst_bar, std::fabs(f.m0_bar[i] - mb) / mb);
      worst_y = std::fmax(worst_y, std::fabs(f.y_inf[i] - y) / std::fmax(std::fabs(y), 1e-3));
    }
    // X^inf = (I - G)^{-1} y by a dense solve against the K = 64 Neumann series.
    hawkes::Forecast<kD> fb{};
    fc_big.compute(model->bank(), rpost, s, fb);
    hawkes::Served<Shape> tmp = s;
    for (std::size_t i = 0; i < kD; ++i) {
      tmp.theta(i, 0) = fb.y_inf[i];  // reuse the dense (I - G)^{-1} b solver with b = y
    }
    hawkes::StabilityGuard<Shape>::solve_lambda_inf(tmp);
    for (std::size_t i = 0; i < kD; ++i) {
      worst_x = std::fmax(worst_x, std::fabs(fb.x_inf[i] - tmp.lambda_inf[i]) /
                                       std::fmax(std::fabs(tmp.lambda_inf[i]), 1e-3));
    }
  }
  RecordProperty("m0", testkit::sci(worst_m0));
  RecordProperty("m0_bar", testkit::sci(worst_bar));
  RecordProperty("y_inf", testkit::sci(worst_y));
  RecordProperty("x_inf_converged", testkit::sci(worst_x));
  EXPECT_LT(worst_m0, 1e-10);
  EXPECT_LT(worst_bar, 1e-10);
  EXPECT_LT(worst_y, 1e-10);
  EXPECT_LT(worst_x, 1e-10);
}

// Proposition 17 (1)-(5) on the score itself.
TEST(Toxicity, SatisfiesProposition17) {
  testkit::Rng rng(31);
  for (int trial = 0; trial < 20000; ++trial) {
    const double mp = 10.0 * rng.uniform();
    const double mm = 10.0 * rng.uniform();
    const double bp = 0.1 + 5.0 * rng.uniform();
    const double bm = 0.1 + 5.0 * rng.uniform();
    double ep = 0;
    double em = 0;
    double tp = 0;
    double tm = 0;
    hawkes::toxicity(mp, mm, bp, bm, ep, em, tp, tm);
    EXPECT_GE(tp, 0.0);
    EXPECT_LT(tp, 1.0);
    EXPECT_GE(tm, 0.0);
    EXPECT_LT(tm, 1.0);
    EXPECT_EQ(tp * tm, 0.0);  // exclusivity
    // Unit invariance (scaling eta, or forecasts and baselines together).
    double ep2 = 0;
    double em2 = 0;
    double tp2 = 0;
    double tm2 = 0;
    hawkes::toxicity(3.7 * mp, 3.7 * mm, 3.7 * bp, 3.7 * bm, ep2, em2, tp2, tm2);
    EXPECT_NEAR(tp2, tp, 1e-14);
    EXPECT_NEAR(tm2, tm, 1e-14);
    // Monotonicity in M^+ when E^+ > E^-.
    if (ep > em) {
      double e3 = 0;
      double e4 = 0;
      double tp3 = 0;
      double tm3 = 0;
      hawkes::toxicity(mp * 1.1, mm, bp, bm, e3, e4, tp3, tm3);
      EXPECT_GE(tp3, tp - 1e-15);
    }
  }
  // Two-sided invariance: equal excesses score zero however large the volume.
  double ep = 0;
  double em = 0;
  double tp = 0;
  double tm = 0;
  hawkes::toxicity(1e6 + 1.0, 1e6 + 1.0, 1.0, 1.0, ep, em, tp, tm);
  EXPECT_EQ(tp, 0.0);
  EXPECT_EQ(tm, 0.0);
  hawkes::toxicity(0.0, 0.0, 1.0, 1.0, ep, em, tp, tm);  // 0/0 := 0
  EXPECT_EQ(tp, 0.0);
}

// Proposition 17 (6): a perp-buy burst raises the spot-buy forecast and tau^+ before any spot trade,
// through the Erlang-2 hump whose intensity is still almost zero at t+. Parameters are held fixed
// (the prior) and both states are evaluated at the same time, so only the burst differs.
TEST(Toxicity, LeadLagBurstRaisesSpotForecastFirst) {
  const auto ecfg = testkit::default_engine_config();
  auto quiet = std::make_unique<ModelT>(ecfg.model);
  auto burst = std::make_unique<ModelT>(ecfg.model);
  const hawkes::Served<Shape> s0 = quiet->served();  // served prior: alpha(spot buy <- perp buy, 5 ms) = 0.15
  hawkes::Forecaster<Shape> fc;
  fc.init(quiet->bank(), ecfg.forecast);
  std::int64_t t = 1'000'000'000;
  for (int k = 0; k < 400; ++k) {  // identical two-sided history
    t += 25'000'000;
    quiet->step(t, k % 4, 1.0);
    burst->step(t, k % 4, 1.0);
  }
  t += 100'000;
  for (int k = 0; k < 10; ++k) {  // ten perp buys at one instant (a sweep batch)
    burst->step(t, 0, 1.0);
  }
  core::FixedMatrix<L::S, kD> rq{};
  core::FixedMatrix<L::S, kD> rb{};
  quiet->kernel().post_event_at(quiet->bank(), core::ns_to_seconds(t - quiet->t_prev()), rq);
  burst->kernel().post_event(rb);
  hawkes::Forecast<kD> fq{};
  hawkes::Forecast<kD> fb{};
  fc.compute(quiet->bank(), rq, s0, fq);
  fc.compute(burst->bank(), rb, s0, fb);
  EXPECT_GT(fb.m0[2], fq.m0[2] * 1.05);  // spot-buy forecast up before any spot trade
  EXPECT_GT(fb.tau_buy, 0.0);
  EXPECT_EQ(fb.tau_sell, 0.0);
  EXPECT_GT(fb.x_inf[2], fq.x_inf[2]);
  // At t+ the Erlang-2 hump's contribution to the instantaneous intensity has not moved at all
  // (R^(1) does not jump), yet the horizon forecast already includes the hump mass still to come.
  EXPECT_EQ(rb(L::row(1, 1), 0), rq(L::row(1, 1), 0));
  EXPECT_EQ(rb(L::row(2, 1), 0), rq(L::row(2, 1), 0));
}

}  // namespace
