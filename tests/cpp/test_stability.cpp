#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <vector>

#include "hawkes_sim.hpp"
#include "pulselob/hawkes/stability.hpp"
#include "report.hpp"
#include "test_config.hpp"

namespace {

using namespace pulselob;
using testkit::kD;
using testkit::kP;
using testkit::Shape;
constexpr std::size_t kN = kD * Shape::Q;

// Reference: exact projection by sorting the breakpoints (O(n log n)).
std::array<double, kN> project_sorted(const std::array<double, kN>& at, const std::array<double, kN>& w,
                                      double c, double& nu_out) {
  std::array<double, kN> out{};
  double s0 = 0.0;
  for (std::size_t m = 0; m < kN; ++m) {
    s0 += w[m] * std::max(at[m], 0.0);
  }
  nu_out = 0.0;
  if (s0 <= c) {
    for (std::size_t m = 0; m < kN; ++m) {
      out[m] = std::max(at[m], 0.0);
    }
    return out;
  }
  std::vector<std::size_t> idx(kN);
  std::iota(idx.begin(), idx.end(), 0U);
  std::sort(idx.begin(), idx.end(),
            [&](std::size_t a, std::size_t b) { return at[a] / w[a] > at[b] / w[b]; });
  double sa = 0.0;
  double sw = 0.0;
  double nu = 0.0;
  for (std::size_t k = 0; k < kN; ++k) {
    const std::size_t m = idx[k];
    if (at[m] <= 0.0) {
      break;
    }
    sa += w[m] * at[m];
    sw += w[m] * w[m];
    nu = (sa - c) / sw;
    const double next = k + 1 < kN ? at[idx[k + 1]] / w[idx[k + 1]] : -1.0;
    if (nu >= next) {
      break;
    }
  }
  nu_out = nu;
  for (std::size_t m = 0; m < kN; ++m) {
    out[m] = std::max(at[m] - nu * w[m], 0.0);
  }
  return out;
}

// V7: the breakpoint projection against the sorting reference, and its KKT residuals.
TEST(Projection, MatchesSortingReferenceAndSatisfiesKkt) {
  testkit::Rng rng(13);
  double worst_ref = 0.0;
  double worst_kkt = 0.0;
  std::size_t active = 0;
  for (int trial = 0; trial < 3000; ++trial) {
    std::array<double, kN> at{};
    std::array<double, kN> w{};
    for (std::size_t m = 0; m < kN; ++m) {
      at[m] = rng.normal() * 0.3 + (trial % 3 == 0 ? 0.2 : 0.0);
      w[m] = 0.5 + rng.uniform();
    }
    const double c = 0.2 + rng.uniform();
    std::array<double, kN> a{};
    const double nu = hawkes::project_row(at, w, c, a);
    double nu_ref = 0.0;
    const auto ref = project_sorted(at, w, c, nu_ref);
    active += nu > 0.0 ? 1U : 0U;
    double wa = 0.0;
    for (std::size_t m = 0; m < kN; ++m) {
      worst_ref = std::max(worst_ref, std::fabs(a[m] - ref[m]));
      wa += w[m] * a[m];
      // Stationarity / complementary slackness: a_m = (at_m - nu w_m)_+.
      worst_kkt = std::max(worst_kkt, std::fabs(a[m] - std::max(at[m] - nu * w[m], 0.0)));
      EXPECT_GE(a[m], 0.0);
    }
    EXPECT_LE(wa, c * (1.0 + 1e-15));
    if (nu > 0.0) {
      worst_kkt = std::max(worst_kkt, std::fabs(wa - c));  // constraint active
    }
  }
  RecordProperty("worst_vs_sorting", testkit::sci(worst_ref));
  RecordProperty("worst_kkt", testkit::sci(worst_kkt));
  EXPECT_GT(active, 1000U);
  EXPECT_LT(worst_ref, 1e-12);
  EXPECT_LT(worst_kkt, 1e-12);
}

double cw_upper(const core::FixedMatrix<kD, kD>& G, const std::array<double, kD>& v) {
  double r = 0.0;
  for (std::size_t i = 0; i < kD; ++i) {
    double s = 0.0;
    for (std::size_t j = 0; j < kD; ++j) {
      s += G(i, j) * v[j];
    }
    r = std::max(r, s / v[i]);
  }
  return r;
}

// Lower Collatz-Wielandt bound at a power-iterated vector: min_i (G v)_i / v_i <= rho(G).
double cw_lower_power(const core::FixedMatrix<kD, kD>& G) {
  std::array<double, kD> v{};
  v.fill(1.0);
  for (int it = 0; it < 5000; ++it) {
    std::array<double, kD> y{};
    double mx = 0.0;
    for (std::size_t i = 0; i < kD; ++i) {
      for (std::size_t j = 0; j < kD; ++j) {
        y[i] += G(i, j) * v[j];
      }
      mx = std::max(mx, y[i]);
    }
    if (!(mx > 0.0)) {
      return 0.0;
    }
    for (std::size_t i = 0; i < kD; ++i) {
      v[i] = y[i] / mx + 1e-300;
    }
  }
  double r = 1e300;
  for (std::size_t i = 0; i < kD; ++i) {
    double s = 0.0;
    for (std::size_t j = 0; j < kD; ++j) {
      s += G(i, j) * v[j];
    }
    r = std::min(r, s / v[i]);
  }
  return r;
}

// V8: the Collatz-Wielandt bound with any positive v is an upper bound on rho (Lemma 13).
TEST(CollatzWielandt, BoundsSpectralRadius) {
  testkit::Rng rng(17);
  for (int trial = 0; trial < 3000; ++trial) {
    core::FixedMatrix<kD, kD> G{};
    for (std::size_t i = 0; i < kD; ++i) {
      for (std::size_t j = 0; j < kD; ++j) {
        G(i, j) = rng.uniform() * (trial % 2 == 0 ? 1.0 : 0.3) + 1e-3;
      }
    }
    std::array<double, kD> v{};
    for (auto& x : v) {
      x = 0.1 + rng.uniform();
    }
    const double lower = cw_lower_power(G);
    EXPECT_GE(cw_upper(G, v), lower * (1.0 - 1e-14));
  }
}

// Served parameters satisfy the certificate at every step, including when the data look
// supercritical, and the baselines stay in [mu_min, mu_max].
TEST(StabilityGuard, ServedModelIsCertifiedSubcritical) {
  auto cfg = testkit::default_engine_config().model;
  hawkes::StabilityGuard<Shape> guard;
  hawkes::Served<Shape> s{};
  guard.init(cfg, s);
  testkit::Rng rng(23);
  std::size_t projected = 0;
  bool saw_supercritical = false;
  for (int step = 0; step < 4000; ++step) {
    core::FixedMatrix<kD, kP> tt{};
    const double scale = step % 4 == 0 ? 1.5 : 0.08;
    for (std::size_t i = 0; i < kD; ++i) {
      tt(i, 0) = (step % 7 == 0) ? -5.0 : 10.0 * rng.uniform();
      for (std::size_t p = 1; p < kP; ++p) {
        tt(i, p) = scale * rng.uniform() - 0.02;
      }
    }
    guard.serve(tt, s);
    projected += s.projected_rows != 0 ? 1U : 0U;
    saw_supercritical = saw_supercritical || s.criticality >= 1.0;
    std::array<double, kD> v{};
    for (std::size_t i = 0; i < kD; ++i) {
      v[i] = s.v[i];
      EXPECT_GE(s.theta(i, 0), cfg.mu_min);
      EXPECT_LE(s.theta(i, 0), cfg.mu_max);
      for (std::size_t p = 1; p < kP; ++p) {
        EXPECT_GE(s.theta(i, p), 0.0);
      }
      // v stays in the box B.
      EXPECT_GE(v[i], guard.v_ref()[i] / std::sqrt(cfg.chi) * (1.0 - 1e-15));
      EXPECT_LE(v[i], guard.v_ref()[i] * std::sqrt(cfg.chi) * (1.0 + 1e-15));
    }
    EXPECT_LE(cw_upper(s.G, v), cfg.rho_max * (1.0 + 1e-14));
    EXPECT_LE(cw_lower_power(s.G), cfg.rho_max * (1.0 + 1e-12));
  }
  EXPECT_GT(projected, 500U);
  EXPECT_TRUE(saw_supercritical);
}

// The per-step Jacobi sweep tracks lambda_inf = (I - G)^{-1} mu for fixed parameters.
TEST(StabilityGuard, JacobiSweepConvergesToStationaryIntensity) {
  const auto cfg = testkit::default_engine_config().model;
  hawkes::StabilityGuard<Shape> guard;
  hawkes::Served<Shape> s{};
  guard.init(cfg, s);
  auto tt = testkit::ground_truth_theta(0.4);
  for (int k = 0; k < 2000; ++k) {
    guard.serve(tt, s);
  }
  hawkes::Served<Shape> exact = s;
  hawkes::StabilityGuard<Shape>::solve_lambda_inf(exact);
  for (std::size_t i = 0; i < kD; ++i) {
    EXPECT_NEAR(s.lambda_inf[i], exact.lambda_inf[i], 1e-10 * exact.lambda_inf[i]);
  }
}

}  // namespace
