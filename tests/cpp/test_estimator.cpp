#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "hawkes_sim.hpp"
#include "pulselob/core/linalg.hpp"
#include "pulselob/hawkes/estimator.hpp"
#include "pulselob/hawkes/model.hpp"
#include "report.hpp"
#include "test_config.hpp"

namespace {

using namespace pulselob;
using testkit::kD;
using testkit::kP;
using testkit::Shape;
using Potter = hawkes::Model<Shape, hawkes::PotterEstimator<Shape>>;
using Chol = hawkes::Model<Shape, hawkes::CholeskyEstimator<Shape>>;

bool long_tests() {
#if defined(NDEBUG) && !defined(__SANITIZE_ADDRESS__) && !defined(__SANITIZE_THREAD__)
#if defined(__has_feature)
#if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer)
  return false;
#endif
#endif
  return true;
#else
  return false;
#endif
}

struct Step {
  std::int64_t t;
  int ch;  // -1 heartbeat
  double g;
};

// Simulated channel events with ties, heartbeats every 100 ms of silence, and two long gaps
// (2 s and 500 s without heartbeats) that take the full-restore path.
std::vector<Step> make_stream(std::uint64_t seed, double horizon_s, bool silent_channel3 = false) {
  testkit::Rng rng(seed);
  auto theta = testkit::ground_truth_theta(0.3);
  if (silent_channel3) {
    theta(3, 0) = 0.0;
    for (std::size_t p = 1; p < kP; ++p) {
      theta(3, p) = 0.0;
    }
  }
  testkit::HawkesSimulator<Shape> sim(testkit::default_beta(), theta);
  const auto ev = sim.simulate(horizon_s, rng);
  std::vector<Step> out;
  std::int64_t shift = 0;
  std::int64_t last = 0;
  for (std::size_t k = 0; k < ev.size(); ++k) {
    if (k == ev.size() / 3) {
      shift += 2'000'000'000;
    }
    if (k == 2 * ev.size() / 3) {
      shift += 500'000'000'000;
    }
    std::int64_t t = 1'000'000'000 + shift + static_cast<std::int64_t>(std::llround(ev[k].t_s * 1e9));
    if (k % 97 == 13 && !out.empty()) {
      t = out.back().t;  // cross-channel tie
    }
    while (k > 0 && t - last > 100'000'000 && k != ev.size() / 3 && k != 2 * ev.size() / 3) {
      last += 100'000'000;
      out.push_back({last, -1, 0.0});
    }
    t = t < last ? last : t;
    out.push_back({t, ev[k].channel, 0.5 + rng.uniform()});
    last = t;
  }
  return out;
}

double rel_diff(const core::FixedMatrix<kD, kP>& a, const core::FixedMatrix<kD, kP>& b) {
  double num = 0.0;
  double den = 0.0;
  for (std::size_t i = 0; i < kD; ++i) {
    for (std::size_t p = 0; p < kP; ++p) {
      num = std::fmax(num, std::fabs(a(i, p) - b(i, p)));
      den = std::fmax(den, std::fabs(b(i, p)));
    }
  }
  return num / den;
}

// Dense solve of Theta = C M^{-1} from the stored statistics (Gaussian elimination, partial pivoting).
core::FixedMatrix<kD, kP> dense_theta(const hawkes::LsStatistics<Shape>& st) {
  const auto M = st.symmetric_M();
  std::vector<double> A(kP * (kP + kD));
  for (std::size_t i = 0; i < kP; ++i) {
    for (std::size_t j = 0; j < kP; ++j) {
      A[i * (kP + kD) + j] = M(i, j);
    }
    for (std::size_t r = 0; r < kD; ++r) {
      A[i * (kP + kD) + kP + r] = st.C(r, i);
    }
  }
  const std::size_t W = kP + kD;
  for (std::size_t k = 0; k < kP; ++k) {
    std::size_t piv = k;
    for (std::size_t i = k + 1; i < kP; ++i) {
      if (std::fabs(A[i * W + k]) > std::fabs(A[piv * W + k])) {
        piv = i;
      }
    }
    for (std::size_t j = 0; j < W; ++j) {
      std::swap(A[k * W + j], A[piv * W + j]);
    }
    for (std::size_t i = k + 1; i < kP; ++i) {
      const double f = A[i * W + k] / A[k * W + k];
      for (std::size_t j = k; j < W; ++j) {
        A[i * W + j] -= f * A[k * W + j];
      }
    }
  }
  core::FixedMatrix<kD, kP> th{};
  for (std::size_t r = 0; r < kD; ++r) {
    for (std::size_t kk = kP; kk-- > 0;) {
      double s = A[kk * W + kP + r];
      for (std::size_t j = kk + 1; j < kP; ++j) {
        s -= A[kk * W + j] * th(r, j);
      }
      th(r, kk) = s / A[kk * W + kk];
    }
  }
  return th;
}

TEST(Estimator, StartsAtThePrior) {
  const auto cfg = testkit::default_engine_config().model;
  auto p = std::make_unique<Potter>(cfg);
  auto c = std::make_unique<Chol>(cfg);
  EXPECT_EQ(rel_diff(p->estimator().theta(), cfg.theta0), 0.0);
  EXPECT_EQ(rel_diff(c->estimator().theta(), cfg.theta0), 0.0);
}

// Both back-ends against each other and against a dense solve of the exact statistics, through
// ties, heartbeats, CUSUM-driven forgetting switches and two long gaps.
TEST(Estimator, BackendsAgreeWithEachOtherAndWithDenseSolve) {
  auto cfg = testkit::default_engine_config().model;
  cfg.cusum_threshold = 6.0;  // make forgetting switches happen
  auto p = std::make_unique<Potter>(cfg);
  auto c = std::make_unique<Chol>(cfg);
  const auto stream = make_stream(3, 120.0);
  double worst_pc = 0.0;
  double worst_dense = 0.0;
  for (std::size_t k = 0; k < stream.size(); ++k) {
    const auto& s = stream[k];
    ASSERT_EQ(p->step(s.t, s.ch, s.g), Potter::StepResult::Ok);
    ASSERT_EQ(c->step(s.t, s.ch, s.g), Chol::StepResult::Ok);
    if (k % 50 == 0 || k + 1 == stream.size()) {
      worst_pc = std::fmax(worst_pc, rel_diff(p->estimator().theta(), c->estimator().theta()));
      worst_dense =
          std::fmax(worst_dense, rel_diff(c->estimator().theta(), dense_theta(c->estimator().stats())));
    }
  }
  EXPECT_GT(c->counters().alarms, 0U);
  EXPECT_EQ(c->counters().long_gaps, 2U);
  EXPECT_EQ(p->estimator().factorization_failures(), 0U);
  EXPECT_EQ(c->estimator().factorization_failures(), 0U);
  RecordProperty("potter_vs_cholesky", testkit::sci(worst_pc));
  RecordProperty("cholesky_vs_dense", testkit::sci(worst_dense));
  EXPECT_LT(worst_pc, 1e-10);
  EXPECT_LT(worst_dense, 1e-10);
  // Both models accumulate identical statistics (same operations in the same order).
  EXPECT_EQ(0, std::memcmp(&p->estimator().stats().M, &c->estimator().stats().M,
                           sizeof(core::FixedMatrix<kP, kP>)));
}

// V6: U U^T M - I stays small over a long stream (1e7 events in optimized builds, 1e5 otherwise).
TEST(Estimator, PotterFactorStaysInverseOfMomentMatrix) {
  const auto cfg = testkit::default_engine_config().model;
  auto p = std::make_unique<Potter>(cfg);
  testkit::Rng rng(5);
  testkit::HawkesSimulator<Shape> sim(testkit::default_beta(), testkit::ground_truth_theta());
  const std::size_t target = long_tests() ? 10'000'000 : 100'000;
  std::size_t done = 0;
  std::int64_t offset = 1'000'000'000;
  double worst = 0.0;
  while (done < target) {
    const auto ev = sim.simulate(100.0, rng, target - done);
    for (const auto& e : ev) {
      p->step(offset + static_cast<std::int64_t>(std::llround(e.t_s * 1e9)), e.channel, 1.0);
    }
    offset += 100'000'000'000;
    done += ev.size();
    const auto M = p->estimator().stats().symmetric_M();
    const auto& U = p->estimator().U();
    for (std::size_t i = 0; i < kP; ++i) {
      for (std::size_t j = 0; j < kP; ++j) {
        double s = 0.0;
        for (std::size_t k = 0; k < kP; ++k) {
          double uu = 0.0;
          for (std::size_t m = 0; m < kP; ++m) {
            uu += U(i, m) * U(k, m);
          }
          s += uu * M(k, j);
        }
        worst = std::fmax(worst, std::fabs(s - (i == j ? 1.0 : 0.0)));
      }
    }
  }
  RecordProperty("events", testkit::sci(done));
  RecordProperty("worst_UUtM_minus_I", testkit::sci(worst));
  EXPECT_LT(worst, 1e-8);
}

// Proposition 9: a channel that never fires keeps its column at the prior (no wind-up), and M stays
// above the ridge floor min(omega) e^{-kappa_max P T_hb}.
TEST(Estimator, SilentChannelFallsBackToPriorWithoutWindup) {
  const auto cfg = testkit::default_engine_config().model;
  auto c = std::make_unique<Chol>(cfg);
  const auto stream = make_stream(9, 200.0, /*silent_channel3=*/true);
  double omega_min = cfg.omega[0];
  for (std::size_t p = 0; p < kP; ++p) {
    omega_min = std::fmin(omega_min, cfg.omega[p]);
  }
  const double floor = omega_min * std::exp(-cfg.kappa_fast() * static_cast<double>(kP) * 0.1);
  for (std::size_t k = 0; k < stream.size(); ++k) {
    ASSERT_NE(stream[k].ch, 3);
    c->step(stream[k].t, stream[k].ch, stream[k].g);
    if (k % 500 == 0) {
      auto M = c->estimator().stats().M;
      for (std::size_t p = 0; p < kP; ++p) {
        M(p, p) -= 0.999 * floor;
      }
      core::FixedVector<kP> col{};
      EXPECT_TRUE(core::cholesky_lower(M, col)) << "M below the ridge floor at step " << k;
    }
  }
  for (std::size_t i = 0; i < kD; ++i) {
    for (std::size_t q = 0; q < Shape::Q; ++q) {
      const std::size_t p = Shape::coord(3, q);
      EXPECT_NEAR(c->estimator().theta()(i, p), cfg.theta0(i, p), 1e-9 + 1e-9 * cfg.theta0(i, p));
    }
  }
}

}  // namespace
