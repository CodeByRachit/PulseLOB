#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include "hawkes_sim.hpp"
#include "pulselob/hawkes/estimator.hpp"
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

struct Ev {
  std::int64_t t_ns;
  int ch;
  double g;
};

// Event times with gaps covering 1 ns .. 100 s, ties, and random marks.
std::vector<Ev> make_events(std::uint64_t seed, std::size_t n) {
  testkit::Rng rng(seed);
  std::vector<Ev> ev;
  std::int64_t t = 1'000'000'000;
  const double gaps[] = {1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0};
  for (std::size_t k = 0; k < n; ++k) {
    double gap = k % 7 == 3 ? 0.0 : rng.exponential(300.0);
    if (k % 11 == 5) {
      gap = gaps[(k / 11) % 12];
    }
    t += static_cast<std::int64_t>(std::llround(gap * 1e9));
    ev.push_back({t, static_cast<int>(rng.next() % kD), 0.5 + 1.5 * rng.uniform()});
  }
  return ev;
}

double h_direct(std::size_t q, std::size_t n, double s) {
  // R^{q,(n)} kernel: s^n / n! e^{-beta s}
  const double b = testkit::default_beta()[q];
  return (n == 0 ? 1.0 : s) * std::exp(-b * s);
}

// V1: R^{q,(n)}_j(t_k-) against direct O(N) sums, with ties (strict left limits).
TEST(KernelState, RecursionMatchesDirectSumsWithTies) {
  const auto cfg = testkit::default_engine_config().model;
  ModelT model(cfg);
  const auto ev = make_events(7, 600);
  double worst = 0.0;
  for (std::size_t k = 0; k < ev.size(); ++k) {
    ASSERT_EQ(model.step(ev[k].t_ns, ev[k].ch, ev[k].g), ModelT::StepResult::Ok);
    for (std::size_t j = 0; j < kD; ++j) {
      for (std::size_t q = 0; q < Shape::Q; ++q) {
        for (std::size_t n = 0; n < Shape::orders[q]; ++n) {
          double ref = 0.0;
          for (std::size_t l = 0; l < k; ++l) {
            if (ev[l].t_ns < ev[k].t_ns && static_cast<std::size_t>(ev[l].ch) == j) {
              ref += ev[l].g * h_direct(q, n, core::ns_to_seconds(ev[k].t_ns - ev[l].t_ns));
            }
          }
          const double got = model.kernel().R(L::row(q, n), j);
          // The cutoff drops mass below e^{-50} relative; compare with a matching absolute floor.
          const double err = std::fabs(got - ref) / std::fmax(std::fabs(ref), 1e-12);
          worst = std::fmax(worst, std::fabs(got - ref) < 1e-18 ? 0.0 : err);
        }
      }
    }
  }
  RecordProperty("worst_relative_error", testkit::sci(worst));
  EXPECT_LT(worst, 1e-10);
}

// V2: compensator increments Z(dt) for dt in [1e-9, 1e2] s against quadrature of direct sums.
TEST(KernelState, CompensatorMatchesQuadrature) {
  const auto cfg = testkit::default_engine_config().model;
  ModelT model(cfg);
  const testkit::GaussLegendre30 gl;
  const auto ev = make_events(11, 400);
  double worst = 0.0;
  std::size_t checked = 0;
  for (std::size_t k = 0; k < ev.size(); ++k) {
    const std::int64_t t_prev = model.started() ? model.t_prev() : ev[k].t_ns;
    model.step(ev[k].t_ns, ev[k].ch, ev[k].g);
    if (ev[k].t_ns == t_prev || k == 0) {
      continue;
    }
    const double dt = core::ns_to_seconds(ev[k].t_ns - t_prev);
    const auto& Z = model.last_compensator();
    EXPECT_EQ(Z[0], dt);
    for (std::size_t j = 0; j < kD; ++j) {
      for (std::size_t q = 0; q < Shape::Q; ++q) {
        const double b = testkit::default_beta()[q];
        auto x = [&](double u) {
          double s = 0.0;
          for (std::size_t l = 0; l < k; ++l) {
            if (ev[l].t_ns <= t_prev && static_cast<std::size_t>(ev[l].ch) == j) {
              s += ev[l].g *
                   testkit::erlang_kernel(Shape::orders[q], b, core::ns_to_seconds(t_prev - ev[l].t_ns) + u);
            }
          }
          return s;
        };
        const double ref = gl.integrate_decaying(x, dt, b);
        const double got = Z[Shape::coord(j, q)];
        // The cutoff (Section 4.6) drops mass below e^{-50} of a state; compare with an absolute floor.
        if (std::fabs(got - ref) > 1e-18) {
          worst = std::fmax(worst, std::fabs(got - ref) / std::fmax(ref, 1e-12));
        }
        ++checked;
      }
    }
  }
  RecordProperty("worst_relative_error", testkit::sci(worst));
  EXPECT_GT(checked, 1000U);
  EXPECT_LT(worst, 1e-10);
}

// V4: exact low-rank interval moment increment (Theorem 7), both the sparse V^T F V used by the
// statistics and the pivoted-Cholesky rank-one form used by Potter, against quadrature of
// e^{-kappa (dt - u)} z(t+u) z(t+u)^T with z from direct sums.
TEST(IntervalGram, MatchesQuadratureAndRankOneForm) {
  const auto cfg = testkit::default_engine_config().model;
  ModelT model(cfg);
  const auto ev = make_events(19, 120);
  for (const auto& e : ev) {
    model.step(e.t_ns, e.ch, e.g);
  }
  // State at t_prev+ (flush the pending batch) for the interval starting at the last event.
  hawkes::KernelState<Shape> ks = model.kernel();
  ks.flush();
  const std::int64_t t0 = model.t_prev();
  const testkit::GaussLegendre30 gl;
  for (double kappa : {cfg.kappa_slow(), cfg.kappa_fast()}) {
    for (double dt : {1e-9, 1e-7, 1e-5, 1e-3, 0.02, 0.5, 5.0, 60.0}) {
      hawkes::IntervalTerms<Shape> it{};
      compute_interval(model.bank(), dt, kappa, it);
      hawkes::LsStatistics<Shape> st{};
      hawkes::build_v(model.bank(), ks, st.V);
      st.add_interval(it);  // M starts at zero: M = V^T F V (lower)
      auto zfun = [&](double u, std::array<double, kP * kP>& out) {
        std::array<double, kP> z{};
        z[0] = 1.0;
        for (const auto& e : ev) {
          if (e.t_ns > t0) {
            continue;
          }
          const double age = core::ns_to_seconds(t0 - e.t_ns) + u;
          for (std::size_t q = 0; q < Shape::Q; ++q) {
            z[Shape::coord(static_cast<std::size_t>(e.ch), q)] +=
                e.g * testkit::erlang_kernel(Shape::orders[q], testkit::default_beta()[q], age);
          }
        }
        const double w = std::exp(-kappa * (dt - u));
        for (std::size_t a = 0; a < kP; ++a) {
          for (std::size_t b = 0; b < kP; ++b) {
            out[a * kP + b] = w * z[a] * z[b];
          }
        }
      };
      const auto ref = gl.integrate_decaying_vec<kP * kP>(zfun, dt, 2.0 * testkit::default_beta()[0]);
      double scale = 0.0;
      for (double v : ref) {
        scale = std::fmax(scale, std::fabs(v));
      }
      double worst = 0.0;
      for (std::size_t a = 0; a < kP; ++a) {
        for (std::size_t b = 0; b <= a; ++b) {
          worst = std::fmax(worst, std::fabs(st.M(a, b) - ref[a * kP + b]) / scale);
        }
      }
      EXPECT_LT(worst, 1e-10) << "dt=" << dt << " kappa=" << kappa;
      // Rank-one form: sum_r (V^T l_r)(V^T l_r)^T.
      core::FixedMatrix<L::T, L::T> lrows{};
      const std::size_t rank = hawkes::pivoted_cholesky(it.F, cfg.pivot_eta, lrows);
      EXPECT_GE(rank, 1U);
      EXPECT_LE(rank, L::T);
      double worst_r1 = 0.0;
      for (std::size_t a = 0; a < kP; ++a) {
        for (std::size_t b = 0; b <= a; ++b) {
          double s = 0.0;
          for (std::size_t r = 0; r < rank; ++r) {
            double wa = 0.0;
            double wb = 0.0;
            for (std::size_t m = 0; m < L::T; ++m) {
              wa += st.V(m, a) * lrows(r, m);
              wb += st.V(m, b) * lrows(r, m);
            }
            s += wa * wb;
          }
          worst_r1 = std::fmax(worst_r1, std::fabs(s - st.M(a, b)) / scale);
        }
      }
      EXPECT_LT(worst_r1, 1e-12) << "dt=" << dt;
    }
  }
}

// The F matrix is a Gram matrix (positive semi-definite) and its numerical rank adapts to the
// interval length (math_spec Q4).
TEST(IntervalGram, RankAdaptsToIntervalLength) {
  const auto cfg = testkit::default_engine_config().model;
  const hawkes::KernelBank<Shape> bank(cfg.beta, cfg.x_cut);
  std::size_t prev_rank = 0;
  for (double dt : {1e-7, 1e-5, 1e-3, 0.1, 10.0}) {
    hawkes::IntervalTerms<Shape> it{};
    compute_interval(bank, dt, cfg.kappa_slow(), it);
    core::FixedMatrix<L::T, L::T> lrows{};
    const std::size_t rank = hawkes::pivoted_cholesky(it.F, cfg.pivot_eta, lrows);
    EXPECT_GE(rank, prev_rank) << "dt=" << dt;
    prev_rank = rank;
  }
  EXPECT_EQ(prev_rank, L::T);
}

}  // namespace
