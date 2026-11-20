#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "pulselob/core/fixed_matrix.hpp"
#include "pulselob/core/linalg.hpp"
#include "pulselob/hawkes/config.hpp"
#include "pulselob/hawkes/kernel.hpp"

namespace pulselob::hawkes {

// Pivoted Cholesky factorization F = sum_r l_r l_r^T of the (M+1) x (M+1) interval Gram matrix,
// stopping when the largest remaining diagonal entry is at most eta * tr(F) (math_spec Theorem 7).
// Returns the numerical rank; rows [0, rank) of out hold the l_r.
template <std::size_t T>
inline std::size_t pivoted_cholesky(const core::FixedMatrix<T, T>& F, double eta,
                                    core::FixedMatrix<T, T>& out) noexcept {
  std::array<double, T * T> A{};
  double tr = 0.0;
  for (std::size_t i = 0; i < T; ++i) {
    for (std::size_t j = 0; j < T; ++j) {
      A[i * T + j] = F(i, j);
    }
    tr += F(i, i);
  }
  std::size_t rank = 0;
  for (std::size_t it = 0; it < T; ++it) {
    std::size_t p = 0;
    for (std::size_t i = 1; i < T; ++i) {
      if (A[i * T + i] > A[p * T + p]) {
        p = i;
      }
    }
    const double app = A[p * T + p];
    if (!(app > eta * tr)) {
      break;
    }
    const double inv = 1.0 / std::sqrt(app);
    double* l = out.data.data() + rank * out.stride;
    for (std::size_t i = 0; i < T; ++i) {
      l[i] = A[i * T + p] * inv;
    }
    for (std::size_t i = 0; i < T; ++i) {
      for (std::size_t j = 0; j < T; ++j) {
        A[i * T + j] -= l[i] * l[j];
      }
    }
    ++rank;
  }
  return rank;
}

// The exact discounted sufficient statistics of math_spec Section 5.4, stored in information form:
// the moment matrix M (lower triangle) including the ridge, the cross moments C, and the effective
// ridge. Both estimator back-ends maintain these; they are the definition the online estimate must
// equal (Theorem 10).
template <class Shape>
struct LsStatistics {
  using L = BankLayout<Shape>;
  static constexpr std::size_t D = L::D;
  static constexpr std::size_t P = L::P;
  static constexpr std::size_t T = L::T;

  core::FixedMatrix<P, P> M{};  // lower triangle authoritative
  core::FixedMatrix<D, P> C{};
  core::FixedMatrix<D, P> theta0{};
  core::FixedVector<P> omega{};
  core::FixedVector<P> omega_eff{};
  core::FixedMatrix<T, P> V{};  // interval coefficients (scratch, sparse pattern)
  core::FixedMatrix<T, P> W{};  // F V (scratch)

  void init(const ModelConfig<Shape>& cfg) noexcept {
    theta0 = cfg.theta0;
    omega = cfg.omega;
    omega_eff = cfg.omega;
    M = {};
    for (std::size_t p = 0; p < P; ++p) {
      M(p, p) = omega[p];
    }
    for (std::size_t i = 0; i < D; ++i) {
      for (std::size_t p = 0; p < P; ++p) {
        C(i, p) = theta0(i, p) * omega[p];
      }
    }
  }

  // Forgetting over an interval: M <- d M, C <- d C, omega_eff <- d omega_eff.
  void forget(double d) noexcept {
    core::scale(M, d);
    core::scale(C, d);
    core::scale(omega_eff, d);
  }

  // M += V^T F V on the lower triangle, exploiting the sparsity of V (each coordinate has at most
  // r_max nonzero coefficients). Requires V to be built for the interval's starting state.
  void add_interval(const IntervalTerms<Shape>& it) noexcept {
    // W = F V, column by column.
    for (std::size_t m = 0; m < T; ++m) {
      W(m, 0) = it.F(m, 0);
    }
    for (std::size_t j = 0; j < D; ++j) {
      for (std::size_t q = 0; q < L::Q; ++q) {
        const std::size_t b = Shape::coord(j, q);
        for (std::size_t m = 0; m < T; ++m) {
          double s = it.F(m, L::tb(q, 0)) * V(L::tb(q, 0), b);
          if (L::order(q) == 2) {
            s += it.F(m, L::tb(q, 1)) * V(L::tb(q, 1), b);
          }
          W(m, b) = s;
        }
      }
    }
    // M(a, b) += sum_m V(m, a) W(m, b) for b <= a.
    M(0, 0) += W(0, 0);
    for (std::size_t j = 0; j < D; ++j) {
      for (std::size_t q = 0; q < L::Q; ++q) {
        const std::size_t a = Shape::coord(j, q);
        const double v0 = V(L::tb(q, 0), a);
        const double v1 = L::order(q) == 2 ? V(L::tb(q, 1), a) : 0.0;
        const double* w0 = W.data.data() + L::tb(q, 0) * W.stride;
        const double* w1 = W.data.data() + (L::order(q) == 2 ? L::tb(q, 1) : 0) * W.stride;
        double* row = M.data.data() + a * M.stride;
        if (L::order(q) == 2) {
          for (std::size_t b = 0; b <= a; ++b) {
            row[b] += v0 * w0[b] + v1 * w1[b];
          }
        } else {
          for (std::size_t b = 0; b <= a; ++b) {
            row[b] += v0 * w0[b];
          }
        }
      }
    }
  }

  void add_event(std::size_t i, const core::FixedVector<P>& z) noexcept {
    core::axpy_lanes<core::FixedMatrix<D, P>::stride>(C.data.data() + i * C.stride, 1.0, z.data.data());
  }

  // Restores coordinate p of the ridge to omega_p (math_spec Section 5.4). Returns the amount added.
  double replenish(std::size_t p) noexcept {
    const double a = omega[p] - omega_eff[p];
    if (a > 0.0) {
      M(p, p) += a;
      for (std::size_t i = 0; i < D; ++i) {
        C(i, p) += a * theta0(i, p);
      }
    }
    omega_eff[p] = omega[p];
    return a;
  }

  // Full symmetric copy of M (for tests and diagnostics).
  core::FixedMatrix<P, P> symmetric_M() const noexcept {
    core::FixedMatrix<P, P> S{};
    for (std::size_t a = 0; a < P; ++a) {
      for (std::size_t b = 0; b <= a; ++b) {
        S(a, b) = M(a, b);
        S(b, a) = M(a, b);
      }
    }
    return S;
  }
};

// Back-end 1 (math_spec Section 5.4, the spec's choice): square-root covariance form. Maintains U
// with U U^T = M^{-1} by Potter rank-one updates, and Theta~ incrementally; one row of Theta~ is
// recomputed exactly from C per step to remove drift. Long gaps (no event or heartbeat for more
// than T_hb) take the refactor path instead: after the full ridge restore U is rebuilt from M by
// Cholesky, which avoids Potter updates on a covariance inflated by 1/d.
template <class Shape>
class PotterEstimator {
 public:
  using L = BankLayout<Shape>;
  static constexpr std::size_t D = L::D;
  static constexpr std::size_t P = L::P;
  static constexpr std::size_t T = L::T;
  static constexpr const char* kName = "potter";

  void init(const ModelConfig<Shape>& cfg) noexcept {
    stats_.init(cfg);
    eta_ = cfg.pivot_eta;
    U_ = {};
    for (std::size_t p = 0; p < P; ++p) {
      U_(p, p) = 1.0 / std::sqrt(cfg.omega[p]);
    }
    theta_ = cfg.theta0;
  }

  void interval(const KernelBank<Shape>& bank, const KernelState<Shape>& ks, const IntervalTerms<Shape>& it,
                bool long_gap) noexcept {
    build_v(bank, ks, stats_.V);
    stats_.forget(it.d);
    stats_.add_interval(it);
    if (long_gap) {
      return;  // U and Theta~ are rebuilt by full_restore()
    }
    core::scale(U_, 1.0 / std::sqrt(it.d));
    const std::size_t rank = pivoted_cholesky(it.F, eta_, lrows_);
    last_rank_ = rank;
    for (std::size_t r = 0; r < rank; ++r) {
      // w_r = V^T l_r
      core::FixedVector<T> l{};
      for (std::size_t m = 0; m < T; ++m) {
        l[m] = lrows_(r, m);
      }
      core::matvec_transposed(stats_.V, l, w_);
      core::matvec_transposed(U_, w_, a_);  // a = U^T w (U^T w = sum_r w_r U_r)
      potter_apply();
      // Theta~ <- Theta~ - (Theta~ w) k^T
      core::matvec(theta_, w_, tw_);
      core::rank_one_sub(theta_, 1.0, tw_, k_);
    }
  }

  void event(std::size_t i, const core::FixedVector<P>& z) noexcept {
    stats_.add_event(i, z);
    core::matvec_transposed(U_, z, a_);  // U^T z
    core::matvec(U_, a_, ua_);           // U U^T z
    core::axpy_lanes<core::FixedMatrix<D, P>::stride>(theta_.data.data() + i * theta_.stride, 1.0,
                                                      ua_.data.data());
  }

  void replenish(std::size_t p) noexcept {
    const double amount = stats_.replenish(p);
    if (!(amount > 0.0)) {
      return;
    }
    const double s = std::sqrt(amount);
    // a = U^T (s e_p) = s * (row p of U)
    for (std::size_t c = 0; c < core::FixedVector<P>::padded; ++c) {
      a_.data[c] = s * U_.data[p * U_.stride + c];
    }
    potter_apply();
    // Theta~ <- Theta~ + s (Theta0_{:,p} - Theta~_{:,p}) k^T
    for (std::size_t i = 0; i < D; ++i) {
      tw_[i] = s * (stats_.theta0(i, p) - theta_(i, p));
    }
    core::rank_one_sub(theta_, -1.0, tw_, k_);
  }

  void full_restore() noexcept {
    for (std::size_t p = 0; p < P; ++p) {
      stats_.replenish(p);
    }
    refactor();
  }

  // Per-step drift control: recompute one row of Theta~ exactly as C_r U U^T.
  void finish() noexcept {
    const std::size_t r = refresh_row_;
    refresh_row_ = (refresh_row_ + 1) % D;
    refresh_row(r);
  }

  const core::FixedMatrix<D, P>& theta() const noexcept { return theta_; }
  const LsStatistics<Shape>& stats() const noexcept { return stats_; }
  const core::FixedMatrix<P, P>& U() const noexcept { return U_; }
  std::uint64_t factorization_failures() const noexcept { return failures_; }
  std::size_t last_rank() const noexcept { return last_rank_; }

 private:
  // Potter update for the rank-one information w w^T with a = U^T w already in a_. Leaves the gain
  // k = (M + w w^T)^{-1} w in k_.
  void potter_apply() noexcept {
    const double s2 = core::dot(a_, a_);
    core::matvec(U_, a_, ua_);
    const double root = std::sqrt(1.0 + s2);
    const double c = 1.0 / (root * (root + 1.0));
    core::rank_one_sub(U_, c, ua_, a_);
    const double inv = 1.0 / (1.0 + s2);
    for (std::size_t i = 0; i < core::FixedVector<P>::padded; ++i) {
      k_.data[i] = ua_.data[i] * inv;
    }
  }

  void refresh_row(std::size_t r) noexcept {
    core::FixedVector<P> c{};
    for (std::size_t p = 0; p < P; ++p) {
      c[p] = stats_.C(r, p);
    }
    core::matvec_transposed(U_, c, a_);
    core::matvec(U_, a_, ua_);
    for (std::size_t p = 0; p < core::FixedVector<P>::padded; ++p) {
      theta_.data[r * theta_.stride + p] = ua_.data[p];
    }
  }

  void refactor() noexcept {
    chol_ = stats_.M;
    if (!core::cholesky_lower(chol_, col_)) {
      ++failures_;
      return;
    }
    core::inverse_transpose_factor(chol_, U_);
    for (std::size_t r = 0; r < D; ++r) {
      refresh_row(r);
    }
  }

  LsStatistics<Shape> stats_{};
  core::FixedMatrix<P, P> U_{};
  core::FixedMatrix<D, P> theta_{};
  core::FixedMatrix<T, T> lrows_{};
  core::FixedMatrix<P, P> chol_{};
  core::FixedVector<P> col_{};
  core::FixedVector<P> w_{};
  core::FixedVector<P> a_{};
  core::FixedVector<P> ua_{};
  core::FixedVector<P> k_{};
  core::FixedVector<D> tw_{};
  double eta_ = 1e-15;
  std::size_t refresh_row_ = 0;
  std::size_t last_rank_ = 0;
  std::uint64_t failures_ = 0;
};

// Back-end 2 (information form): keeps only the exact statistics and solves
// Theta~ = C M^{-1} by a fresh Cholesky factorization of M once per step. Cost per step is
// O(P^3 / 6 + D P^2 + T P r_max) instead of O(T P^2) for Potter; for D <= ~10 with the default bank
// this is fewer operations (docs/benchmarks.md has the measured comparison). No incremental state
// is carried in Theta~, so it equals the batch estimate up to the rounding of one solve.
template <class Shape>
class CholeskyEstimator {
 public:
  using L = BankLayout<Shape>;
  static constexpr std::size_t D = L::D;
  static constexpr std::size_t P = L::P;
  static constexpr const char* kName = "cholesky";

  void init(const ModelConfig<Shape>& cfg) noexcept {
    stats_.init(cfg);
    theta_ = cfg.theta0;
  }

  void interval(const KernelBank<Shape>& bank, const KernelState<Shape>& ks, const IntervalTerms<Shape>& it,
                bool /*long_gap*/) noexcept {
    build_v(bank, ks, stats_.V);
    stats_.forget(it.d);
    stats_.add_interval(it);
  }

  void event(std::size_t i, const core::FixedVector<P>& z) noexcept { stats_.add_event(i, z); }

  void replenish(std::size_t p) noexcept { stats_.replenish(p); }

  void full_restore() noexcept {
    for (std::size_t p = 0; p < P; ++p) {
      stats_.replenish(p);
    }
  }

  void finish() noexcept {
    // Lower triangle of M and the transposed right-hand sides C^T.
    for (std::size_t a = 0; a < P; ++a) {
      const double* src = stats_.M.data.data() + a * stats_.M.stride;
      double* dst = chol_.data.data() + a * chol_.stride;
      for (std::size_t b = 0; b <= a; ++b) {
        dst[b] = src[b];
      }
      for (std::size_t i = 0; i < D; ++i) {
        rhs_[a][i] = stats_.C(i, a);
      }
    }
    if (!core::cholesky_factor_solve(chol_, rhs_, col_, inv_diag_)) {
      ++failures_;  // keep the previous estimate
      return;
    }
    for (std::size_t i = 0; i < D; ++i) {
      for (std::size_t p = 0; p < P; ++p) {
        theta_(i, p) = rhs_[p][i];
      }
    }
  }

  const core::FixedMatrix<D, P>& theta() const noexcept { return theta_; }
  const LsStatistics<Shape>& stats() const noexcept { return stats_; }
  std::uint64_t factorization_failures() const noexcept { return failures_; }

 private:
  LsStatistics<Shape> stats_{};
  core::FixedMatrix<D, P> theta_{};
  core::FixedMatrix<P, P> chol_{};
  alignas(core::kCacheLineBytes) std::array<std::array<double, D>, P> rhs_{};
  core::FixedVector<P> col_{};
  core::FixedVector<P> inv_diag_{};
  std::uint64_t failures_ = 0;
};

}  // namespace pulselob::hawkes
