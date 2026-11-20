#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "pulselob/core/fixed_matrix.hpp"
#include "pulselob/hawkes/config.hpp"

namespace pulselob::hawkes {

// Served parameters and stability diagnostics (math_spec Section 6). Trivially copyable: in split
// mode this is the payload published through the seqlock.
template <class Shape>
struct Served {
  static constexpr std::size_t D = Shape::D;
  static constexpr std::size_t P = Shape::P;
  core::FixedMatrix<D, P> theta{};    // Theta^: clipped baselines, projected excitation rows
  core::FixedMatrix<D, D> G{};        // branching matrix of Theta^
  core::FixedVector<D> lambda_inf{};  // (I - G^)^{-1} mu^, tracked by one Jacobi sweep per step
  core::FixedVector<D> v{};           // Collatz-Wielandt weight vector
  double criticality = 0.0;           // rho~ = max_i (G~ v)_i / v_i, G~ from Theta~ clamped at 0
  double n_eff = 0.0;                 // discounted number of events (warm-up indicator)
  std::uint64_t steps = 0;            // estimator steps included (split-mode freshness)
  std::uint32_t projected_rows = 0;   // bit i set: row i was projected at the last step
  std::uint32_t flags = 0;            // kFlagFastForgetting
  static constexpr std::uint32_t kFlagFastForgetting = 1U;
};

// Exact Euclidean projection of a row onto {a >= 0, w^T a <= c} (Proposition 14). The root of the
// piecewise-linear function phi(nu) = w^T (a~ - nu w)_+ - c is located exactly by evaluating phi at
// every breakpoint nu_m = a~_m / w_m (O(n^2) with a fixed bound n = DQ, no iteration count to tune),
// then finalized in closed form on the identified active set. Returns nu* (0 if inactive).
template <std::size_t N>
inline double project_row(const std::array<double, N>& at, const std::array<double, N>& w, double c,
                          std::array<double, N>& out) noexcept {
  double s0 = 0.0;
  for (std::size_t m = 0; m < N; ++m) {
    s0 += w[m] * std::fmax(at[m], 0.0);
  }
  if (s0 <= c) {
    for (std::size_t m = 0; m < N; ++m) {
      out[m] = std::fmax(at[m], 0.0);
    }
    return 0.0;
  }
  std::array<double, N> bp{};
  for (std::size_t m = 0; m < N; ++m) {
    bp[m] = at[m] > 0.0 ? at[m] / w[m] : 0.0;
  }
  // nu_L = the largest breakpoint (or 0) at which phi > 0; phi(0) = s0 - c > 0.
  double nu_lo = 0.0;
  for (std::size_t m = 0; m < N; ++m) {
    const double nu = bp[m];
    if (!(nu > nu_lo)) {
      continue;
    }
    double phi = -c;
    for (std::size_t k = 0; k < N; ++k) {
      phi += w[k] * std::fmax(at[k] - nu * w[k], 0.0);
    }
    if (phi > 0.0) {
      nu_lo = nu;
    }
  }
  double sa = 0.0;
  double sw = 0.0;
  for (std::size_t m = 0; m < N; ++m) {
    if (bp[m] > nu_lo) {
      sa += w[m] * at[m];
      sw += w[m] * w[m];
    }
  }
  const double nu_star = (sa - c) / sw;
  double s = 0.0;
  for (std::size_t m = 0; m < N; ++m) {
    out[m] = std::fmax(at[m] - nu_star * w[m], 0.0);
    s += w[m] * out[m];
  }
  if (s > c) {  // rounding-level feasibility rescale
    const double f = c / s;
    for (std::size_t m = 0; m < N; ++m) {
      out[m] *= f;
    }
  }
  return nu_star;
}

// The serving step (spec Section 9, step 8): clip baselines, one Perron step for v, exact row
// projection onto the Collatz-Wielandt polytope, criticality index, one Jacobi sweep for
// lambda_inf. Cost O(D^2 Q) for the Perron step and sweep plus O((DQ)^2) per row for the projection.
template <class Shape>
class StabilityGuard {
 public:
  static constexpr std::size_t D = Shape::D;
  static constexpr std::size_t Q = Shape::Q;
  static constexpr std::size_t P = Shape::P;
  static constexpr std::size_t N = D * Q;

  // Initializes v_ref, then serves Theta_0 and solves lambda_inf exactly (cold path).
  void init(const ModelConfig<Shape>& cfg, Served<Shape>& out) noexcept {
    rho_max_ = cfg.rho_max;
    eps_v_ = cfg.eps_v;
    mu_min_ = cfg.mu_min;
    mu_max_ = cfg.mu_max;
    lo_ = 1.0 / std::sqrt(cfg.chi);
    hi_ = std::sqrt(cfg.chi);
    for (std::size_t i = 0; i < D; ++i) {
      v_ref_[i] = 1.0;
    }
    if (cfg.v_ref_from_prior) {
      core::FixedMatrix<D, D> G0{};
      branching(cfg.theta0, G0, false);
      // Power iteration with the same eps_v regularization as the per-event step (1000 sweeps).
      core::FixedVector<D> v = v_ref_;
      for (int it = 0; it < 1000; ++it) {
        core::FixedVector<D> y{};
        double mx = 0.0;
        for (std::size_t i = 0; i < D; ++i) {
          double s = eps_v_;
          for (std::size_t j = 0; j < D; ++j) {
            s += G0(i, j) * v[j];
          }
          y[i] = s;
          mx = std::fmax(mx, s);
        }
        for (std::size_t i = 0; i < D; ++i) {
          v[i] = y[i] / mx;
        }
      }
      v_ref_ = v;
    }
    out = {};
    out.v = v_ref_;
    serve(cfg.theta0, out, false);
    solve_lambda_inf(out);
  }

  // One serving step from the unconstrained estimate. Uses (and updates) out.G and out.v.
  void serve(const core::FixedMatrix<D, P>& theta_tilde, Served<Shape>& out,
             bool perron_step = true) noexcept {
    // Baselines.
    for (std::size_t i = 0; i < D; ++i) {
      out.theta(i, 0) = std::fmin(std::fmax(theta_tilde(i, 0), mu_min_), mu_max_);
    }
    // Perron step with the previous served G.
    if (perron_step) {
      core::FixedVector<D> y{};
      double mx = 0.0;
      for (std::size_t i = 0; i < D; ++i) {
        double s = eps_v_ * v_ref_[i];
        for (std::size_t j = 0; j < D; ++j) {
          s += out.G(i, j) * out.v[j];
        }
        y[i] = s;
        mx = std::fmax(mx, s);
      }
      for (std::size_t i = 0; i < D; ++i) {
        out.v[i] = std::fmin(std::fmax(y[i] / mx, lo_ * v_ref_[i]), hi_ * v_ref_[i]);
      }
    }
    // Row projections.
    std::array<double, N> w{};
    for (std::size_t j = 0; j < D; ++j) {
      for (std::size_t q = 0; q < Q; ++q) {
        w[j * Q + q] = out.v[j];
      }
    }
    std::uint32_t projected = 0;
    std::array<double, N> at{};
    std::array<double, N> a{};
    for (std::size_t i = 0; i < D; ++i) {
      for (std::size_t m = 0; m < N; ++m) {
        at[m] = theta_tilde(i, 1 + m);
      }
      const double nu = project_row(at, w, rho_max_ * out.v[i], a);
      if (nu > 0.0) {
        projected |= 1U << i;
      }
      for (std::size_t m = 0; m < N; ++m) {
        out.theta(i, 1 + m) = a[m];
      }
    }
    out.projected_rows = projected;
    branching(out.theta, out.G, false);
    // Criticality index from the unconstrained estimate, clamped at zero.
    core::FixedMatrix<D, D> Gt{};
    branching(theta_tilde, Gt, true);
    double crit = 0.0;
    for (std::size_t i = 0; i < D; ++i) {
      double s = 0.0;
      for (std::size_t j = 0; j < D; ++j) {
        s += Gt(i, j) * out.v[j];
      }
      crit = std::fmax(crit, s / out.v[i]);
    }
    out.criticality = crit;
    // One Jacobi sweep: lambda_inf <- mu^ + G^ lambda_inf.
    core::FixedVector<D> li{};
    for (std::size_t i = 0; i < D; ++i) {
      double s = out.theta(i, 0);
      for (std::size_t j = 0; j < D; ++j) {
        s += out.G(i, j) * out.lambda_inf[j];
      }
      li[i] = s;
    }
    out.lambda_inf = li;
  }

  const core::FixedVector<D>& v_ref() const noexcept { return v_ref_; }

  // G_ij = sum_q theta(i, coord(j, q)); optionally with negative weights clamped to zero.
  static void branching(const core::FixedMatrix<D, P>& th, core::FixedMatrix<D, D>& G, bool clamp) noexcept {
    for (std::size_t i = 0; i < D; ++i) {
      for (std::size_t j = 0; j < D; ++j) {
        double s = 0.0;
        for (std::size_t q = 0; q < Q; ++q) {
          const double a = th(i, Shape::coord(j, q));
          s += clamp ? std::fmax(a, 0.0) : a;
        }
        G(i, j) = s;
      }
    }
  }

  // Exact lambda_inf = (I - G^)^{-1} mu^ by Gaussian elimination with partial pivoting (cold path).
  static void solve_lambda_inf(Served<Shape>& out) noexcept {
    std::array<double, D*(D + 1)> A{};
    for (std::size_t i = 0; i < D; ++i) {
      for (std::size_t j = 0; j < D; ++j) {
        A[i * (D + 1) + j] = (i == j ? 1.0 : 0.0) - out.G(i, j);
      }
      A[i * (D + 1) + D] = out.theta(i, 0);
    }
    for (std::size_t k = 0; k < D; ++k) {
      std::size_t piv = k;
      for (std::size_t i = k + 1; i < D; ++i) {
        if (std::fabs(A[i * (D + 1) + k]) > std::fabs(A[piv * (D + 1) + k])) {
          piv = i;
        }
      }
      for (std::size_t j = 0; j <= D; ++j) {
        const double tmp = A[k * (D + 1) + j];
        A[k * (D + 1) + j] = A[piv * (D + 1) + j];
        A[piv * (D + 1) + j] = tmp;
      }
      for (std::size_t i = k + 1; i < D; ++i) {
        const double f = A[i * (D + 1) + k] / A[k * (D + 1) + k];
        for (std::size_t j = k; j <= D; ++j) {
          A[i * (D + 1) + j] -= f * A[k * (D + 1) + j];
        }
      }
    }
    for (std::size_t kk = D; kk-- > 0;) {
      double s = A[kk * (D + 1) + D];
      for (std::size_t j = kk + 1; j < D; ++j) {
        s -= A[kk * (D + 1) + j] * out.lambda_inf[j];
      }
      out.lambda_inf[kk] = s / A[kk * (D + 1) + kk];
    }
  }

 private:
  core::FixedVector<D> v_ref_{};
  double rho_max_ = 0.97;
  double eps_v_ = 1e-6;
  double mu_min_ = 1e-4;
  double mu_max_ = 1e6;
  double lo_ = 1.0;
  double hi_ = 1.0;
};

}  // namespace pulselob::hawkes
