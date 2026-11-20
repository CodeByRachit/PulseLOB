#pragma once

#include <array>
#include <cmath>
#include <cstddef>

#include "pulselob/core/fixed_matrix.hpp"
#include "pulselob/core/linalg.hpp"
#include "pulselob/core/psi.hpp"

namespace pulselob::hawkes {

// Compile-time bookkeeping of the kernel bank (math_spec Sections 3.1, 4.1 and 5.4).
//  * R row base[q] + n holds R^{q,(n)}_j (one column per source channel j), n = 0..r_q-1.
//  * Time-basis function tb(q, n) = 1 + base[q] + n is u^n e^{-beta_q u}; index 0 is f_0 = 1.
template <class Shape>
struct BankLayout {
  static constexpr std::size_t D = Shape::D;
  static constexpr std::size_t Q = Shape::Q;
  static constexpr std::size_t P = Shape::P;
  static constexpr std::size_t S = Shape::kStatesPerChannel;  // R rows
  static constexpr std::size_t T = Shape::kTimeBasis;         // M + 1
  static constexpr std::size_t kMaxPsi = 2 * (Shape::r_max - 1);

  static constexpr std::array<std::size_t, Q> base = [] {
    std::array<std::size_t, Q> b{};
    std::size_t s = 0;
    for (std::size_t q = 0; q < Q; ++q) {
      b[q] = s;
      s += Shape::orders[q];
    }
    return b;
  }();
  static constexpr std::size_t order(std::size_t q) noexcept { return Shape::orders[q]; }
  static constexpr std::size_t row(std::size_t q, std::size_t n) noexcept { return base[q] + n; }
  static constexpr std::size_t tb(std::size_t q, std::size_t n) noexcept { return 1 + base[q] + n; }
};

// Runtime constants of the bank.
template <class Shape>
struct KernelBank {
  using L = BankLayout<Shape>;
  std::array<double, L::Q> beta{};
  std::array<double, L::Q> beta_pow_r{};  // beta_q^{r_q}
  double x_cut = 50.0;

  KernelBank() noexcept = default;
  KernelBank(const std::array<double, L::Q>& b, double cut) noexcept : beta(b), x_cut(cut) {
    for (std::size_t q = 0; q < L::Q; ++q) {
      beta_pow_r[q] = L::order(q) == 1 ? beta[q] : beta[q] * beta[q];
    }
  }
};

// Everything that depends only on the length of an event-free interval (and on kappa): decay
// factors, compensator coefficients (Theorem 3) and the time-basis Gram matrix F (Theorem 7).
// Exactly Q + 1 exponentials are evaluated, plus one more per basis only when d underflows
// (intervals of more than ~575 / kappa seconds).
template <class Shape>
struct IntervalTerms {
  using L = BankLayout<Shape>;
  double dt = 0.0;                                 // seconds
  double kappa = 0.0;                              // forgetting rate on this interval
  double d = 1.0;                                  // e^{-kappa dt}
  std::array<double, L::Q> e{};                    // e^{-beta_q dt} (raw, before the cutoff)
  std::array<bool, L::Q> cut{};                    // beta_q dt > x_cut: states of basis q are zeroed
  std::array<std::array<double, 2>, L::Q> comp{};  // X^q_j = sum_l comp[q][l] R^{q,(l)}_j(t+)
  core::FixedMatrix<L::T, L::T> F{};               // F_ab = int_0^dt e^{-kappa (dt-u)} f_a(u) f_b(u) du
};

template <class Shape>
inline void compute_interval(const KernelBank<Shape>& bank, double dt, double kappa,
                             IntervalTerms<Shape>& it) noexcept {
  using L = BankLayout<Shape>;
  constexpr std::size_t Q = L::Q;
  constexpr std::size_t NM = L::kMaxPsi < 1 ? 1 : L::kMaxPsi;  // psi_1 is needed by Erlang-2 only
  it.dt = dt;
  it.kappa = kappa;
  const double d = std::exp(-kappa * dt);
  it.d = d;
  const double dt2 = dt * dt;
  const double dtp[4] = {dt, dt2, dt2 * dt, dt2 * dt2};  // dt^{n+1}, n = 0..3

  // g_q = e^{-(beta_q - kappa) dt}; ratio form unless d is too small to divide by.
  std::array<double, Q> g{};
  for (std::size_t q = 0; q < Q; ++q) {
    const double bdt = bank.beta[q] * dt;
    it.e[q] = std::exp(-bdt);
    it.cut[q] = bdt > bank.x_cut;
    g[q] = d > 1e-250 ? it.e[q] / d : std::exp(-(bank.beta[q] - kappa) * dt);
  }

  std::array<double, NM + 1> ps{};
  // Compensator coefficients: Erlang-1: beta dt psi_0(beta dt);
  // Erlang-2: l = 0 (R^(0)): beta^2 dt^2 psi_1, l = 1 (R^(1)): beta^2 dt psi_0.
  for (std::size_t q = 0; q < Q; ++q) {
    core::psi_upto<NM>(bank.beta[q] * dt, it.e[q], ps);
    if (L::order(q) == 1) {
      it.comp[q][0] = bank.beta[q] * dt * ps[0];
      it.comp[q][1] = 0.0;
    } else {
      it.comp[q][0] = bank.beta_pow_r[q] * dt2 * ps[1];
      it.comp[q][1] = bank.beta_pow_r[q] * dt * ps[0];
    }
  }

  // F_00 = int_0^dt e^{-kappa v} dv = dt psi_0(kappa dt). This equals the spec's
  // d dt psi_0(-kappa dt) but never forms e^{+kappa dt}, which overflows for idle gaps.
  core::psi_upto<NM>(kappa * dt, d, ps);
  it.F(0, 0) = dt * ps[0];
  // F_{0,(q,n)} = d dt^{n+1} psi_n((beta_q - kappa) dt).
  for (std::size_t q = 0; q < Q; ++q) {
    core::psi_upto<NM>((bank.beta[q] - kappa) * dt, g[q], ps);
    for (std::size_t n = 0; n < L::order(q); ++n) {
      const double v = d * dtp[n] * ps[n];
      it.F(0, L::tb(q, n)) = v;
      it.F(L::tb(q, n), 0) = v;
    }
  }
  // F_{(q,n),(q',n')} = d dt^{n+n'+1} psi_{n+n'}((beta_q + beta_q' - kappa) dt).
  for (std::size_t q = 0; q < Q; ++q) {
    for (std::size_t qq = q; qq < Q; ++qq) {
      core::psi_upto<NM>((bank.beta[q] + bank.beta[qq] - kappa) * dt, it.e[q] * g[qq], ps);
      for (std::size_t n = 0; n < L::order(q); ++n) {
        for (std::size_t nn = 0; nn < L::order(qq); ++nn) {
          const double v = d * dtp[n + nn] * ps[n + nn];
          it.F(L::tb(q, n), L::tb(qq, nn)) = v;
          it.F(L::tb(qq, nn), L::tb(q, n)) = v;
        }
      }
    }
  }
}

// The Erlang states R^{q,(n)}_j and the pending tie buffer pi (math_spec Sections 4.1 and 4.4).
template <class Shape>
struct KernelState {
  using L = BankLayout<Shape>;
  core::FixedMatrix<L::S, L::D> R{};
  core::FixedVector<L::D> pending{};

  void add_pending(std::size_t j, double g) noexcept { pending[j] += g; }

  // Moves the tie batch into R^{q,(0)}: the state at t_prev+.
  void flush() noexcept {
    constexpr std::size_t lanes = core::FixedMatrix<L::S, L::D>::stride;
    for (std::size_t q = 0; q < L::Q; ++q) {
      core::axpy_lanes<lanes>(R.data.data() + L::row(q, 0) * R.stride, 1.0, pending.data.data());
    }
    for (auto& x : pending.data) {
      x = 0.0;
    }
  }

  // Theorem 2(a) over the interval, with the cutoff of Section 4.6. Returns the number of bases cut.
  std::size_t decay(const IntervalTerms<Shape>& it) noexcept {
    constexpr std::size_t lanes = core::FixedMatrix<L::S, L::D>::stride;
    std::size_t cuts = 0;
    for (std::size_t q = 0; q < L::Q; ++q) {
      double* r0 = R.data.data() + L::row(q, 0) * R.stride;
      const double e = it.cut[q] ? 0.0 : it.e[q];
      cuts += it.cut[q] ? 1U : 0U;
      if (L::order(q) == 2) {
        double* r1 = R.data.data() + L::row(q, 1) * R.stride;
        for (std::size_t j = 0; j < lanes; ++j) {
          r1[j] = e * (r1[j] + it.dt * r0[j]);  // uses the old R^(0)
        }
      }
      core::scale_lanes<lanes>(r0, e);
    }
    return cuts;
  }

  // Regressor z = (1, x^1_1, ..., x^Q_D) from a state matrix: x^q_j = beta_q^{r_q} R^{q,(r_q-1)}_j.
  static void regressor_from(const KernelBank<Shape>& bank, const core::FixedMatrix<L::S, L::D>& Rm,
                             core::FixedVector<L::P>& z) noexcept {
    z[0] = 1.0;
    for (std::size_t j = 0; j < L::D; ++j) {
      for (std::size_t q = 0; q < L::Q; ++q) {
        z[Shape::coord(j, q)] = bank.beta_pow_r[q] * Rm(L::row(q, L::order(q) - 1), j);
      }
    }
  }

  // z(t-): excludes the pending batch (strict left limit, Proposition 4).
  void regressor(const KernelBank<Shape>& bank, core::FixedVector<L::P>& z) const noexcept {
    regressor_from(bank, R, z);
  }

  // Post-event state R + pi (read-only, used for forecasts and decisions at t+).
  void post_event(core::FixedMatrix<L::S, L::D>& out) const noexcept {
    out = R;
    constexpr std::size_t lanes = core::FixedMatrix<L::S, L::D>::stride;
    for (std::size_t q = 0; q < L::Q; ++q) {
      core::axpy_lanes<lanes>(out.data.data() + L::row(q, 0) * out.stride, 1.0, pending.data.data());
    }
  }

  // Post-event state decayed by dt seconds without changing this object: the state at a later time
  // with no event in between (used for read-only signals at book updates).
  void post_event_at(const KernelBank<Shape>& bank, double dt,
                     core::FixedMatrix<L::S, L::D>& out) const noexcept {
    post_event(out);
    if (!(dt > 0.0)) {
      return;
    }
    constexpr std::size_t lanes = core::FixedMatrix<L::S, L::D>::stride;
    for (std::size_t q = 0; q < L::Q; ++q) {
      const double bdt = bank.beta[q] * dt;
      const double e = bdt > bank.x_cut ? 0.0 : std::exp(-bdt);
      double* r0 = out.data.data() + L::row(q, 0) * out.stride;
      if (L::order(q) == 2) {
        double* r1 = out.data.data() + L::row(q, 1) * out.stride;
        for (std::size_t j = 0; j < lanes; ++j) {
          r1[j] = e * (r1[j] + dt * r0[j]);
        }
      }
      core::scale_lanes<lanes>(r0, e);
    }
  }

  // Compensator increments Z(t, dt) = (dt, X^1_1, ..., X^Q_D) from the state at t_prev+ (Theorem 3).
  void compensator(const IntervalTerms<Shape>& it, core::FixedVector<L::P>& Z) const noexcept {
    Z[0] = it.dt;
    for (std::size_t j = 0; j < L::D; ++j) {
      for (std::size_t q = 0; q < L::Q; ++q) {
        double x = it.comp[q][0] * R(L::row(q, 0), j);
        if (L::order(q) == 2) {
          x += it.comp[q][1] * R(L::row(q, 1), j);
        }
        Z[Shape::coord(j, q)] = x;
      }
    }
  }
};

// Sparse coefficient matrix V of Theorem 7, stored densely (T x P; entries outside the fixed
// sparsity pattern stay zero): z(t+u) = V^T f(u) on an event-free interval.
// Coordinate (j,q) has coefficient beta_q^{r_q} R^{q,(r_q-1-n)}_j(t+) / n! on u^n e^{-beta_q u}.
template <class Shape>
inline void build_v(const KernelBank<Shape>& bank, const KernelState<Shape>& ks,
                    core::FixedMatrix<Shape::kTimeBasis, Shape::P>& V) noexcept {
  using L = BankLayout<Shape>;
  V(0, 0) = 1.0;
  for (std::size_t j = 0; j < L::D; ++j) {
    for (std::size_t q = 0; q < L::Q; ++q) {
      const std::size_t c = Shape::coord(j, q);
      if (L::order(q) == 1) {
        V(L::tb(q, 0), c) = bank.beta_pow_r[q] * ks.R(L::row(q, 0), j);
      } else {
        V(L::tb(q, 0), c) = bank.beta_pow_r[q] * ks.R(L::row(q, 1), j);
        V(L::tb(q, 1), c) = bank.beta_pow_r[q] * ks.R(L::row(q, 0), j);
      }
    }
  }
}

}  // namespace pulselob::hawkes
