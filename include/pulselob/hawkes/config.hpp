#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "pulselob/core/fixed_matrix.hpp"
#include "pulselob/core/time.hpp"

namespace pulselob::hawkes {

// Runtime configuration of the model (docs/architecture.md Section 4.1). Defaults are the starting
// values of math_spec Appendix B. Everything here is validated once, at construction, by
// validate(); the hot path assumes a valid configuration.
template <class Shape>
struct ModelConfig {
  static constexpr std::size_t D = Shape::D;
  static constexpr std::size_t Q = Shape::Q;
  static constexpr std::size_t P = Shape::P;

  // Kernel bank decay rates beta_q (1/s); orders come from Shape.
  std::array<double, Q> beta{};

  // Forgetting (Section 5.5): half-lives in seconds, kappa = ln 2 / half-life.
  double half_life_slow_s = 60.0;
  double half_life_fast_s = 5.0;
  double dwell_s = 10.0;

  // Time-rescaling CUSUM (Section 5.5).
  bool cusum_enabled = true;
  double cusum_rho = 1.5;
  double cusum_threshold = 14.0;

  // Ridge prior (Section 5.4): per-coordinate strength omega_p and prior mean Theta_0 (D x P).
  core::FixedVector<P> omega{};
  core::FixedMatrix<D, P> theta0{};
  // Heartbeat period T_hb: an interval longer than this (no event or heartbeat in between) ends
  // with a full ridge restore, which keeps Proposition 9's bound unconditional (spec 0.2.0, 5.4).
  core::DurationNs heartbeat_ns = 100'000'000;

  // Stability guardrail (Section 6).
  double rho_max = 0.97;
  double chi = 1.01;
  double eps_v = 1e-6;
  double rho_crit = 1.0;
  double mu_min = 1e-4;
  double mu_max = 1e6;
  // If true, v_ref is the Perron vector of Theta_0's branching matrix; otherwise the ones vector.
  bool v_ref_from_prior = true;

  // Numerics (Sections 4.6, 5.4, 7.1).
  double x_cut = 50.0;
  double pivot_eta = 1e-15;

  // Effective number of discounted events below which signals are flagged "warming up".
  double warmup_events = 200.0;

  double kappa_slow() const noexcept { return std::log(2.0) / half_life_slow_s; }
  double kappa_fast() const noexcept { return std::log(2.0) / half_life_fast_s; }
};

namespace detail {
inline bool finite_positive(double x) noexcept {
  return std::isfinite(x) && x > 0.0;
}
}  // namespace detail

// Returns nullptr if the configuration satisfies every assumption of the spec, otherwise a static
// string naming the first violated condition. Never throws (headers compile without exceptions).
template <class Shape>
const char* validate(const ModelConfig<Shape>& c) noexcept {
  constexpr std::size_t Q = Shape::Q;
  double beta_min = 0.0;
  for (std::size_t q = 0; q < Q; ++q) {
    if (!detail::finite_positive(c.beta[q])) {
      return "beta must be finite and positive";
    }
    beta_min = (q == 0) ? c.beta[q] : std::fmin(beta_min, c.beta[q]);
  }
  // (I1) distinct modes and means.
  for (std::size_t a = 0; a < Q; ++a) {
    for (std::size_t b = a + 1; b < Q; ++b) {
      const double ra = static_cast<double>(Shape::orders[a]);
      const double rb = static_cast<double>(Shape::orders[b]);
      const bool same_mean = ra / c.beta[a] == rb / c.beta[b];
      const bool same_mode = (ra - 1.0) / c.beta[a] == (rb - 1.0) / c.beta[b];
      if (same_mean && same_mode) {
        return "(I1) two kernels share both mode and mean";
      }
    }
  }
  if (!detail::finite_positive(c.half_life_slow_s) || !detail::finite_positive(c.half_life_fast_s)) {
    return "half-lives must be finite and positive";
  }
  if (c.half_life_fast_s > c.half_life_slow_s) {
    return "fast half-life must not exceed the slow half-life";
  }
  // (I2) kappa_max < beta_min.
  if (!(c.kappa_fast() < beta_min)) {
    return "(I2) kappa_fast must be below the smallest decay rate";
  }
  if (!(std::isfinite(c.dwell_s) && c.dwell_s >= 0.0)) {
    return "dwell must be finite and non-negative";
  }
  if (!(std::isfinite(c.cusum_rho) && c.cusum_rho > 1.0)) {
    return "CUSUM rho must exceed 1";
  }
  if (!detail::finite_positive(c.cusum_threshold)) {
    return "CUSUM threshold must be positive";
  }
  for (std::size_t p = 0; p < Shape::P; ++p) {
    if (!detail::finite_positive(c.omega[p])) {
      return "ridge omega must be finite and positive";
    }
  }
  if (c.heartbeat_ns <= 0) {
    return "heartbeat period must be positive";
  }
  if (!(detail::finite_positive(c.mu_min) && std::isfinite(c.mu_max) && c.mu_min < c.mu_max)) {
    return "need 0 < mu_min < mu_max";
  }
  if (!(detail::finite_positive(c.rho_max) && c.rho_max < 1.0)) {
    return "rho_max must lie in (0, 1)";
  }
  if (!(std::isfinite(c.chi) && c.chi >= 1.0)) {
    return "chi must be >= 1";
  }
  // (S4) with normalized marks (g_bar = 1).
  if (!(c.rho_max * c.chi < 1.0)) {
    return "(S4) rho_max * chi must be below 1";
  }
  if (!detail::finite_positive(c.eps_v)) {
    return "eps_v must be positive";
  }
  if (!detail::finite_positive(c.rho_crit)) {
    return "rho_crit must be positive";
  }
  if (!(std::isfinite(c.x_cut) && c.x_cut > 0.0 && c.x_cut < 700.0)) {
    return "x_cut must lie in (0, 700)";
  }
  if (!(std::isfinite(c.pivot_eta) && c.pivot_eta >= 0.0 && c.pivot_eta < 1.0)) {
    return "pivot_eta must lie in [0, 1)";
  }
  if (!(std::isfinite(c.warmup_events) && c.warmup_events >= 0.0)) {
    return "warmup_events must be finite and non-negative";
  }
  for (std::size_t i = 0; i < Shape::D; ++i) {
    const double mu0 = c.theta0(i, 0);
    if (!(std::isfinite(mu0) && mu0 >= c.mu_min && mu0 <= c.mu_max)) {
      return "prior baseline must lie in [mu_min, mu_max]";
    }
    for (std::size_t p = 1; p < Shape::P; ++p) {
      if (!(std::isfinite(c.theta0(i, p)) && c.theta0(i, p) >= 0.0)) {
        return "prior excitation weights must be finite and non-negative";
      }
    }
  }
  return nullptr;
}

}  // namespace pulselob::hawkes
