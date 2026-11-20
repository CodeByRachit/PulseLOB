#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "pulselob/core/fixed_matrix.hpp"
#include "pulselob/core/psi.hpp"
#include "pulselob/hawkes/kernel.hpp"
#include "pulselob/hawkes/stability.hpp"

namespace pulselob::hawkes {

// Configuration of the forecasts and the toxicity score (math_spec Section 7).
template <std::size_t D>
struct ForecastConfig {
  double horizon_s = 0.05;         // H
  std::size_t neumann_terms = 3;   // K
  std::array<int, D> direction{};  // s_i in {-1, 0, +1}
  std::array<double, D> impact{};  // eta_i >= 0
};

template <std::size_t D>
const char* validate(const ForecastConfig<D>& c) noexcept {
  if (!(std::isfinite(c.horizon_s) && c.horizon_s > 0.0)) {
    return "forecast horizon must be positive";
  }
  if (c.neumann_terms > 64) {
    return "neumann_terms must be at most 64";
  }
  for (std::size_t i = 0; i < D; ++i) {
    if (c.direction[i] < -1 || c.direction[i] > 1) {
      return "channel direction must be -1, 0 or +1";
    }
    if (!(std::isfinite(c.impact[i]) && c.impact[i] >= 0.0)) {
      return "impact weights must be finite and non-negative";
    }
  }
  return nullptr;
}

template <std::size_t D>
struct Forecast {
  core::FixedVector<D> lambda{};  // served intensities at t+
  core::FixedVector<D> m0{};      // m^(0)(t; H)
  core::FixedVector<D> m0_bar{};  // stationary m-bar^(0)(H)
  core::FixedVector<D> y_inf{};   // y^infinity
  core::FixedVector<D> x_inf{};   // X^infinity (K-term Neumann series)
  double flow_buy = 0.0;          // M^+
  double flow_sell = 0.0;         // M^-
  double excess_buy = 0.0;        // E^+
  double excess_sell = 0.0;       // E^-
  double tau_buy = 0.0;           // tau^+
  double tau_sell = 0.0;          // tau^-
};

// Toxicity score of Section 7.3 from the impact-weighted flows, with 0/0 := 0.
inline void toxicity(double flow_p, double flow_m, double bar_p, double bar_m, double& e_p, double& e_m,
                     double& tau_p, double& tau_m) noexcept {
  e_p = std::fmax(flow_p - bar_p, 0.0);
  e_m = std::fmax(flow_m - bar_m, 0.0);
  const double tot = e_p + e_m;
  const double frac_p = flow_p > 0.0 ? e_p / flow_p : 0.0;
  const double frac_m = flow_m > 0.0 ? e_m / flow_m : 0.0;
  const double pur_p = tot > 0.0 ? std::fmax(e_p - e_m, 0.0) / tot : 0.0;
  const double pur_m = tot > 0.0 ? std::fmax(e_m - e_p, 0.0) / tot : 0.0;
  tau_p = frac_p * pur_p;
  tau_m = frac_m * pur_m;
}

// Exact horizon forecasts from the post-event state (Sections 7.1 to 7.4). Every horizon-dependent
// constant (psi_n(beta_q H), and so on) is precomputed, so a forecast costs O(DP) and evaluates no
// exponential.
template <class Shape>
class Forecaster {
 public:
  using L = BankLayout<Shape>;
  static constexpr std::size_t D = L::D;
  static constexpr std::size_t Q = L::Q;
  static constexpr std::size_t P = L::P;

  void init(const KernelBank<Shape>& bank, const ForecastConfig<D>& cfg) noexcept {
    cfg_ = cfg;
    const double H = cfg.horizon_s;
    std::array<double, 2> ps{};
    for (std::size_t q = 0; q < Q; ++q) {
      const double b = bank.beta[q];
      core::psi_upto<1>(b * H, std::exp(-b * H), ps);
      if (L::order(q) == 1) {
        // X = beta H psi_0(beta H) R^(0);  k-bar = H psi_0(beta H);  mass = R^(0)
        xcoef_[q] = {b * H * ps[0], 0.0};
        kbar_[q] = H * ps[0];
        mass_[q] = {1.0, 0.0};
      } else {
        // X = beta^2 (H^2 psi_1 R^(0) + H psi_0 R^(1));  k-bar = H psi_0 + beta H^2 psi_1;
        // mass = R^(0) + beta R^(1)
        xcoef_[q] = {b * b * H * H * ps[1], b * b * H * ps[0]};
        kbar_[q] = H * ps[0] + b * H * H * ps[1];
        mass_[q] = {1.0, b};
      }
      mean_[q] = static_cast<double>(L::order(q)) / b;
    }
  }

  // Rpost: post-event state (R + pi), possibly decayed to the evaluation time.
  void compute(const KernelBank<Shape>& bank, const core::FixedMatrix<L::S, D>& Rpost, const Served<Shape>& s,
               Forecast<D>& f) const noexcept {
    const double H = cfg_.horizon_s;
    // Per-source quantities.
    std::array<std::array<double, Q>, D> X{};
    std::array<std::array<double, Q>, D> rem{};
    core::FixedVector<P> z{};
    z[0] = 1.0;
    for (std::size_t j = 0; j < D; ++j) {
      for (std::size_t q = 0; q < Q; ++q) {
        const double r0 = Rpost(L::row(q, 0), j);
        const double r1 = L::order(q) == 2 ? Rpost(L::row(q, 1), j) : 0.0;
        X[j][q] = xcoef_[q][0] * r0 + xcoef_[q][1] * r1;
        rem[j][q] = (mass_[q][0] * r0 + mass_[q][1] * r1) - s.lambda_inf[j] * mean_[q];
        z[Shape::coord(j, q)] = bank.beta_pow_r[q] * Rpost(L::row(q, L::order(q) - 1), j);
      }
    }
    for (std::size_t i = 0; i < D; ++i) {
      const double mu = s.theta(i, 0);
      double lam = mu;
      double m0 = mu * H;
      double mb = mu * H;
      double y = 0.0;
      for (std::size_t j = 0; j < D; ++j) {
        for (std::size_t q = 0; q < Q; ++q) {
          const double a = s.theta(i, Shape::coord(j, q));
          lam += a * z[Shape::coord(j, q)];
          m0 += a * X[j][q];
          mb += a * s.lambda_inf[j] * kbar_[q];
          y += a * rem[j][q];
        }
      }
      f.lambda[i] = lam;
      f.m0[i] = m0;
      f.m0_bar[i] = mb;
      f.y_inf[i] = y;
    }
    // X^inf = sum_{n=0}^{K} G^n y (Horner form x <- y + G x).
    core::FixedVector<D> x = f.y_inf;
    for (std::size_t n = 0; n < cfg_.neumann_terms; ++n) {
      core::FixedVector<D> nx{};
      for (std::size_t i = 0; i < D; ++i) {
        double acc = f.y_inf[i];
        for (std::size_t j = 0; j < D; ++j) {
          acc += s.G(i, j) * x[j];
        }
        nx[i] = acc;
      }
      x = nx;
    }
    f.x_inf = x;
    // Toxicity.
    double mp = 0.0;
    double mm = 0.0;
    double bp = 0.0;
    double bm = 0.0;
    for (std::size_t i = 0; i < D; ++i) {
      if (cfg_.direction[i] > 0) {
        mp += cfg_.impact[i] * f.m0[i];
        bp += cfg_.impact[i] * f.m0_bar[i];
      } else if (cfg_.direction[i] < 0) {
        mm += cfg_.impact[i] * f.m0[i];
        bm += cfg_.impact[i] * f.m0_bar[i];
      }
    }
    f.flow_buy = mp;
    f.flow_sell = mm;
    toxicity(mp, mm, bp, bm, f.excess_buy, f.excess_sell, f.tau_buy, f.tau_sell);
  }

  const ForecastConfig<D>& config() const noexcept { return cfg_; }

 private:
  ForecastConfig<D> cfg_{};
  std::array<std::array<double, 2>, Q> xcoef_{};
  std::array<std::array<double, 2>, Q> mass_{};
  std::array<double, Q> kbar_{};
  std::array<double, Q> mean_{};
};

}  // namespace pulselob::hawkes
