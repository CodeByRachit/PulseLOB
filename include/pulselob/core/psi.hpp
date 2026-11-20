#pragma once

#include <array>
#include <cmath>
#include <cstddef>

namespace pulselob::core {

// psi_n(x) = int_0^1 s^n e^{-x s} ds (math_spec Section 4.2).
//
// Evaluation rule: for |x| < 1 the alternating series sum_m (-x)^m / (m! (n+m+1)), truncated once
// the next term is below 1e-18 (at most 22 terms, truncation error < 1/22! ~ 9e-22); otherwise the
// upward recurrence psi_0 = (1 - e^{-x}) / x, psi_n = (n psi_{n-1} - e^{-x}) / x, whose error
// amplification is n / x <= n for x >= 1. The engine only needs n <= 2, and only x > 0 (the one
// spec case with a negative argument is evaluated in the equivalent form Delta * psi_0(kappa Delta),
// see hawkes/interval.hpp).
inline constexpr double kPsiSeriesRadius = 1.0;
inline constexpr int kPsiSeriesMaxTerms = 22;
inline constexpr std::size_t kPsiMaxOrder = 3;

namespace detail {
// 1/k for k = 0..26 (entry 0 unused). Multiplying by a correctly rounded reciprocal instead of
// dividing changes the result by at most one ulp per term, and keeps the series loop free of
// divisions.
inline constexpr std::array<double, 27> kInv = {
    0.0,      1.0,      1.0 / 2,  1.0 / 3,  1.0 / 4,  1.0 / 5,  1.0 / 6,  1.0 / 7,  1.0 / 8,
    1.0 / 9,  1.0 / 10, 1.0 / 11, 1.0 / 12, 1.0 / 13, 1.0 / 14, 1.0 / 15, 1.0 / 16, 1.0 / 17,
    1.0 / 18, 1.0 / 19, 1.0 / 20, 1.0 / 21, 1.0 / 22, 1.0 / 23, 1.0 / 24, 1.0 / 25, 1.0 / 26};
}  // namespace detail

// Writes psi_0(x) .. psi_NMax(x) into out. emx must equal e^{-x}; it is used only when |x| >= 1,
// so callers can pass a product or ratio of exponentials they already hold.
template <std::size_t NMax>
inline void psi_upto(double x, double emx, std::array<double, NMax + 1>& out) noexcept {
  static_assert(NMax <= kPsiMaxOrder, "psi_n is validated for n <= 3 only");
  if (std::fabs(x) < kPsiSeriesRadius) {
    for (std::size_t n = 0; n <= NMax; ++n) {
      out[n] = 0.0;
    }
    double term = 1.0;  // (-x)^m / m!
    for (int m = 0; m < kPsiSeriesMaxTerms; ++m) {
      const auto um = static_cast<std::size_t>(m);
      for (std::size_t n = 0; n <= NMax; ++n) {
        out[n] += term * detail::kInv[n + um + 1];
      }
      term *= -x * detail::kInv[um + 1];
      if (std::fabs(term) < 1e-18) {
        break;
      }
    }
    return;
  }
  const double inv_x = 1.0 / x;
  double p = (1.0 - emx) * inv_x;
  out[0] = p;
  for (std::size_t n = 1; n <= NMax; ++n) {
    p = (static_cast<double>(n) * p - emx) * inv_x;
    out[n] = p;
  }
}

inline double psi(std::size_t n, double x, double emx) noexcept {
  std::array<double, kPsiMaxOrder + 1> v{};
  psi_upto<kPsiMaxOrder>(x, emx, v);
  return v[n];
}

}  // namespace pulselob::core
