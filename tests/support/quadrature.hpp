#pragma once

// Definition-level references for the equivalence tests: composite Gauss-Legendre quadrature and
// direct O(N) sums over the event history. Deliberately independent of the closed forms under test.

#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

namespace pulselob::testkit {

class GaussLegendre30 {
 public:
  static constexpr std::size_t kN = 30;

  GaussLegendre30() {
    // Newton iteration on P_30 from the Chebyshev initial guesses.
    for (std::size_t i = 0; i < kN; ++i) {
      double x =
          std::cos(3.141592653589793 * (static_cast<double>(i) + 0.75) / (static_cast<double>(kN) + 0.5));
      double dp = 0.0;
      for (int it = 0; it < 100; ++it) {
        double p0 = 1.0;
        double p1 = x;
        for (std::size_t k = 2; k <= kN; ++k) {
          const double kk = static_cast<double>(k);
          const double p2 = ((2.0 * kk - 1.0) * x * p1 - (kk - 1.0) * p0) / kk;
          p0 = p1;
          p1 = p2;
        }
        dp = static_cast<double>(kN) * (x * p1 - p0) / (x * x - 1.0);
        const double dx = p1 / dp;
        x -= dx;
        if (std::fabs(dx) < 1e-17) {
          break;
        }
      }
      nodes_[i] = x;
      weights_[i] = 2.0 / ((1.0 - x * x) * dp * dp);
    }
  }

  // Integral of f over [a, b].
  template <class F>
  double integrate(F&& f, double a, double b) const {
    const double h = 0.5 * (b - a);
    const double m = 0.5 * (a + b);
    double s = 0.0;
    for (std::size_t i = 0; i < kN; ++i) {
      s += weights_[i] * f(m + h * nodes_[i]);
    }
    return s * h;
  }

  // Integral over [0, len] split geometrically so that the first piece has rate * length <= 0.5 and
  // each later piece doubles: exact to rounding for integrands like u^n e^{-rate u}.
  template <class F>
  double integrate_decaying(F&& f, double len, double rate) const {
    int k = 0;
    while (k < 80 && len * rate * std::ldexp(1.0, -k) > 0.5) {
      ++k;
    }
    double a = 0.0;
    double b = std::ldexp(len, -k);
    double s = integrate(f, a, b);
    for (int j = 0; j < k; ++j) {
      a = b;
      b = 2.0 * b;
      s += integrate(f, a, b);
    }
    return s;
  }

  // Vector-valued version (f fills out[0..n) at a point).
  template <std::size_t M, class F>
  std::array<double, M> integrate_decaying_vec(F&& f, double len, double rate) const {
    std::array<double, M> acc{};
    int k = 0;
    while (k < 80 && len * rate * std::ldexp(1.0, -k) > 0.5) {
      ++k;
    }
    double a = 0.0;
    double b = std::ldexp(len, -k);
    for (int j = 0; j <= k; ++j) {
      const double h = 0.5 * (b - a);
      const double m = 0.5 * (a + b);
      std::array<double, M> v{};
      for (std::size_t i = 0; i < kN; ++i) {
        f(m + h * nodes_[i], v);
        for (std::size_t c = 0; c < M; ++c) {
          acc[c] += h * weights_[i] * v[c];
        }
      }
      a = b;
      b = 2.0 * b;
    }
    return acc;
  }

 private:
  std::array<double, kN> nodes_{};
  std::array<double, kN> weights_{};
};

// Erlang kernel h(s) = beta^r s^{r-1} e^{-beta s} / (r-1)! for r in {1, 2}.
inline double erlang_kernel(std::size_t r, double beta, double s) {
  if (s < 0.0) {
    return 0.0;
  }
  return r == 1 ? beta * std::exp(-beta * s) : beta * beta * s * std::exp(-beta * s);
}

}  // namespace pulselob::testkit
