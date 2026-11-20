#pragma once

#include <cmath>
#include <cstddef>

#include "pulselob/core/fixed_matrix.hpp"

namespace pulselob::hawkes {

// Mark function and its online normalization (math_spec Section 2.3).
struct MarkConfig {
  double c_u = 0.5;
  double c_L = 0.5;
  double c_iota = 0.5;
  double u_max = 1e4;
  double L_max = 50.0;
  double eta_g = 0.01;   // EWMA weight of the normalizer g-bar
  double eta_nu = 0.01;  // Robbins-Monro step of the log-median tracker

  double g_min() const noexcept { return 1.0 - c_iota; }
  double g_max() const noexcept {
    return (1.0 + c_u * std::log1p(u_max)) * (1.0 + c_L * std::log1p(L_max)) * (1.0 + c_iota);
  }
};

inline const char* validate(const MarkConfig& c) noexcept {
  if (!(std::isfinite(c.c_u) && c.c_u >= 0.0 && std::isfinite(c.c_L) && c.c_L >= 0.0)) {
    return "c_u and c_L must be finite and non-negative";
  }
  if (!(std::isfinite(c.c_iota) && c.c_iota >= 0.0 && c.c_iota < 1.0)) {
    return "c_iota must lie in [0, 1)";
  }
  if (!(std::isfinite(c.u_max) && c.u_max > 0.0 && std::isfinite(c.L_max) && c.L_max >= 0.0)) {
    return "u_max must be positive and L_max non-negative";
  }
  if (!(c.eta_g > 0.0 && c.eta_g <= 1.0 && c.eta_nu > 0.0 && std::isfinite(c.eta_nu))) {
    return "normalizer rates must lie in (0, 1]";
  }
  return nullptr;
}

// Raw mark g~(u, L, iota) (Lemma 1: always in [g_min, g_max]).
inline double raw_mark(const MarkConfig& c, double u, double L, double iota) noexcept {
  return (1.0 + c.c_u * std::log1p(std::fmin(u, c.u_max))) *
         (1.0 + c.c_L * std::log1p(std::fmin(L, c.L_max))) * (1.0 + c.c_iota * iota);
}

// Per-channel normalizer: a sign-based Robbins-Monro tracker of the median log size and an
// event-time EWMA of g~. Both are read before being updated with the current event (causal). The
// first event of a channel initializes both, so its relative size is 1 and its mark is 1.
template <std::size_t D>
class MarkNormalizer {
 public:
  void init(const MarkConfig& c) noexcept {
    cfg_ = c;
    gmin_ = c.g_min();
    gmax_ = c.g_max();
    seen_ = {};
  }

  // Pending update of one channel's statistics, produced by peek() and applied by commit().
  struct Update {
    std::size_t channel = 0;
    double nu = 0.0;
    double gbar = 0.0;
  };

  // Returns g_k for an event of channel j with the given size (> 0), sweep depth (ticks, >= 0) and
  // aligned imbalance iota in [-1, 1], without changing the statistics; commit(up) applies the
  // update. Splitting the two lets the engine leave the statistics untouched if the event is rejected.
  double peek(std::size_t j, double size, double depth, double iota, Update& up) const noexcept {
    const double ell = std::log(size);
    double nu = nu_[j];
    double gbar = gbar_[j];
    if (seen_[j] == 0.0) {
      nu = ell;
      gbar = std::fmin(std::fmax(raw_mark(cfg_, 1.0, depth, iota), gmin_), gmax_);
    }
    const double u = size / std::exp(nu);
    const double gt = raw_mark(cfg_, u, depth, iota);
    const double g = gt / gbar;
    up.channel = j;
    up.nu = nu + cfg_.eta_nu * ((ell > nu ? 1.0 : 0.0) - (ell < nu ? 1.0 : 0.0));
    up.gbar = std::fmin(std::fmax(gbar + cfg_.eta_g * (gt - gbar), gmin_), gmax_);
    return g;
  }

  void commit(const Update& up) noexcept {
    seen_[up.channel] = 1.0;
    nu_[up.channel] = up.nu;
    gbar_[up.channel] = up.gbar;
  }

  double mark(std::size_t j, double size, double depth, double iota) noexcept {
    Update up{};
    const double g = peek(j, size, depth, iota, up);
    commit(up);
    return g;
  }

  const core::FixedVector<D>& log_median() const noexcept { return nu_; }
  const core::FixedVector<D>& g_bar() const noexcept { return gbar_; }

 private:
  MarkConfig cfg_{};
  core::FixedVector<D> nu_{};
  core::FixedVector<D> gbar_{};
  core::FixedVector<D> seen_{};
  double gmin_ = 0.5;
  double gmax_ = 25.0;
};

}  // namespace pulselob::hawkes
