#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "pulselob/core/fixed_matrix.hpp"
#include "pulselob/core/time.hpp"

namespace pulselob::hawkes {

// Time-rescaling CUSUM with variable forgetting (math_spec Section 5.5).
//
// Per channel i the detector accumulates the compensator since channel i's previous event,
// xi = Lambda_i(t_{i,n}) - Lambda_i(t_{i,n-1}), which is Exp(1) under a correct model, and runs two
// CUSUMs (events faster / slower than predicted). An alarm switches the forgetting rate to
// kappa_fast for a dwell time and resets every CUSUM.
//
// Batch rule (spec 0.2.0): alarms are detected per event, but the reset and the start of the dwell
// take effect when the tie batch closes (the first strictly later timestamp). The forgetting rate is
// only ever used for later intervals, so this changes nothing for distinct timestamps, and it makes
// the detector invariant to the order of events inside a batch (Proposition 4).
template <std::size_t D>
class RegimeDetector {
 public:
  void init(bool enabled, double rho, double threshold, double kappa_slow, double kappa_fast,
            double dwell_s) noexcept {
    enabled_ = enabled;
    ln_rho_ = std::log(rho);
    up_slope_ = rho - 1.0;
    down_slope_ = 1.0 / rho - 1.0;
    threshold_ = threshold;
    kappa_slow_ = kappa_slow;
    kappa_fast_ = kappa_fast;
    dwell_ns_ = static_cast<core::DurationNs>(std::llround(dwell_s * 1e9));
  }

  // Forgetting rate for the interval that starts at t_start (decided at its left endpoint).
  double kappa(core::TimestampNs t_start) const noexcept {
    return t_start < fast_until_ns_ ? kappa_fast_ : kappa_slow_;
  }
  bool fast(core::TimestampNs t) const noexcept { return t < fast_until_ns_; }

  void add_compensator(const core::FixedVector<D>& dlambda) noexcept {
    for (std::size_t i = 0; i < D; ++i) {
      since_last_[i] += dlambda[i];
    }
  }

  // Residual of an event of channel i. Returns true if this update raised an alarm.
  bool on_event(std::size_t i) noexcept {
    const double xi = since_last_[i];
    since_last_[i] = 0.0;
    last_xi_ = xi;
    if (!enabled_) {
      return false;
    }
    double& up = cusum_(0, i);
    double& down = cusum_(1, i);
    up = std::fmax(0.0, up + ln_rho_ - up_slope_ * xi);
    down = std::fmax(0.0, down - ln_rho_ - down_slope_ * xi);
    const bool alarm = up > threshold_ || down > threshold_;
    alarm_pending_ = alarm_pending_ || alarm;
    return alarm;
  }

  // Closes the tie batch at time t_batch. Returns true if an alarm took effect.
  bool close_batch(core::TimestampNs t_batch) noexcept {
    if (!alarm_pending_) {
      return false;
    }
    alarm_pending_ = false;
    cusum_ = {};
    fast_until_ns_ = t_batch + dwell_ns_;
    ++alarms_;
    return true;
  }

  const core::FixedMatrix<2, D>& cusum() const noexcept { return cusum_; }
  const core::FixedVector<D>& since_last() const noexcept { return since_last_; }
  double last_residual() const noexcept { return last_xi_; }
  std::uint64_t alarms() const noexcept { return alarms_; }
  core::TimestampNs fast_until() const noexcept { return fast_until_ns_; }

 private:
  core::FixedVector<D> since_last_{};
  core::FixedMatrix<2, D> cusum_{};
  double ln_rho_ = 0.0;
  double up_slope_ = 0.0;
  double down_slope_ = 0.0;
  double threshold_ = 0.0;
  double kappa_slow_ = 0.0;
  double kappa_fast_ = 0.0;
  double last_xi_ = 0.0;
  core::DurationNs dwell_ns_ = 0;
  core::TimestampNs fast_until_ns_ = INT64_MIN;
  std::uint64_t alarms_ = 0;
  bool alarm_pending_ = false;
  bool enabled_ = true;
};

}  // namespace pulselob::hawkes
