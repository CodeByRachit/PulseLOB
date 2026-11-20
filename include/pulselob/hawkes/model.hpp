#pragma once

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "pulselob/core/fixed_matrix.hpp"
#include "pulselob/core/linalg.hpp"
#include "pulselob/core/time.hpp"
#include "pulselob/hawkes/config.hpp"
#include "pulselob/hawkes/estimator.hpp"
#include "pulselob/hawkes/kernel.hpp"
#include "pulselob/hawkes/regime.hpp"
#include "pulselob/hawkes/stability.hpp"

namespace pulselob::hawkes {

// Scale-free ridge of math_spec Section 5.4: omega_p = T_omega * E[z_p^2], with the second moments
// of the stationary process under the prior Theta_0 in the Poisson (Campbell) approximation:
// E[x^q_j] = lambda_j, Var[x^q_j] ~ lambda_j int h_q^2 (int h^2 = beta/2 for Erlang-1, beta/4 for
// Erlang-2). Cold path; overwrites cfg.omega.
template <class Shape>
inline void set_scale_free_ridge(ModelConfig<Shape>& cfg, double t_omega_s) noexcept {
  Served<Shape> s{};
  StabilityGuard<Shape>::branching(cfg.theta0, s.G, false);
  for (std::size_t i = 0; i < Shape::D; ++i) {
    s.theta(i, 0) = cfg.theta0(i, 0);
  }
  StabilityGuard<Shape>::solve_lambda_inf(s);
  cfg.omega[0] = t_omega_s;
  for (std::size_t j = 0; j < Shape::D; ++j) {
    for (std::size_t q = 0; q < Shape::Q; ++q) {
      const double lam = s.lambda_inf[j];
      const double h2 = Shape::orders[q] == 1 ? cfg.beta[q] / 2.0 : cfg.beta[q] / 4.0;
      cfg.omega[Shape::coord(j, q)] = t_omega_s * (lam * lam + lam * h2);
    }
  }
}

// Counters of the model's numerical guards (docs/architecture.md Section 8).
struct ModelCounters {
  std::uint64_t steps = 0;
  std::uint64_t events = 0;
  std::uint64_t heartbeats = 0;
  std::uint64_t intervals = 0;
  std::uint64_t cutoffs = 0;          // kernel bases zeroed by the cutoff (Section 4.6)
  std::uint64_t long_gaps = 0;        // full ridge restores after an interval longer than T_hb
  std::uint64_t projections = 0;      // rows projected (Section 6.4)
  std::uint64_t alarms = 0;           // CUSUM alarms that took effect
  std::uint64_t rejected_inputs = 0;  // step() calls with a bad channel, mark or time
};

// The model half of the per-event algorithm (math_spec Section 9, steps 2 to 9), on channel events
// (t, d, g). It knows nothing about markets; feed::EventMapper and the engine produce its input.
//
// Estimator is PotterEstimator<Shape> or CholeskyEstimator<Shape>. The object is fixed-size and
// trivially copyable: a snapshot is a copy.
template <class Shape, class Estimator>
class Model {
 public:
  static constexpr std::size_t D = Shape::D;
  static constexpr std::size_t P = Shape::P;
  using Layout = BankLayout<Shape>;

  enum class StepResult : std::uint8_t { Ok = 0, Rejected = 1 };

  Model() noexcept = default;

  // cfg must be valid (validate(cfg) == nullptr); the engine checks this before constructing.
  explicit Model(const ModelConfig<Shape>& cfg) noexcept { init(cfg); }

  void init(const ModelConfig<Shape>& cfg) noexcept {
    bank_ = KernelBank<Shape>(cfg.beta, cfg.x_cut);
    heartbeat_ns_ = cfg.heartbeat_ns;
    estimator_.init(cfg);
    detector_.init(cfg.cusum_enabled, cfg.cusum_rho, cfg.cusum_threshold, cfg.kappa_slow(), cfg.kappa_fast(),
                   cfg.dwell_s);
    guard_.init(cfg, served_);
    kernel_ = {};
    started_ = false;
    t_prev_ = 0;
    step_count_ = 0;
    counters_ = {};
  }

  // One step at time t (non-decreasing; the engine enforces the out-of-order policy). channel < 0 is a
  // heartbeat. g must be finite and positive for events. Rejected input leaves the state unchanged.
  StepResult step(core::TimestampNs t, int channel, double g) noexcept {
    const bool is_event = channel >= 0;
    if ((is_event && (static_cast<std::size_t>(channel) >= D || !std::isfinite(g) || !(g > 0.0))) ||
        (started_ && t < t_prev_)) {
      ++counters_.rejected_inputs;
      return StepResult::Rejected;
    }
    if (!started_) {
      started_ = true;
      t_prev_ = t;
    }
    const core::DurationNs dt_ns = t - t_prev_;
    if (dt_ns > 0) {
      advance(t, dt_ns);
    }
    if (is_event) {
      const auto i = static_cast<std::size_t>(channel);
      kernel_.regressor(bank_, z_);  // z(t-): pending batch excluded
      detector_.on_event(i);
      estimator_.event(i, z_);
      n_eff_ += 1.0;
      ++counters_.events;
    } else {
      ++counters_.heartbeats;
    }
    estimator_.replenish(step_count_ % P);
    ++step_count_;
    estimator_.finish();
    guard_.serve(estimator_.theta(), served_);
    counters_.projections += static_cast<std::uint64_t>(std::popcount(served_.projected_rows));
    served_.n_eff = n_eff_;
    served_.steps = step_count_;
    served_.flags = detector_.fast(t) ? Served<Shape>::kFlagFastForgetting : 0U;
    if (is_event) {
      kernel_.add_pending(static_cast<std::size_t>(channel), g);
    }
    ++counters_.steps;
    return StepResult::Ok;
  }

  const KernelBank<Shape>& bank() const noexcept { return bank_; }
  const KernelState<Shape>& kernel() const noexcept { return kernel_; }
  const Estimator& estimator() const noexcept { return estimator_; }
  const RegimeDetector<D>& detector() const noexcept { return detector_; }
  const Served<Shape>& served() const noexcept { return served_; }
  const StabilityGuard<Shape>& guard() const noexcept { return guard_; }
  const ModelCounters& counters() const noexcept { return counters_; }
  const core::FixedVector<P>& last_regressor() const noexcept { return z_; }
  const core::FixedVector<P>& last_compensator() const noexcept { return Z_; }
  const IntervalTerms<Shape>& last_interval() const noexcept { return it_; }
  core::TimestampNs t_prev() const noexcept { return t_prev_; }
  bool started() const noexcept { return started_; }
  double n_eff() const noexcept { return n_eff_; }

 private:
  // Steps 2 and 3: flush the batch, then the exact interval update (t_prev, t].
  void advance(core::TimestampNs t, core::DurationNs dt_ns) noexcept {
    if (detector_.close_batch(t_prev_)) {
      ++counters_.alarms;
    }
    kernel_.flush();
    const double dt = core::ns_to_seconds(dt_ns);
    compute_interval(bank_, dt, detector_.kappa(t_prev_), it_);
    // Compensators with the served parameters, for the residuals.
    kernel_.compensator(it_, Z_);
    core::FixedVector<D> dl{};
    core::matvec(served_.theta, Z_, dl);
    detector_.add_compensator(dl);
    const bool long_gap = dt_ns > heartbeat_ns_;
    estimator_.interval(bank_, kernel_, it_, long_gap);
    counters_.cutoffs += kernel_.decay(it_);
    n_eff_ *= it_.d;
    if (long_gap) {
      estimator_.full_restore();
      ++counters_.long_gaps;
    }
    t_prev_ = t;
    ++counters_.intervals;
  }

  KernelBank<Shape> bank_{};
  KernelState<Shape> kernel_{};
  Estimator estimator_{};
  RegimeDetector<D> detector_{};
  StabilityGuard<Shape> guard_{};
  Served<Shape> served_{};
  IntervalTerms<Shape> it_{};
  core::FixedVector<P> z_{};
  core::FixedVector<P> Z_{};
  ModelCounters counters_{};
  core::DurationNs heartbeat_ns_ = 100'000'000;
  core::TimestampNs t_prev_ = 0;
  std::uint64_t step_count_ = 0;
  double n_eff_ = 0.0;
  bool started_ = false;
};

}  // namespace pulselob::hawkes
