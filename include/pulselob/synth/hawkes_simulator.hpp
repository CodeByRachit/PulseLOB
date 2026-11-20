#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "pulselob/core/fixed_matrix.hpp"

namespace pulselob::synth {

// Portable random numbers: std::mt19937_64 is fully specified by the standard, unlike the standard
// distributions, so a seed gives the same stream on every platform (up to libm differences in
// log/cos, which only matter for decisions within an ulp of a threshold).
class Rng {
 public:
  explicit Rng(std::uint64_t seed) : gen_(seed) {}
  double uniform() { return static_cast<double>(gen_() >> 11) * 0x1.0p-53; }              // [0, 1)
  double uniform_pos() { return (static_cast<double>(gen_() >> 11) + 0.5) * 0x1.0p-53; }  // (0, 1)
  double exponential(double rate) { return -std::log(uniform_pos()) / rate; }
  double normal() {
    const double u1 = uniform_pos();
    const double u2 = uniform();
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
  }
  std::uint64_t next() { return gen_(); }

 private:
  std::mt19937_64 gen_;
};

struct ChannelEvent {
  double t_s;
  int channel;
  double g;
};

// Exact simulation of the linear multivariate Hawkes model of math_spec Section 3 (unit marks) by
// Ogata thinning, with piecewise-constant parameters: segment k uses theta_k on [t_k, t_{k+1}).
// The kernel bank and its exact state recursion are those of the engine. Between events the
// total intensity is bounded using e^{-b u}(R1 + u R0) <= R1 + R0 / (e b) for Erlang-2 bases;
// the bound is recomputed after every candidate and at every segment boundary.
//
// Segments express every Phase 5 scenario: a stationary regime (one segment), a regime shift
// (two), exogenous bursts (short segments with a raised baseline), and cascades.
template <class Shape>
class HawkesSimulator {
 public:
  static constexpr std::size_t D = Shape::D;
  static constexpr std::size_t Q = Shape::Q;
  static constexpr std::size_t P = Shape::P;
  using Theta = core::FixedMatrix<D, P>;

  struct Segment {
    double t_start_s;
    Theta theta;
  };

  HawkesSimulator(const std::array<double, Q>& beta, const Theta& theta) : beta_(beta) {
    segments_.push_back({0.0, theta});
  }

  // Parameters theta2 from t_switch_s on (appended; segments must be added in time order).
  void schedule_switch(double t_switch_s, const Theta& theta2) { segments_.push_back({t_switch_s, theta2}); }
  const std::vector<Segment>& segments() const { return segments_; }

  // Simulates on [0, horizon_s). If lambda_true is non-null, the left-limit intensities
  // lambda(t_k-) of all D channels are appended for every accepted event (row-major, D per event).
  std::vector<ChannelEvent> simulate(double horizon_s, Rng& rng, std::size_t max_events = 50'000'000,
                                     std::vector<double>* lambda_true = nullptr) {
    std::vector<ChannelEvent> out;
    std::array<std::array<double, 2>, Q * D> R{};  // R[j*Q+q] = {R0, R1}
    double t = 0.0;
    std::size_t seg = 0;
    while (out.size() < max_events) {
      while (seg + 1 < segments_.size() && t >= segments_[seg + 1].t_start_s) {
        ++seg;
      }
      const Theta& th = segments_[seg].theta;
      const double next_boundary = seg + 1 < segments_.size() ? segments_[seg + 1].t_start_s : horizon_s;
      double bound = 0.0;
      for (std::size_t i = 0; i < D; ++i) {
        double s = th(i, 0);
        for (std::size_t j = 0; j < D; ++j) {
          for (std::size_t q = 0; q < Q; ++q) {
            const auto& r = R[j * Q + q];
            const double b = beta_[q];
            const double xb =
                Shape::orders[q] == 1 ? b * r[0] : b * b * (r[1] + r[0] / (2.718281828459045 * b));
            s += th(i, Shape::coord(j, q)) * xb;
          }
        }
        bound += s;
      }
      const double w = rng.exponential(bound);
      if (seg + 1 < segments_.size() && t + w >= next_boundary) {
        decay(R, next_boundary - t);  // do not step over a parameter change with the old bound
        t = next_boundary;
        continue;
      }
      if (t + w > horizon_s) {
        break;
      }
      decay(R, w);
      t += w;
      std::array<double, D> lam{};
      double total = 0.0;
      for (std::size_t i = 0; i < D; ++i) {
        double s = th(i, 0);
        for (std::size_t j = 0; j < D; ++j) {
          for (std::size_t q = 0; q < Q; ++q) {
            const auto& r = R[j * Q + q];
            const double x = Shape::orders[q] == 1 ? beta_[q] * r[0] : beta_[q] * beta_[q] * r[1];
            s += th(i, Shape::coord(j, q)) * x;
          }
        }
        lam[i] = s;
        total += s;
      }
      const double u = rng.uniform() * bound;
      if (u >= total) {
        continue;  // rejected candidate
      }
      std::size_t ch = 0;
      double acc = lam[0];
      while (u >= acc && ch + 1 < D) {
        ++ch;
        acc += lam[ch];
      }
      out.push_back({t, static_cast<int>(ch), 1.0});
      if (lambda_true != nullptr) {
        lambda_true->insert(lambda_true->end(), lam.begin(), lam.end());
      }
      for (std::size_t q = 0; q < Q; ++q) {
        R[ch * Q + q][0] += 1.0;
      }
    }
    return out;
  }

 private:
  void decay(std::array<std::array<double, 2>, Q * D>& R, double dt) const {
    for (std::size_t j = 0; j < D; ++j) {
      for (std::size_t q = 0; q < Q; ++q) {
        auto& r = R[j * Q + q];
        const double e = std::exp(-beta_[q] * dt);
        if (Shape::orders[q] == 2) {
          r[1] = e * (r[1] + dt * r[0]);
        }
        r[0] *= e;
      }
    }
  }

  std::array<double, Q> beta_;
  std::vector<Segment> segments_;
};

}  // namespace pulselob::synth
