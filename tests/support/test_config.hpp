#pragma once

// Default configurations for tests and benchmarks (math_spec Appendix B starting values), with a
// prior close to the simulated ground truth of hawkes_sim.hpp.

#include <cstddef>

#include "pulselob/core/model_shape.hpp"
#include "pulselob/engine/config.hpp"
#include "pulselob/feed/market_event.hpp"
#include "pulselob/hawkes/model.hpp"

namespace pulselob::testkit {

using Shape = core::DefaultShape;
inline constexpr std::size_t kD = Shape::D;
inline constexpr std::size_t kP = Shape::P;

// Channels: 0 perp buy, 1 perp sell, 2 spot buy, 3 spot sell. Kernel bank (1000, 200, 50, 1) /s.
// Ground truth: self-excitation on the fast and slow exponentials, perp -> spot lead-lag on the
// 5 ms Erlang-2 hump.
inline core::FixedMatrix<kD, kP> ground_truth_theta(double cross = 0.15) {
  core::FixedMatrix<kD, kP> th{};
  const double mu[kD] = {20.0, 20.0, 12.0, 12.0};
  for (std::size_t i = 0; i < kD; ++i) {
    th(i, 0) = mu[i];
    th(i, Shape::coord(i, 0)) = 0.25;  // 1 ms self-excitation
    th(i, Shape::coord(i, 3)) = 0.10;  // 1 s clustering
  }
  th(2, Shape::coord(0, 1)) = cross;  // perp buy -> spot buy, 5 ms hump
  th(3, Shape::coord(1, 1)) = cross;  // perp sell -> spot sell
  return th;
}

inline std::array<double, Shape::Q> default_beta() {
  return {1000.0, 200.0, 50.0, 1.0};
}

inline engine::EngineConfig<Shape> default_engine_config() {
  engine::EngineConfig<Shape> c{};
  c.model.beta = default_beta();
  c.model.theta0 = ground_truth_theta();
  // A weak, uniform prior on the excitation weights.
  for (std::size_t i = 0; i < kD; ++i) {
    for (std::size_t p = 1; p < kP; ++p) {
      c.model.theta0(i, p) = c.model.theta0(i, p) > 0.0 ? c.model.theta0(i, p) : 0.01;
    }
  }
  hawkes::set_scale_free_ridge(c.model, 1.0);
  c.forecast.horizon_s = 0.05;
  c.forecast.neumann_terms = 3;
  c.forecast.direction = {+1, -1, +1, -1};
  c.forecast.impact = {1.0, 1.0, 0.6, 0.6};
  c.channels.direction = c.forecast.direction;
  c.channels.instruments[0] = {true, 0, 10'000'000};  // perp, tick 0.1
  c.channels.instruments[1] = {true, 1, 1'000'000};   // spot, tick 0.01
  c.channels.set_channel(0, feed::Side::Buy, 0);
  c.channels.set_channel(0, feed::Side::Sell, 1);
  c.channels.set_channel(1, feed::Side::Buy, 2);
  c.channels.set_channel(1, feed::Side::Sell, 3);
  c.quote_instrument = 1;
  c.control.gamma = 0.05;
  c.control.k = 2.0;
  c.control.horizon_inv_s = 1.0;
  c.control.sigma_w2 = 0.5;
  c.control.delta_max = 2.0;
  c.control.q_max = 10.0;
  c.control.jump_benign = {0.3, 0.3, 0.2, 0.2};
  c.control.jump_informed = {1.5, 1.5, 1.0, 1.0};
  c.control.buy_channel = 2;
  c.control.sell_channel = 3;
  return c;
}

}  // namespace pulselob::testkit
