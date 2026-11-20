#pragma once

// Bitwise comparison of model state, field by field. memcmp over whole objects is not used: the
// 64-byte-aligned state types contain padding, whose bytes are indeterminate.

#include <cstring>
#include <type_traits>

#include "pulselob/hawkes/estimator.hpp"
#include "pulselob/hawkes/model.hpp"
#include "pulselob/hawkes/stability.hpp"

namespace pulselob::testkit {

template <class T>
bool same_bits(const T& a, const T& b) {
  static_assert(std::is_trivially_copyable_v<T>);
  return std::memcmp(&a, &b, sizeof(T)) == 0;
}

// Containers of doubles (FixedMatrix / FixedVector) have no padding beyond their zero lanes.
template <class Shape>
bool same_served(const hawkes::Served<Shape>& a, const hawkes::Served<Shape>& b) {
  return same_bits(a.theta, b.theta) && same_bits(a.G, b.G) && same_bits(a.lambda_inf, b.lambda_inf) &&
         same_bits(a.v, b.v) && same_bits(a.criticality, b.criticality) && same_bits(a.n_eff, b.n_eff) &&
         a.steps == b.steps && a.projected_rows == b.projected_rows && a.flags == b.flags;
}

template <class Shape>
bool same_stats(const hawkes::LsStatistics<Shape>& a, const hawkes::LsStatistics<Shape>& b) {
  return same_bits(a.M, b.M) && same_bits(a.C, b.C) && same_bits(a.omega_eff, b.omega_eff);
}

template <class Shape>
bool same_estimator(const hawkes::CholeskyEstimator<Shape>& a, const hawkes::CholeskyEstimator<Shape>& b) {
  return same_bits(a.theta(), b.theta()) && same_stats(a.stats(), b.stats()) &&
         a.factorization_failures() == b.factorization_failures();
}

template <class Shape>
bool same_estimator(const hawkes::PotterEstimator<Shape>& a, const hawkes::PotterEstimator<Shape>& b) {
  return same_bits(a.theta(), b.theta()) && same_bits(a.U(), b.U()) && same_stats(a.stats(), b.stats()) &&
         a.factorization_failures() == b.factorization_failures();
}

template <class Shape, class Est>
bool same_model_state(const hawkes::Model<Shape, Est>& a, const hawkes::Model<Shape, Est>& b) {
  return same_bits(a.kernel().R, b.kernel().R) && same_bits(a.kernel().pending, b.kernel().pending) &&
         same_estimator(a.estimator(), b.estimator()) && same_served(a.served(), b.served()) &&
         same_bits(a.detector().cusum(), b.detector().cusum()) &&
         same_bits(a.detector().since_last(), b.detector().since_last()) &&
         a.detector().fast_until() == b.detector().fast_until() && a.t_prev() == b.t_prev() &&
         a.started() == b.started() && same_bits(a.n_eff(), b.n_eff()) &&
         a.counters().steps == b.counters().steps;
}

}  // namespace pulselob::testkit
