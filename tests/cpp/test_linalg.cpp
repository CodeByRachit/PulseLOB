#include <gtest/gtest.h>

#include <cmath>

#include "hawkes_sim.hpp"
#include "pulselob/core/linalg.hpp"

namespace {

using pulselob::core::FixedMatrix;
using pulselob::core::FixedVector;
using pulselob::testkit::Rng;

constexpr std::size_t N = 17;

template <std::size_t R, std::size_t C>
void fill(FixedMatrix<R, C>& A, Rng& rng) {
  for (std::size_t r = 0; r < R; ++r) {
    for (std::size_t c = 0; c < C; ++c) {
      A(r, c) = rng.normal();
    }
  }
}

template <std::size_t M>
void fill(FixedVector<M>& v, Rng& rng) {
  for (std::size_t i = 0; i < M; ++i) {
    v[i] = rng.normal();
  }
}

// Random SPD matrix B B^T + n I (lower and upper filled).
FixedMatrix<N, N> random_spd(Rng& rng) {
  FixedMatrix<N, N> B{};
  fill(B, rng);
  FixedMatrix<N, N> A{};
  for (std::size_t i = 0; i < N; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      double s = i == j ? static_cast<double>(N) : 0.0;
      for (std::size_t k = 0; k < N; ++k) {
        s += B(i, k) * B(j, k);
      }
      A(i, j) = s;
    }
  }
  return A;
}

TEST(Linalg, KernelsMatchNaiveLoopsAndKeepPaddingZero) {
  Rng rng(1);
  FixedMatrix<4, N> A{};
  fill(A, rng);
  FixedVector<N> x{};
  fill(x, rng);
  FixedVector<4> y{};
  pulselob::core::matvec(A, x, y);
  for (std::size_t r = 0; r < 4; ++r) {
    double s = 0.0;
    for (std::size_t c = 0; c < N; ++c) {
      s += A(r, c) * x[c];
    }
    EXPECT_NEAR(y[r], s, 1e-13);
  }
  FixedVector<4> u{};
  fill(u, rng);
  FixedVector<N> yt{};
  pulselob::core::matvec_transposed(A, u, yt);
  for (std::size_t c = 0; c < N; ++c) {
    double s = 0.0;
    for (std::size_t r = 0; r < 4; ++r) {
      s += A(r, c) * u[r];
    }
    EXPECT_NEAR(yt[c], s, 1e-13);
  }
  EXPECT_TRUE(yt.padding_is_zero());
  pulselob::core::rank_one_sub(A, 0.5, u, x);
  pulselob::core::scale(A, 1.5);
  EXPECT_TRUE(A.padding_is_zero());
}

TEST(Linalg, CholeskyInverseFactorAndSolve) {
  Rng rng(2);
  const FixedMatrix<N, N> A = random_spd(rng);
  FixedMatrix<N, N> L = A;
  FixedVector<N> col{};
  ASSERT_TRUE(pulselob::core::cholesky_lower(L, col));
  // L L^T == A.
  double worst = 0.0;
  for (std::size_t i = 0; i < N; ++i) {
    for (std::size_t j = 0; j <= i; ++j) {
      double s = 0.0;
      for (std::size_t k = 0; k <= j; ++k) {
        s += L(i, k) * L(j, k);
      }
      worst = std::fmax(worst, std::fabs(s - A(i, j)) / std::fabs(A(i, i)));
    }
  }
  EXPECT_LT(worst, 1e-14);
  // U U^T A == I with U = L^{-T}.
  FixedMatrix<N, N> U{};
  pulselob::core::inverse_transpose_factor(L, U);
  EXPECT_TRUE(U.padding_is_zero());
  double worst_inv = 0.0;
  for (std::size_t i = 0; i < N; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      double s = 0.0;
      for (std::size_t k = 0; k < N; ++k) {
        double uu = 0.0;
        for (std::size_t m = 0; m < N; ++m) {
          uu += U(i, m) * U(k, m);
        }
        s += uu * A(k, j);
      }
      worst_inv = std::fmax(worst_inv, std::fabs(s - (i == j ? 1.0 : 0.0)));
    }
  }
  EXPECT_LT(worst_inv, 1e-12);
  // Block solve: rows of B replaced by A^{-1} B_r^T.
  FixedMatrix<4, N> B{};
  fill(B, rng);
  FixedMatrix<4, N> X = B;
  FixedMatrix<N, 4> scratch{};
  pulselob::core::cholesky_solve_rows(L, X, scratch);
  EXPECT_TRUE(X.padding_is_zero());
  for (std::size_t r = 0; r < 4; ++r) {
    for (std::size_t i = 0; i < N; ++i) {
      double s = 0.0;
      for (std::size_t k = 0; k < N; ++k) {
        s += A(i, k) * X(r, k);
      }
      EXPECT_NEAR(s, B(r, i), 1e-11);
    }
  }
}

TEST(Linalg, CholeskyRejectsIndefiniteAndNaN) {
  FixedMatrix<3, 3> A{};
  A(0, 0) = 1.0;
  A(1, 1) = -1.0;
  A(2, 2) = 1.0;
  FixedVector<3> col{};
  EXPECT_FALSE(pulselob::core::cholesky_lower(A, col));
  FixedMatrix<3, 3> B{};
  B(0, 0) = std::nan("");
  EXPECT_FALSE(pulselob::core::cholesky_lower(B, col));
}

TEST(Linalg, ReductionOrderIsFixed) {
  // Same inputs, same bits: the reduction tree is defined by the source.
  Rng rng(3);
  FixedVector<N> a{};
  FixedVector<N> b{};
  fill(a, rng);
  fill(b, rng);
  const double d1 = pulselob::core::dot(a, b);
  const double d2 = pulselob::core::dot(a, b);
  EXPECT_EQ(d1, d2);
}

}  // namespace
