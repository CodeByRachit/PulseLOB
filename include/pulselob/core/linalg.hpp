#pragma once

#include <array>
#include <cmath>
#include <cstddef>

#include "pulselob/core/fixed_matrix.hpp"
#include "pulselob/core/platform.hpp"

namespace pulselob::core {

// The small set of dense kernels the estimator needs (docs/architecture.md Section 5.2). Every
// kernel runs over full padded rows (a multiple of 8 lanes), relying on the zero-padding invariant
// of FixedMatrix / FixedVector, and every kernel preserves that invariant.
//
// Reductions use eight independent accumulators combined in a fixed tree. The order of operations
// is therefore fixed by the source, not by the optimizer: results are bitwise reproducible under
// -ffp-contract=off without fast-math, and the loop still vectorizes.

template <std::size_t Lanes>
inline double dot_lanes(const double* a, const double* b) noexcept {
  static_assert(Lanes % kDoublesPerLine == 0, "dot_lanes needs whole padded rows");
  double acc[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  for (std::size_t i = 0; i < Lanes; i += 8) {
    for (std::size_t l = 0; l < 8; ++l) {
      acc[l] += a[i + l] * b[i + l];
    }
  }
  return ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
}

// y += alpha * x over full padded rows.
template <std::size_t Lanes>
inline void axpy_lanes(double* y, double alpha, const double* x) noexcept {
  for (std::size_t i = 0; i < Lanes; ++i) {
    y[i] += alpha * x[i];
  }
}

template <std::size_t Lanes>
inline void scale_lanes(double* y, double alpha) noexcept {
  for (std::size_t i = 0; i < Lanes; ++i) {
    y[i] *= alpha;
  }
}

template <std::size_t N>
inline double dot(const FixedVector<N>& a, const FixedVector<N>& b) noexcept {
  return dot_lanes<FixedVector<N>::padded>(a.data.data(), b.data.data());
}

// y = A x.
template <std::size_t R, std::size_t C>
inline void matvec(const FixedMatrix<R, C>& A, const FixedVector<C>& x, FixedVector<R>& y) noexcept {
  static_assert(FixedMatrix<R, C>::stride == FixedVector<C>::padded);
  for (std::size_t r = 0; r < R; ++r) {
    y[r] = dot_lanes<FixedMatrix<R, C>::stride>(A.data.data() + r * A.stride, x.data.data());
  }
}

// y = A^T x, computed as a sum of scaled rows (contiguous, vectorizable).
template <std::size_t R, std::size_t C>
inline void matvec_transposed(const FixedMatrix<R, C>& A, const FixedVector<R>& x,
                              FixedVector<C>& y) noexcept {
  static_assert(FixedMatrix<R, C>::stride == FixedVector<C>::padded);
  for (std::size_t i = 0; i < FixedVector<C>::padded; ++i) {
    y.data[i] = 0.0;
  }
  for (std::size_t r = 0; r < R; ++r) {
    axpy_lanes<FixedMatrix<R, C>::stride>(y.data.data(), x[r], A.data.data() + r * A.stride);
  }
}

// A -= alpha * u v^T.
template <std::size_t R, std::size_t C>
inline void rank_one_sub(FixedMatrix<R, C>& A, double alpha, const FixedVector<R>& u,
                         const FixedVector<C>& v) noexcept {
  static_assert(FixedMatrix<R, C>::stride == FixedVector<C>::padded);
  for (std::size_t r = 0; r < R; ++r) {
    axpy_lanes<FixedMatrix<R, C>::stride>(A.data.data() + r * A.stride, -alpha * u[r], v.data.data());
  }
}

template <std::size_t R, std::size_t C>
inline void scale(FixedMatrix<R, C>& A, double alpha) noexcept {
  scale_lanes<R * FixedMatrix<R, C>::stride>(A.data.data(), alpha);
}

template <std::size_t N>
inline void scale(FixedVector<N>& v, double alpha) noexcept {
  scale_lanes<FixedVector<N>::padded>(v.data.data(), alpha);
}

// In-place Cholesky factorization A = L L^T of a symmetric positive-definite matrix whose lower
// triangle (diagonal included) holds A. On success the lower triangle holds L and the strict upper
// triangle is left untouched. Right-looking (outer-product) form: every inner loop is a contiguous
// row update, so there are no reductions. Returns false, with A partially overwritten, if a pivot is
// not strictly positive (or is NaN); callers keep their previous state in that case.
template <std::size_t N>
inline bool cholesky_lower(FixedMatrix<N, N>& A, FixedVector<N>& col_scratch) noexcept {
  for (std::size_t k = 0; k < N; ++k) {
    const double pivot = A(k, k);
    if (!(pivot > 0.0)) {
      return false;
    }
    const double lkk = std::sqrt(pivot);
    const double inv = 1.0 / lkk;
    A(k, k) = lkk;
    for (std::size_t i = k + 1; i < N; ++i) {
      A(i, k) *= inv;
      col_scratch[i] = A(i, k);
    }
    for (std::size_t i = k + 1; i < N; ++i) {
      const double lik = col_scratch[i];
      double* row = A.data.data() + i * A.stride;
      for (std::size_t j = k + 1; j <= i; ++j) {
        row[j] -= lik * col_scratch[j];
      }
    }
  }
  return true;
}

// Given L (lower triangle of Lmat, from cholesky_lower), writes U = L^{-T} into U (upper
// triangular, strict lower part zero), so that U U^T = (L L^T)^{-1}.
template <std::size_t N>
inline void inverse_transpose_factor(const FixedMatrix<N, N>& Lmat, FixedMatrix<N, N>& U) noexcept {
  // X = L^{-1} is lower triangular; column c solves L x = e_c by forward substitution.
  // U = X^T, so U(c, i) = X(i, c), i.e. row c of U holds column c of X.
  for (std::size_t i = 0; i < FixedMatrix<N, N>::rows * FixedMatrix<N, N>::stride; ++i) {
    U.data[i] = 0.0;
  }
  for (std::size_t c = 0; c < N; ++c) {
    double* x = U.data.data() + c * U.stride;  // x[i] = X(i, c)
    x[c] = 1.0 / Lmat(c, c);
    for (std::size_t i = c + 1; i < N; ++i) {
      double s = 0.0;
      for (std::size_t k = c; k < i; ++k) {
        s += Lmat(i, k) * x[k];
      }
      x[i] = -s / Lmat(i, i);
    }
  }
}

// Factors the SPD matrix whose lower triangle is in A (A = L L^T, L left in the lower triangle) and
// solves A X = B for R right-hand sides stored as rows of B (B[p][r] is coordinate p of right-hand
// side r), overwriting B with X. Forward elimination of the right-hand sides is fused into the
// right-looking factorization, so A is traversed once, and the right-hand-side rows are exactly R
// wide (no padding lanes). Returns false, leaving A and B partially overwritten, if a pivot is not
// strictly positive or is NaN.
template <std::size_t N, std::size_t R>
inline bool cholesky_factor_solve(FixedMatrix<N, N>& A, std::array<std::array<double, R>, N>& B,
                                  FixedVector<N>& col, FixedVector<N>& inv_diag) noexcept {
  for (std::size_t k = 0; k < N; ++k) {
    const double pivot = A(k, k);
    if (!(pivot > 0.0)) {
      return false;
    }
    const double lkk = std::sqrt(pivot);
    const double inv = 1.0 / lkk;
    A(k, k) = lkk;
    inv_diag[k] = inv;
    std::array<double, R>& yk = B[k];
    for (std::size_t r = 0; r < R; ++r) {
      yk[r] *= inv;
    }
    for (std::size_t i = k + 1; i < N; ++i) {
      const double lik = A(i, k) * inv;
      A(i, k) = lik;
      col[i] = lik;
      std::array<double, R>& bi = B[i];
      for (std::size_t r = 0; r < R; ++r) {
        bi[r] -= lik * yk[r];
      }
    }
    for (std::size_t i = k + 1; i < N; ++i) {
      const double lik = col[i];
      double* row = A.data.data() + i * A.stride;
      for (std::size_t j = k + 1; j <= i; ++j) {
        row[j] -= lik * col[j];
      }
    }
  }
  for (std::size_t kk = N; kk-- > 0;) {
    std::array<double, R>& xk = B[kk];
    const double inv = inv_diag[kk];
    for (std::size_t r = 0; r < R; ++r) {
      xk[r] *= inv;
    }
    const double* lrow = A.data.data() + kk * A.stride;
    for (std::size_t i = 0; i < kk; ++i) {
      const double l = lrow[i];
      std::array<double, R>& bi = B[i];
      for (std::size_t r = 0; r < R; ++r) {
        bi[r] -= l * xk[r];
      }
    }
  }
  return true;
}

// Solves (L L^T) X = B in place for a block of right-hand sides stored as the rows of B, i.e. it
// overwrites row r of B (length N) with the solution of M x = B_r^T. Columns are processed
// together, so the inner loops run across the R right-hand sides of one coordinate (contiguous in
// the transposed scratch).
template <std::size_t N, std::size_t R>
inline void cholesky_solve_rows(const FixedMatrix<N, N>& Lmat, FixedMatrix<R, N>& B,
                                FixedMatrix<N, R>& scratch) noexcept {
  for (std::size_t p = 0; p < N; ++p) {
    for (std::size_t r = 0; r < R; ++r) {
      scratch(p, r) = B(r, p);
    }
  }
  constexpr std::size_t kLanes = FixedMatrix<N, R>::stride;
  // Forward substitution: L Y = B^T.
  for (std::size_t k = 0; k < N; ++k) {
    double* yk = scratch.data.data() + k * scratch.stride;
    scale_lanes<kLanes>(yk, 1.0 / Lmat(k, k));
    for (std::size_t i = k + 1; i < N; ++i) {
      axpy_lanes<kLanes>(scratch.data.data() + i * scratch.stride, -Lmat(i, k), yk);
    }
  }
  // Back substitution: L^T X = Y.
  for (std::size_t kk = N; kk-- > 0;) {
    double* xk = scratch.data.data() + kk * scratch.stride;
    scale_lanes<kLanes>(xk, 1.0 / Lmat(kk, kk));
    for (std::size_t i = 0; i < kk; ++i) {
      axpy_lanes<kLanes>(scratch.data.data() + i * scratch.stride, -Lmat(kk, i), xk);
    }
  }
  for (std::size_t p = 0; p < N; ++p) {
    for (std::size_t r = 0; r < R; ++r) {
      B(r, p) = scratch(p, r);
    }
  }
}

}  // namespace pulselob::core
