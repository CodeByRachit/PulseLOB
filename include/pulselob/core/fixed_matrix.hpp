#pragma once

#include <array>
#include <cstddef>
#include <span>

#include "pulselob/core/platform.hpp"

namespace pulselob::core {

// Fixed-size numeric containers for the engine state (docs/architecture.md Section 5).
//
// Layout contract:
//  * storage is 64-byte aligned and every row starts on a 64-byte boundary;
//  * rows are padded to a multiple of 8 doubles, so kernels can run full-width SIMD loops over
//    `stride` lanes without remainder handling;
//  * padding lanes are zero at construction and every kernel must keep them zero, which makes
//    full-width loops over the padding numerically inert. padding_is_zero() checks this in tests.
//  * no member ever allocates: the size is fixed at compile time.

template <std::size_t N>
struct alignas(kCacheLineBytes) FixedVector {
  static_assert(N >= 1, "FixedVector must be non-empty");
  static constexpr std::size_t extent = N;
  static constexpr std::size_t padded = round_up(N, kDoublesPerLine);

  std::array<double, padded> data{};

  constexpr double& operator[](std::size_t i) noexcept { return data[i]; }
  constexpr const double& operator[](std::size_t i) const noexcept { return data[i]; }

  constexpr std::span<double, N> view() noexcept { return std::span<double, N>(data.data(), N); }
  constexpr std::span<const double, N> view() const noexcept {
    return std::span<const double, N>(data.data(), N);
  }
  // All lanes including the zero padding, for full-width SIMD loops.
  constexpr std::span<double, padded> lanes() noexcept { return std::span<double, padded>(data); }
  constexpr std::span<const double, padded> lanes() const noexcept {
    return std::span<const double, padded>(data);
  }

  constexpr bool padding_is_zero() const noexcept {
    for (std::size_t i = N; i < padded; ++i) {
      if (data[i] != 0.0) {
        return false;
      }
    }
    return true;
  }
};

template <std::size_t Rows, std::size_t Cols>
struct alignas(kCacheLineBytes) FixedMatrix {
  static_assert(Rows >= 1 && Cols >= 1, "FixedMatrix must be non-empty");
  static constexpr std::size_t rows = Rows;
  static constexpr std::size_t cols = Cols;
  static constexpr std::size_t stride = round_up(Cols, kDoublesPerLine);

  std::array<double, Rows * stride> data{};

  constexpr double& operator()(std::size_t r, std::size_t c) noexcept { return data[r * stride + c]; }
  constexpr const double& operator()(std::size_t r, std::size_t c) const noexcept {
    return data[r * stride + c];
  }

  constexpr std::span<double, Cols> row(std::size_t r) noexcept {
    return std::span<double, Cols>(data.data() + r * stride, Cols);
  }
  constexpr std::span<const double, Cols> row(std::size_t r) const noexcept {
    return std::span<const double, Cols>(data.data() + r * stride, Cols);
  }
  // A full padded row, for full-width SIMD loops.
  constexpr std::span<double, stride> row_lanes(std::size_t r) noexcept {
    return std::span<double, stride>(data.data() + r * stride, stride);
  }
  constexpr std::span<const double, stride> row_lanes(std::size_t r) const noexcept {
    return std::span<const double, stride>(data.data() + r * stride, stride);
  }

  constexpr bool padding_is_zero() const noexcept {
    for (std::size_t r = 0; r < Rows; ++r) {
      for (std::size_t c = Cols; c < stride; ++c) {
        if (data[r * stride + c] != 0.0) {
          return false;
        }
      }
    }
    return true;
  }
};

}  // namespace pulselob::core
