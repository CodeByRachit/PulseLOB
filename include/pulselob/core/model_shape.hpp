#pragma once

#include <algorithm>
#include <array>
#include <cstddef>

namespace pulselob::core {

// Compile-time shape of the model: D event channels and a kernel bank of Q Erlang kernels with
// orders r_q (math_spec Section 3.1). Decay rates beta_q are runtime configuration; only the orders
// are compile-time, because they fix the size of every state array.
template <std::size_t D_, std::size_t... Orders>
struct ModelShape {
  static_assert(D_ >= 1 && D_ <= 64, "D must be in [1, 64]");
  static_assert(sizeof...(Orders) >= 1, "the kernel bank needs at least one basis");
  static_assert(((Orders == 1 || Orders == 2) && ...),
                "Erlang orders above 2 need psi_n with n > 3, outside the validated range "
                "(math_spec Section 4.2)");

  static constexpr std::size_t D = D_;                 // channels
  static constexpr std::size_t Q = sizeof...(Orders);  // kernel bases
  static constexpr std::array<std::size_t, Q> orders{Orders...};
  static constexpr std::size_t P = 1 + D * Q;                       // parameters per target channel
  static constexpr std::size_t kStatesPerChannel = (Orders + ...);  // sum of r_q
  static constexpr std::size_t kTimeBasis = 1 + kStatesPerChannel;  // M + 1
  static constexpr std::size_t r_max = std::max({Orders...});

  // Index of the regressor coordinate for source channel j and basis q (coordinate 0 is the
  // baseline): z = (1, x_1^1, ..., x_1^Q, ..., x_D^Q).
  static constexpr std::size_t coord(std::size_t j, std::size_t q) noexcept { return 1 + j * Q + q; }
};

// Default configuration: 4 channels and the bank of math_spec Section 3.1:
// Erlang-1 (1 ms), Erlang-2 (5 ms mode), Erlang-2 (20 ms mode), Erlang-1 (1 s).
using DefaultShape = ModelShape<4, 1, 2, 2, 1>;

}  // namespace pulselob::core
