#include "pulselob/core/model_shape.hpp"

#include <gtest/gtest.h>

namespace {

using pulselob::core::DefaultShape;
using pulselob::core::ModelShape;

// Dimensions quoted in math_spec Sections 3.1 and 5.6.
static_assert(DefaultShape::D == 4);
static_assert(DefaultShape::Q == 4);
static_assert(DefaultShape::P == 17);
static_assert(DefaultShape::kStatesPerChannel == 6);
static_assert(DefaultShape::kTimeBasis == 7);  // M + 1
static_assert(DefaultShape::r_max == 2);

using AllExp = ModelShape<4, 1, 1, 1>;  // the Q = 3 all-Erlang-1 fallback bank of math_spec 5.6
static_assert(AllExp::P == 13);
static_assert(AllExp::kTimeBasis == 4);

TEST(ModelShape, CoordinatesEnumerateRegressorInOrder) {
  std::size_t expected = 1;
  for (std::size_t j = 0; j < DefaultShape::D; ++j) {
    for (std::size_t q = 0; q < DefaultShape::Q; ++q) {
      EXPECT_EQ(DefaultShape::coord(j, q), expected++);
    }
  }
  EXPECT_EQ(expected, DefaultShape::P);
}

}  // namespace
