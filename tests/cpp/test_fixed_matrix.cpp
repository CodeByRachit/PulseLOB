#include "pulselob/core/fixed_matrix.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>

#include "pulselob/core/model_shape.hpp"

namespace {

using pulselob::core::DefaultShape;
using pulselob::core::FixedMatrix;
using pulselob::core::FixedVector;

constexpr std::size_t P = DefaultShape::P;  // 17

// Layout contract from docs/architecture.md Section 5.3.
static_assert(FixedVector<17>::padded == 24);
static_assert(sizeof(FixedVector<17>) == 24 * 8);
static_assert(FixedMatrix<P, P>::stride == 24);
static_assert(sizeof(FixedMatrix<P, P>) == 17 * 24 * 8);  // 3264 B = 51 cache lines
static_assert(sizeof(FixedMatrix<4, P>) == 4 * 24 * 8);
static_assert(alignof(FixedMatrix<P, P>) == 64);
static_assert(sizeof(FixedMatrix<3, 8>) == 3 * 64);  // already a multiple of 8: no padding

TEST(FixedMatrix, RowsStartOnCacheLines) {
  auto m = std::make_unique<FixedMatrix<P, P>>();
  for (std::size_t r = 0; r < P; ++r) {
    const auto addr = reinterpret_cast<std::uintptr_t>(m->row(r).data());
    EXPECT_EQ(addr % 64, 0u) << "row " << r;
  }
}

TEST(FixedMatrix, ZeroInitializedIncludingPadding) {
  auto m = std::make_unique<FixedMatrix<P, P>>();
  for (double v : m->data) {
    EXPECT_EQ(v, 0.0);
  }
  EXPECT_TRUE(m->padding_is_zero());
}

TEST(FixedMatrix, RowViewsAndPaddingCheck) {
  FixedMatrix<3, 5> m;
  m(1, 4) = 2.5;
  EXPECT_EQ(m.row(1)[4], 2.5);
  EXPECT_EQ(m.row_lanes(1).size(), 8u);
  EXPECT_TRUE(m.padding_is_zero());
  m.row_lanes(2)[7] = 1.0;  // write into padding
  EXPECT_FALSE(m.padding_is_zero());
}

TEST(FixedVector, ViewsAndPadding) {
  FixedVector<17> v;
  v[16] = 1.0;
  EXPECT_EQ(v.view().size(), 17u);
  EXPECT_EQ(v.lanes().size(), 24u);
  EXPECT_TRUE(v.padding_is_zero());
  v.lanes()[20] = 3.0;
  EXPECT_FALSE(v.padding_is_zero());
}

}  // namespace
