#include "pulselob/version.hpp"

#include <gtest/gtest.h>

#include <string>

namespace {

TEST(Version, StringMatchesNumericComponents) {
  const std::string expected = std::to_string(pulselob::kVersionMajor) + "." +
                               std::to_string(pulselob::kVersionMinor) + "." +
                               std::to_string(pulselob::kVersionPatch);
  const std::string actual = pulselob::kVersionString;
  ASSERT_GE(actual.size(), expected.size());
  EXPECT_EQ(actual.substr(0, expected.size()), expected);
}

}  // namespace
