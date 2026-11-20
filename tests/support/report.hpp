#pragma once

// Formatting of recorded test properties (gtest --gtest_output=json): errors in scientific
// notation, counts as integers.

#include <cstdio>
#include <string>
#include <type_traits>

namespace pulselob::testkit {

template <class T>
std::string sci(T v) {
  char buf[64];
  if constexpr (std::is_floating_point_v<T>) {
    const double d = v;
    std::snprintf(buf, sizeof(buf), "%.3e", d);
  } else {
    std::snprintf(buf, sizeof(buf), "%llu", static_cast<unsigned long long>(v));
  }
  return buf;
}

}  // namespace pulselob::testkit
