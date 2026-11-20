#pragma once

// Low-overhead timestamps and an allocation-free latency histogram for the latency report
// (docs/architecture.md Section 9.2).

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <thread>

#if defined(__x86_64__)
#include <x86intrin.h>
#endif

namespace pulselob::bench {

// Reads the platform's constant-rate counter: CNTVCT_EL0 on AArch64 (ordered by an ISB), the TSC
// on x86-64 (RDTSCP), steady_clock elsewhere.
struct CycleClock {
  static std::uint64_t now() noexcept {
#if defined(__aarch64__)
    std::uint64_t v;
    __asm__ __volatile__("isb\n\tmrs %0, cntvct_el0" : "=r"(v)::"memory");
    return v;
#elif defined(__x86_64__)
    unsigned aux;
    return __rdtscp(&aux);
#else
    return static_cast<std::uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
#endif
  }

  // Counter frequency in Hz (read from CNTFRQ_EL0 on AArch64, calibrated against steady_clock on x86).
  static double frequency() {
#if defined(__aarch64__)
    std::uint64_t f;
    __asm__ __volatile__("mrs %0, cntfrq_el0" : "=r"(f));
    return static_cast<double>(f);
#else
    const auto t0 = std::chrono::steady_clock::now();
    const std::uint64_t c0 = now();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    const auto t1 = std::chrono::steady_clock::now();
    const std::uint64_t c1 = now();
    return static_cast<double>(c1 - c0) / std::chrono::duration<double>(t1 - t0).count();
#endif
  }
};

// Log-linear histogram of nanosecond values: 32 linear sub-buckets per power of two from 1 ns to
// ~1 s, fixed size, no allocation. Relative bucket width <= 1/32.
class LatencyHistogram {
 public:
  static constexpr int kSub = 32;
  static constexpr int kOctaves = 31;

  void add(double ns) noexcept {
    ++count_;
    sum_ += ns;
    max_ = std::max(max_, ns);
    min_ = std::min(min_, ns);
    ++buckets_[index(ns)];
  }

  std::uint64_t count() const noexcept { return count_; }
  double mean() const noexcept { return count_ ? sum_ / static_cast<double>(count_) : 0.0; }
  double max() const noexcept { return max_; }
  double min() const noexcept { return min_; }

  // Upper edge of the bucket holding the q-quantile.
  double quantile(double q) const noexcept {
    const auto target = static_cast<std::uint64_t>(std::ceil(q * static_cast<double>(count_)));
    std::uint64_t acc = 0;
    for (std::size_t i = 0; i < buckets_.size(); ++i) {
      acc += buckets_[i];
      if (acc >= target && acc > 0) {
        return upper(i);
      }
    }
    return max_;
  }

 private:
  static std::size_t index(double ns) noexcept {
    if (ns < 1.0) {
      return 0;
    }
    int e = 0;
    const double m = std::frexp(ns, &e);  // ns = m * 2^e, m in [0.5, 1)
    const int octave = std::min(e - 1, kOctaves - 1);
    const int sub = std::min(static_cast<int>((m * 2.0 - 1.0) * kSub), kSub - 1);
    return static_cast<std::size_t>(1 + octave * kSub + sub);
  }
  static double upper(std::size_t i) noexcept {
    if (i == 0) {
      return 1.0;
    }
    const auto k = static_cast<int>(i - 1);
    const int octave = k / kSub;
    const int sub = k % kSub;
    return std::ldexp(1.0 + static_cast<double>(sub + 1) / kSub, octave);
  }

  std::array<std::uint64_t, 1 + kSub * kOctaves> buckets_{};
  std::uint64_t count_ = 0;
  double sum_ = 0.0;
  double max_ = 0.0;
  double min_ = 1e300;
};

}  // namespace pulselob::bench
