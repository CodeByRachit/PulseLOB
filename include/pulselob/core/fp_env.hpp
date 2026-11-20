#pragma once

#include <cstdint>

#if defined(__x86_64__) || defined(_M_X64)
#include <xmmintrin.h>
#endif

namespace pulselob::core {

// RAII guard that enables flush-to-zero (and denormals-are-zero on x86) on the calling thread and
// restores the previous floating-point control state on destruction (docs/architecture.md
// Section 4.3). The cutoff of math_spec Section 4.6 keeps denormals out of every meaningful
// computation, so results are expected to be identical with and without the guard; the guard only
// removes the denormal slow path on x86.
class FlushToZeroGuard {
 public:
  FlushToZeroGuard() noexcept : saved_(read()) { write(saved_ | kFlushBits); }
  ~FlushToZeroGuard() { write(saved_); }
  FlushToZeroGuard(const FlushToZeroGuard&) = delete;
  FlushToZeroGuard& operator=(const FlushToZeroGuard&) = delete;

  // True if flush-to-zero is currently enabled on this thread (false on unsupported targets).
  static bool enabled() noexcept { return kFlushBits != 0 && (read() & kFlushBits) == kFlushBits; }

 private:
#if defined(__x86_64__) || defined(_M_X64)
  using Word = unsigned int;
  static constexpr Word kFlushBits = 0x8040U;  // MXCSR.FTZ (bit 15) | MXCSR.DAZ (bit 6)
  static Word read() noexcept { return _mm_getcsr(); }
  static void write(Word w) noexcept { _mm_setcsr(w); }
#elif defined(__aarch64__)
  using Word = std::uint64_t;
  static constexpr Word kFlushBits = Word{1} << 24;  // FPCR.FZ
  static Word read() noexcept {
    Word w;
    __asm__ __volatile__("mrs %0, fpcr" : "=r"(w));
    return w;
  }
  static void write(Word w) noexcept { __asm__ __volatile__("msr fpcr, %0" : : "r"(w)); }
#else
  using Word = unsigned int;
  static constexpr Word kFlushBits = 0;
  static Word read() noexcept { return 0; }
  static void write(Word) noexcept {}
#endif
  Word saved_;
};

}  // namespace pulselob::core
