#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "pulselob/core/platform.hpp"

namespace pulselob::core {

// Single-writer seqlock for publishing a trivially copyable snapshot to a reader thread
// (docs/architecture.md Section 6.3). The payload is copied word by word through
// std::atomic_ref<std::uint64_t> with relaxed ordering, so concurrent reads and writes are never a
// data race (Boehm, 2012); the fences and the sequence number make a read consistent whenever the
// sequence number is even and unchanged across the copy.
//
// Writer: s = seq; seq = s + 1; fence(release); relaxed word stores; seq = s + 2 (release).
// Reader: s1 = seq (acquire); relaxed word loads; fence(acquire); s2 = seq (relaxed); retry if
//         s1 is odd or s1 != s2.
template <class T>
class Seqlock {
  static_assert(std::is_trivially_copyable_v<T>, "seqlock payload must be trivially copyable");
  static constexpr std::size_t kWords = (sizeof(T) + 7) / 8;

 public:
  Seqlock() noexcept = default;
  explicit Seqlock(const T& initial) noexcept { std::memcpy(words_.data(), &initial, sizeof(T)); }

  // Writer side (one thread only).
  void store(const T& value) noexcept {
    std::array<std::uint64_t, kWords> src{};
    std::memcpy(src.data(), &value, sizeof(T));
    const std::uint64_t s = seq_.load(std::memory_order_relaxed);
    seq_.store(s + 1, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);
    for (std::size_t i = 0; i < kWords; ++i) {
      std::atomic_ref<std::uint64_t>(words_[i]).store(src[i], std::memory_order_relaxed);
    }
    seq_.store(s + 2, std::memory_order_release);
  }

  // Current sequence number (even when no write is in progress). One acquire load: the cheap check
  // a reader performs per event to decide whether a new snapshot exists.
  std::uint64_t version() const noexcept { return seq_.load(std::memory_order_acquire); }

  // Reader side. Copies a consistent snapshot into out and returns its (even) sequence number.
  std::uint64_t load(T& out) const noexcept {
    std::array<std::uint64_t, kWords> dst{};
    for (;;) {
      const std::uint64_t s1 = seq_.load(std::memory_order_acquire);
      if ((s1 & 1U) != 0) {
        continue;
      }
      for (std::size_t i = 0; i < kWords; ++i) {
        dst[i] = std::atomic_ref<std::uint64_t>(words_[i]).load(std::memory_order_relaxed);
      }
      std::atomic_thread_fence(std::memory_order_acquire);
      const std::uint64_t s2 = seq_.load(std::memory_order_relaxed);
      if (s1 == s2) {
        // T is trivially copyable (asserted above); the cast tells GCC the raw copy is intended.
        std::memcpy(static_cast<void*>(&out), dst.data(), sizeof(T));
        return s1;
      }
    }
  }

 private:
  alignas(kFalseSharingBytes) std::atomic<std::uint64_t> seq_{0};
  alignas(kFalseSharingBytes) mutable std::array<std::uint64_t, kWords> words_{};
};

}  // namespace pulselob::core
