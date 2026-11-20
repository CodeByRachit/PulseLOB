#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>

#include "pulselob/core/platform.hpp"

namespace pulselob::feed {

// Wait-free single-producer / single-consumer ring buffer (docs/architecture.md Section 6.2).
//
//  * Capacity is a compile-time power of two; storage is inline, so a large ring must be created
//    once at start-up on the heap (std::make_unique honours the over-alignment). No operation
//    allocates.
//  * head_ and tail_ are monotonically increasing 64-bit counters (they cannot wrap in practice:
//    2^64 events at 1e9 events/s is 584 years); the slot index is counter & (Capacity - 1).
//  * Each side keeps a private cached copy of the other side's counter, so in steady state a push
//    or pop touches no cache line written by the other core.
//  * Every field written by a different thread sits on its own 128-byte block (no false sharing).
//
// Memory ordering: the producer writes the slot, then publishes head_ with release; the consumer
// reads head_ with acquire before reading the slot, so the slot write happens-before the read.
// Symmetrically the consumer releases tail_ after reading, and the producer acquires tail_ before
// overwriting, so a slot is never overwritten while it is being read.
template <class T, std::size_t Capacity>
class SpscRing {
  static_assert(Capacity >= 2 && (Capacity & (Capacity - 1)) == 0, "Capacity must be a power of two");
  static_assert(std::is_trivially_copyable_v<T>, "slots are copied by value");
  static_assert(std::atomic<std::uint64_t>::is_always_lock_free);
  static_assert(sizeof(std::size_t) == sizeof(std::uint64_t), "counter differences convert losslessly");

 public:
  using value_type = T;
  static constexpr std::size_t capacity = Capacity;

  SpscRing() noexcept = default;
  SpscRing(const SpscRing&) = delete;
  SpscRing& operator=(const SpscRing&) = delete;

  // Producer thread only. Returns false (and leaves the ring unchanged) if it is full.
  bool try_push(const T& value) noexcept {
    const std::uint64_t head = head_.load(std::memory_order_relaxed);
    if (head - producer_cached_tail_ == Capacity) {
      producer_cached_tail_ = tail_.load(std::memory_order_acquire);
      if (head - producer_cached_tail_ == Capacity) {
        return false;
      }
    }
    slots_[head & kMask] = value;
    head_.store(head + 1, std::memory_order_release);
    return true;
  }

  // Consumer thread only. Returns false if the ring is empty.
  bool try_pop(T& out) noexcept {
    const std::uint64_t tail = tail_.load(std::memory_order_relaxed);
    if (tail == consumer_cached_head_) {
      consumer_cached_head_ = head_.load(std::memory_order_acquire);
      if (tail == consumer_cached_head_) {
        return false;
      }
    }
    out = slots_[tail & kMask];
    tail_.store(tail + 1, std::memory_order_release);
    return true;
  }

  // Consumer thread only. Pops up to out.size() items with a single release store; returns the
  // number popped.
  std::size_t pop_batch(std::span<T> out) noexcept {
    const std::uint64_t tail = tail_.load(std::memory_order_relaxed);
    if (tail == consumer_cached_head_) {
      consumer_cached_head_ = head_.load(std::memory_order_acquire);
    }
    const std::size_t available = consumer_cached_head_ - tail;
    const std::size_t n = std::min(available, out.size());
    for (std::size_t i = 0; i < n; ++i) {
      out[i] = slots_[(tail + i) & kMask];
    }
    if (n != 0) {
      tail_.store(tail + n, std::memory_order_release);
    }
    return n;
  }

  // Approximate when called concurrently; exact when both sides are quiescent.
  std::size_t size_approx() const noexcept {
    const std::uint64_t head = head_.load(std::memory_order_acquire);
    const std::uint64_t tail = tail_.load(std::memory_order_acquire);
    return head - tail;
  }

 private:
  static constexpr std::uint64_t kMask = Capacity - 1;

  alignas(core::kFalseSharingBytes) std::atomic<std::uint64_t> head_{0};  // written by producer
  alignas(core::kFalseSharingBytes) std::uint64_t producer_cached_tail_{0};
  alignas(core::kFalseSharingBytes) std::atomic<std::uint64_t> tail_{0};  // written by consumer
  alignas(core::kFalseSharingBytes) std::uint64_t consumer_cached_head_{0};
  alignas(core::kFalseSharingBytes) std::array<T, Capacity> slots_{};
};

}  // namespace pulselob::feed
