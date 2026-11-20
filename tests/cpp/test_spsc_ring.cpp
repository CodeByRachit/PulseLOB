#include "pulselob/feed/spsc_ring.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <memory>
#include <thread>

#include "pulselob/core/platform.hpp"
#include "pulselob/feed/market_event.hpp"

namespace {

using pulselob::feed::MarketEvent;
using pulselob::feed::SpscRing;

TEST(SpscRing, FillDrainAndWrapAround) {
  SpscRing<std::uint64_t, 8> ring;
  std::uint64_t out = 0;
  EXPECT_FALSE(ring.try_pop(out));
  for (std::uint64_t round = 0; round < 5; ++round) {  // counters pass the capacity several times
    for (std::uint64_t i = 0; i < 8; ++i) {
      EXPECT_TRUE(ring.try_push(round * 100 + i));
    }
    EXPECT_FALSE(ring.try_push(999)) << "ring must report full at capacity";
    EXPECT_EQ(ring.size_approx(), 8u);
    for (std::uint64_t i = 0; i < 8; ++i) {
      ASSERT_TRUE(ring.try_pop(out));
      EXPECT_EQ(out, round * 100 + i);
    }
    EXPECT_FALSE(ring.try_pop(out));
  }
}

TEST(SpscRing, BatchPopAcrossTheWrapPoint) {
  SpscRing<std::uint64_t, 8> ring;
  std::uint64_t sink = 0;
  for (std::uint64_t i = 0; i < 6; ++i) {
    ASSERT_TRUE(ring.try_push(i));
  }
  for (int i = 0; i < 6; ++i) {
    ASSERT_TRUE(ring.try_pop(sink));
  }
  for (std::uint64_t i = 0; i < 7; ++i) {  // slots 6,7,0,1,2,3,4
    ASSERT_TRUE(ring.try_push(10 + i));
  }
  std::array<std::uint64_t, 16> out{};
  ASSERT_EQ(ring.pop_batch(std::span<std::uint64_t>(out.data(), 4)), 4u);
  ASSERT_EQ(ring.pop_batch(out), 3u);
  EXPECT_EQ(out[0], 14u);
  EXPECT_EQ(out[2], 16u);
  EXPECT_EQ(ring.pop_batch(out), 0u);
}

TEST(SpscRing, FieldsWrittenByDifferentThreadsDoNotShareCacheLines) {
  using Ring = SpscRing<MarketEvent, 1024>;
  static_assert(alignof(Ring) >= pulselob::core::kFalseSharingBytes);
  auto ring = std::make_unique<Ring>();
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(ring.get()) % pulselob::core::kFalseSharingBytes, 0u);
}

// Two-thread stress test; run under ThreadSanitizer in CI (tsan preset).
TEST(SpscRing, ConcurrentProducerConsumerPreservesOrderWithoutLoss) {
  constexpr std::uint64_t kItems = 2'000'000;
  auto ring = std::make_unique<SpscRing<MarketEvent, 1024>>();
  std::thread producer([&] {
    MarketEvent e{};
    for (std::uint64_t i = 0; i < kItems; ++i) {
      e.seq = i;
      e.ts_exchange_ns = static_cast<std::int64_t>(i);
      while (!ring->try_push(e)) {
        std::this_thread::yield();
      }
    }
  });
  std::uint64_t expected = 0;
  std::array<MarketEvent, 64> batch{};
  while (expected < kItems) {
    const std::size_t n = ring->pop_batch(batch);
    if (n == 0) {
      std::this_thread::yield();
      continue;
    }
    for (std::size_t i = 0; i < n; ++i) {
      ASSERT_EQ(batch[i].seq, expected);
      ASSERT_EQ(batch[i].ts_exchange_ns, static_cast<std::int64_t>(expected));
      ++expected;
    }
  }
  producer.join();
  EXPECT_EQ(ring->size_approx(), 0u);
}

}  // namespace
