#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

#include "hawkes_sim.hpp"
#include "pulselob/core/seqlock.hpp"
#include "pulselob/engine/engine.hpp"
#include "pulselob/engine/split_engine.hpp"
#include "report.hpp"
#include "state_compare.hpp"
#include "test_config.hpp"

namespace {

using namespace pulselob;
using testkit::kD;
using testkit::Shape;

struct Payload {
  std::array<std::uint64_t, 150> words;  // ~1.2 KB, the size of the served parameters
};

// Every snapshot a reader obtains is internally consistent (all words from one write), and versions
// are monotone. Runs under ThreadSanitizer in CI: every payload access is atomic, so no race.
TEST(Seqlock, ReadersNeverSeeTornSnapshots) {
  auto lock = std::make_unique<core::Seqlock<Payload>>();
  std::atomic<bool> done{false};
  constexpr std::uint64_t kWrites = 200'000;
  std::thread writer([&] {
    auto p = std::make_unique<Payload>();
    for (std::uint64_t k = 1; k <= kWrites; ++k) {
      p->words.fill(k);
      lock->store(*p);
    }
    done.store(true, std::memory_order_release);
  });
  auto snap = std::make_unique<Payload>();
  std::uint64_t reads = 0;
  std::uint64_t last_version = 0;
  std::uint64_t last_value = 0;
  while (!done.load(std::memory_order_acquire) || reads < 1000) {
    const std::uint64_t v = lock->load(*snap);
    ASSERT_EQ(v % 2, 0U);
    ASSERT_GE(v, last_version);
    const std::uint64_t first = snap->words[0];
    for (std::uint64_t w : snap->words) {
      ASSERT_EQ(w, first) << "torn read";
    }
    ASSERT_GE(first, last_value);
    last_version = v;
    last_value = first;
    ++reads;
  }
  writer.join();
  lock->load(*snap);
  EXPECT_EQ(snap->words[0], kWrites);
}

std::vector<feed::MarketEvent> make_stream(std::uint64_t seed, double horizon_s) {
  testkit::Rng rng(seed);
  testkit::HawkesSimulator<Shape> sim(testkit::default_beta(), testkit::ground_truth_theta(0.3));
  return testkit::to_market_events(sim.simulate(horizon_s, rng), rng);
}

// Split mode: after drain() the estimator thread's served parameters equal the inline engine's bit
// for bit (same input sequence, same code); the hot thread's signals use lagged parameters only.
TEST(SplitEngine, EstimatorThreadMatchesInlineEngineExactly) {
  const auto cfg = testkit::default_engine_config();
  const auto ev = make_stream(113, 30.0);
  auto inline_engine = std::make_unique<engine::Engine<Shape>>(cfg);
  auto split = std::make_unique<engine::SplitEngine<Shape>>(cfg);
  std::size_t stale = 0;
  for (std::size_t k = 0; k < ev.size(); ++k) {
    inline_engine->on_event(ev[k]);
    split->on_event(ev[k]);
    stale += (split->signal().flags & engine::signal_flags::kStaleParameters) != 0 ? 1U : 0U;
    if (k % 5000 == 4999) {
      split->drain();
      EXPECT_TRUE(testkit::same_served(split->served(), inline_engine->model().served()))
          << "after event " << k;
    }
  }
  split->drain();
  EXPECT_TRUE(testkit::same_served(split->served(), inline_engine->model().served()));
  EXPECT_EQ(split->counters().ring_overflows, 0U);
  EXPECT_GT(split->publications(), 0U);
  // Hot-thread kernel state is the same recursion: once the parameters are current, a read-only
  // event (a book update, which does not step the model) gives identical signals.
  feed::MarketEvent bk{};
  for (auto it = ev.rbegin(); it != ev.rend(); ++it) {
    if (it->kind == feed::EventKind::BookTop) {
      bk = *it;
      break;
    }
  }
  bk.ts_local_ns = ev.back().ts_local_ns + 1'000'000;
  split->on_event(bk);
  inline_engine->on_event(bk);
  for (std::size_t i = 0; i < kD; ++i) {
    EXPECT_EQ(split->signal().lambda[i], inline_engine->signal().lambda[i]);
  }
  EXPECT_EQ(split->signal().tau_buy, inline_engine->signal().tau_buy);
  // Quotes are not compared: the throttles and pull triggers carry history computed with lagged
  // parameters, which is exactly the documented difference between the modes.
  split->stop();
  RecordProperty("stale_signals", testkit::sci(stale));
}

}  // namespace
