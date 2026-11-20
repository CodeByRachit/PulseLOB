// Zero-allocation proof (docs/architecture.md Section 5.4). This binary replaces every global
// operator new / delete with counting versions; the hot path must not allocate after construction.
//
// Exception: under ThreadSanitizer on Linux the sanitizer runtime links its own operator new /
// delete statically, so they cannot be replaced. There the count comes from the sanitizer
// allocator's malloc hook instead, which sees every allocation (operator new included).

#include <gtest/gtest.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <new>
#include <vector>

#include "hawkes_sim.hpp"
#include "pulselob/engine/engine.hpp"
#include "pulselob/engine/split_engine.hpp"
#include "report.hpp"
#include "test_config.hpp"

#if defined(__linux__) && defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define PULSELOB_COUNT_VIA_SANITIZER_HOOK 1
#endif
#endif
#if defined(__linux__) && defined(__SANITIZE_THREAD__)
#define PULSELOB_COUNT_VIA_SANITIZER_HOOK 1
#endif

#if defined(PULSELOB_COUNT_VIA_SANITIZER_HOOK)
#include <sanitizer/allocator_interface.h>
#endif

namespace {
std::atomic<bool> g_armed{false};
std::atomic<std::uint64_t> g_allocs{0};
std::atomic<std::uint64_t> g_bytes{0};
}  // namespace

#if defined(PULSELOB_COUNT_VIA_SANITIZER_HOOK)
extern "C" {
static void count_malloc(const volatile void*, std::size_t n) {
  if (g_armed.load(std::memory_order_relaxed)) {
    g_allocs.fetch_add(1, std::memory_order_relaxed);
    g_bytes.fetch_add(n, std::memory_order_relaxed);
  }
}
static void count_free(const volatile void*) {}
}
namespace {
const int g_hooks_installed = __sanitizer_install_malloc_and_free_hooks(count_malloc, count_free);
}  // namespace
#else
namespace {

void* counted_alloc(std::size_t n, std::size_t align) {
  if (g_armed.load(std::memory_order_relaxed)) {
    g_allocs.fetch_add(1, std::memory_order_relaxed);
    g_bytes.fetch_add(n, std::memory_order_relaxed);
  }
  void* p = nullptr;
  if (align <= alignof(std::max_align_t)) {
    p = std::malloc(n == 0 ? 1 : n);
  } else {
    // aligned_alloc needs the size to be a multiple of the alignment.
    p = std::aligned_alloc(align, (n + align - 1) / align * align);
  }
  if (p == nullptr) {
    throw std::bad_alloc();
  }
  return p;
}
}  // namespace

void* operator new(std::size_t n) {
  return counted_alloc(n, alignof(std::max_align_t));
}
void* operator new[](std::size_t n) {
  return counted_alloc(n, alignof(std::max_align_t));
}
void* operator new(std::size_t n, std::align_val_t a) {
  return counted_alloc(n, static_cast<std::size_t>(a));
}
void* operator new[](std::size_t n, std::align_val_t a) {
  return counted_alloc(n, static_cast<std::size_t>(a));
}
void* operator new(std::size_t n, const std::nothrow_t&) noexcept {
  try {
    return counted_alloc(n, alignof(std::max_align_t));
  } catch (...) {
    return nullptr;
  }
}
void* operator new[](std::size_t n, const std::nothrow_t&) noexcept {
  try {
    return counted_alloc(n, alignof(std::max_align_t));
  } catch (...) {
    return nullptr;
  }
}
void operator delete(void* p) noexcept {
  std::free(p);
}
void operator delete[](void* p) noexcept {
  std::free(p);
}
void operator delete(void* p, std::size_t) noexcept {
  std::free(p);
}
void operator delete[](void* p, std::size_t) noexcept {
  std::free(p);
}
void operator delete(void* p, std::align_val_t) noexcept {
  std::free(p);
}
void operator delete[](void* p, std::align_val_t) noexcept {
  std::free(p);
}
void operator delete(void* p, std::size_t, std::align_val_t) noexcept {
  std::free(p);
}
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept {
  std::free(p);
}
#endif  // PULSELOB_COUNT_VIA_SANITIZER_HOOK

namespace {

using namespace pulselob;
using testkit::Shape;

std::size_t event_count() {
#if defined(NDEBUG)
  return 1'000'000;
#else
  return 200'000;
#endif
}

std::vector<feed::MarketEvent> make_stream(std::size_t n) {
  testkit::Rng rng(131);
  testkit::HawkesSimulator<Shape> sim(testkit::default_beta(), testkit::ground_truth_theta(0.3));
  std::vector<feed::MarketEvent> out;
  std::int64_t offset = 0;
  while (out.size() < n) {
    auto chunk = testkit::to_market_events(sim.simulate(60.0, rng), rng);
    for (auto& e : chunk) {
      e.ts_local_ns += offset;
      e.ts_exchange_ns += offset;
    }
    offset = chunk.back().ts_local_ns - chunk.front().ts_local_ns + offset + 1'000'000;
    out.insert(out.end(), chunk.begin(), chunk.end());
  }
  out.resize(n);
  return out;
}

struct NullSink {
  std::uint64_t n = 0;
  void on_signal(const engine::Signal<Shape::D>& s) noexcept { n += s.bid_px_e8 != 0 ? 1U : 0U; }
};

TEST(ZeroAllocation, CounterWorks) {
#if defined(PULSELOB_COUNT_VIA_SANITIZER_HOOK)
  ASSERT_NE(g_hooks_installed, 0) << "sanitizer malloc hooks not installed";
#endif
  g_allocs = 0;
  g_armed = true;
  auto v = std::make_unique<std::vector<int>>(10);
  g_armed = false;
  EXPECT_GE(g_allocs.load(), 1U);
}

template <class Est>
void run_inline() {
  const auto ev = make_stream(event_count());
  auto eng = std::make_unique<engine::Engine<Shape, Est>>(testkit::default_engine_config());
  NullSink sink;
  g_allocs = 0;
  g_bytes = 0;
  g_armed = true;
  engine::replay(*eng, std::span<const feed::MarketEvent>(ev), sink);
  for (int k = 0; k < 1000; ++k) {
    eng->on_heartbeat(eng->clock() + 50'000'000);
  }
  g_armed = false;
  EXPECT_EQ(g_allocs.load(), 0U) << g_bytes.load() << " bytes allocated on the hot path";
  ::testing::Test::RecordProperty("events", testkit::sci(ev.size()));
  EXPECT_GT(eng->counters().trades, ev.size() / 3);
}

TEST(ZeroAllocation, InlineEngineCholesky) {
  run_inline<hawkes::CholeskyEstimator<Shape>>();
}
TEST(ZeroAllocation, InlineEnginePotter) {
  run_inline<hawkes::PotterEstimator<Shape>>();
}

TEST(ZeroAllocation, SplitEngineBothThreads) {
  const auto ev = make_stream(event_count() / 4);
  auto eng = std::make_unique<engine::SplitEngine<Shape>>(testkit::default_engine_config());
  g_allocs = 0;
  g_armed = true;  // counts allocations on the hot thread and the estimator thread
  for (const auto& e : ev) {
    eng->on_event(e);
  }
  eng->drain();
  g_armed = false;
  EXPECT_EQ(g_allocs.load(), 0U);
  eng->stop();
}

}  // namespace
