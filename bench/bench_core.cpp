// Baseline micro-benchmarks for the Phase 3 cost model (docs/architecture.md Section 9.3).
// They measure primitives whose cost the engine pays per event; engine benchmarks come in Phase 3.
#include <benchmark/benchmark.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "pulselob/feed/market_event.hpp"
#include "pulselob/feed/spsc_ring.hpp"

namespace {

// std::exp over the argument range the engine evaluates: -beta*Delta in [-50, 0]
// (math_spec 4.6 cuts off at beta*Delta = 50). The engine needs Q + 1 of these per event.
void BM_StdExp(benchmark::State& state) {
  std::mt19937_64 rng(42);
  std::uniform_real_distribution<double> log_arg(-13.8, std::log(50.0));  // 1e-6 .. 50
  std::vector<double> args(4096);
  for (double& a : args) {
    a = -std::exp(log_arg(rng));
  }
  std::size_t i = 0;
  for (auto _ : state) {
    benchmark::DoNotOptimize(std::exp(args[i]));
    i = (i + 1) & (args.size() - 1);
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_StdExp);

// Single-thread push+pop of one 64-byte MarketEvent: the ring's own overhead, excluding
// cross-core cache-line transfer (measured with pinned threads in Phase 3).
void BM_SpscPushPopSameThread(benchmark::State& state) {
  using Ring = pulselob::feed::SpscRing<pulselob::feed::MarketEvent, 4096>;
  auto ring = std::make_unique<Ring>();
  pulselob::feed::MarketEvent in{};
  pulselob::feed::MarketEvent out{};
  for (auto _ : state) {
    in.seq += 1;
    benchmark::DoNotOptimize(ring->try_push(in));
    benchmark::DoNotOptimize(ring->try_pop(out));
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_SpscPushPopSameThread);

void BM_ValidateMarketEvent(benchmark::State& state) {
  std::vector<pulselob::feed::MarketEvent> events(4096);
  for (std::size_t k = 0; k < events.size(); ++k) {
    auto& e = events[k];
    e.kind = pulselob::feed::EventKind::Trade;
    e.flags =
        pulselob::feed::event_flags::make(k % 2 ? pulselob::feed::Side::Buy : pulselob::feed::Side::Sell,
                                          pulselob::feed::TsResolution::Millisecond);
    e.px0_e8 = e.px1_e8 = 6'900'000'000'000;
    e.qty0_e8 = 100'000;
    e.count = 1;
  }
  std::size_t i = 0;
  for (auto _ : state) {
    benchmark::DoNotOptimize(pulselob::feed::validate(events[i]));
    i = (i + 1) & (events.size() - 1);
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_ValidateMarketEvent);

}  // namespace
