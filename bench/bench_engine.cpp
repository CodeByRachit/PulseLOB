// Google Benchmark suite for the Phase 3 engine: end-to-end throughput per event, and the
// state-independent kernels. Latency distributions and the in-situ component breakdown are in
// latency_report.cpp (Google Benchmark reports means, not tails).

#include <benchmark/benchmark.h>

#include <memory>
#include <vector>

#include "hawkes_sim.hpp"
#include "pulselob/engine/engine.hpp"
#include "pulselob/engine/split_engine.hpp"
#include "test_config.hpp"

namespace {

using namespace pulselob;
using testkit::Shape;
constexpr std::size_t kD = Shape::D;

const std::vector<feed::MarketEvent>& stream() {
  static const std::vector<feed::MarketEvent> ev = [] {
    testkit::Rng rng(20261120);
    testkit::HawkesSimulator<Shape> sim(testkit::default_beta(), testkit::ground_truth_theta(0.3));
    return testkit::to_market_events(sim.simulate(120.0, rng), rng);
  }();
  return ev;
}

// Full on_event over a realistic mix (trades, book updates, heartbeats); reported per event.
template <class Est>
void BM_EngineOnEvent(benchmark::State& state) {
  const auto& ev = stream();
  auto eng = std::make_unique<engine::Engine<Shape, Est>>(testkit::default_engine_config());
  std::size_t k = 0;
  std::int64_t offset = 0;
  const std::int64_t span = ev.back().ts_local_ns - ev.front().ts_local_ns + 1'000'000;
  for (auto _ : state) {
    feed::MarketEvent e = ev[k];
    e.ts_local_ns += offset;
    eng->on_event(e);
    std::int64_t px = eng->signal().ask_px_e8;
    benchmark::DoNotOptimize(px);
    if (++k == ev.size()) {
      k = 0;
      offset += span;
    }
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK_TEMPLATE(BM_EngineOnEvent, hawkes::CholeskyEstimator<Shape>);
BENCHMARK_TEMPLATE(BM_EngineOnEvent, hawkes::PotterEstimator<Shape>);

void BM_SplitEngineHotThread(benchmark::State& state) {
  const auto& ev = stream();
  auto eng = std::make_unique<engine::SplitEngine<Shape>>(testkit::default_engine_config());
  std::size_t k = 0;
  std::int64_t offset = 0;
  const std::int64_t span = ev.back().ts_local_ns - ev.front().ts_local_ns + 1'000'000;
  for (auto _ : state) {
    feed::MarketEvent e = ev[k];
    e.ts_local_ns += offset;
    eng->on_event(e);
    std::int64_t px = eng->signal().ask_px_e8;
    benchmark::DoNotOptimize(px);
    if (++k == ev.size()) {
      k = 0;
      offset += span;
      state.PauseTiming();
      eng->drain();  // keep the backlog bounded between laps
      state.ResumeTiming();
    }
  }
  state.SetItemsProcessed(state.iterations());
  eng->drain();
}
BENCHMARK(BM_SplitEngineHotThread);

// State-independent kernels.
void BM_ComputeInterval(benchmark::State& state) {
  const auto cfg = testkit::default_engine_config().model;
  const hawkes::KernelBank<Shape> bank(cfg.beta, cfg.x_cut);
  hawkes::IntervalTerms<Shape> it{};
  const double dt = static_cast<double>(state.range(0)) * 1e-6;
  for (auto _ : state) {
    hawkes::compute_interval(bank, dt, cfg.kappa_slow(), it);
    benchmark::DoNotOptimize(it);
  }
}
BENCHMARK(BM_ComputeInterval)->Arg(1)->Arg(1000)->Arg(100000);  // 1 us, 1 ms, 100 ms

void BM_ProjectRow(benchmark::State& state) {
  testkit::Rng rng(3);
  std::array<double, kD * Shape::Q> at{};
  std::array<double, kD * Shape::Q> w{};
  std::array<double, kD * Shape::Q> out{};
  for (std::size_t m = 0; m < at.size(); ++m) {
    at[m] = 0.1 + 0.1 * rng.normal();
    w[m] = 0.9 + 0.2 * rng.uniform();
  }
  for (auto _ : state) {
    benchmark::DoNotOptimize(hawkes::project_row(at, w, 0.9, out));
  }
}
BENCHMARK(BM_ProjectRow);

void BM_Forecast(benchmark::State& state) {
  const auto cfg = testkit::default_engine_config();
  auto m = std::make_unique<hawkes::Model<Shape, hawkes::CholeskyEstimator<Shape>>>(cfg.model);
  testkit::Rng rng(5);
  testkit::HawkesSimulator<Shape> sim(testkit::default_beta(), testkit::ground_truth_theta(0.3));
  for (const auto& e : sim.simulate(10.0, rng)) {
    m->step(1'000'000'000 + static_cast<std::int64_t>(e.t_s * 1e9), e.channel, 1.0);
  }
  hawkes::Forecaster<Shape> fc;
  fc.init(m->bank(), cfg.forecast);
  core::FixedMatrix<hawkes::BankLayout<Shape>::S, kD> rpost{};
  m->kernel().post_event(rpost);
  hawkes::Forecast<kD> f{};
  for (auto _ : state) {
    fc.compute(m->bank(), rpost, m->served(), f);
    benchmark::DoNotOptimize(f);
  }
}
BENCHMARK(BM_Forecast);

}  // namespace
