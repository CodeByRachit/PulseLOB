// Latency report for the Phase 3 engine (docs/architecture.md Section 9.2, docs/benchmarks.md).
//
//   pulselob_latency [--seconds S] [--json out.json]
//
// 1. Tick-to-quote service time per event (inline engine, both estimator back-ends, and the hot
//    thread of the split engine), from the platform counter into an allocation-free histogram:
//    p50 / p95 / p99 / p99.9 / max, separately for trades (full pipeline) and all events, plus
//    closed-loop throughput.
// 2. Component breakdown: a stage-by-stage replica of Model::step timed in situ on the same stream.
//    The replica's final state is compared byte for byte with the real model, so the breakdown
//    measures exactly the code that runs.
// 3. Split mode: parameter staleness seen by the hot thread and the estimator thread's sustained
//    throughput.
// 4. Open-loop latency: measured trade service times fed through a FIFO queue (Lindley recursion)
//    at Poisson arrival rates of 1e5 and 1e6 events/s, so queueing delay is not hidden
//    (coordinated omission).

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#if defined(__APPLE__)
#include <pthread.h>
#include <sys/qos.h>
#include <sys/sysctl.h>
#endif

#include "cycle_clock.hpp"
#include "hawkes_sim.hpp"
#include "pulselob/core/fp_env.hpp"
#include "pulselob/engine/engine.hpp"
#include "pulselob/engine/split_engine.hpp"
#include "pulselob/version.hpp"
#include "state_compare.hpp"
#include "test_config.hpp"

namespace {

using namespace pulselob;
using bench::CycleClock;
using bench::LatencyHistogram;
using testkit::Shape;
constexpr std::size_t kD = Shape::D;

double g_ns_per_tick = 1.0;
double g_overhead_ns = 0.0;

struct Dist {
  LatencyHistogram all;
  LatencyHistogram trades;
  std::vector<double> trade_ns;  // in stream order, for the open-loop analysis
  double wall_s = 0.0;
  std::size_t events = 0;
};

std::string cpu_name() {
#if defined(__APPLE__)
  char buf[256] = {0};
  std::size_t n = sizeof(buf);
  if (sysctlbyname("machdep.cpu.brand_string", buf, &n, nullptr, 0) == 0) {
    return buf;
  }
#endif
  return "unknown";
}

std::vector<feed::MarketEvent> make_stream(double seconds, std::uint64_t seed) {
  testkit::Rng rng(seed);
  testkit::HawkesSimulator<Shape> sim(testkit::default_beta(), testkit::ground_truth_theta(0.3));
  return testkit::to_market_events(sim.simulate(seconds, rng), rng);
}

template <class E>
void measure(E& eng, const std::vector<feed::MarketEvent>& ev, Dist& d) {
  d.trade_ns.reserve(ev.size());
  const auto w0 = std::chrono::steady_clock::now();
  for (const auto& e : ev) {
    const std::uint64_t c0 = CycleClock::now();
    eng.on_event(e);
    const std::uint64_t c1 = CycleClock::now();
    const double ns = std::max(0.0, static_cast<double>(c1 - c0) * g_ns_per_tick - g_overhead_ns);
    d.all.add(ns);
    if (e.kind == feed::EventKind::Trade) {
      d.trades.add(ns);
      d.trade_ns.push_back(ns);
    }
  }
  d.wall_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count();
  d.events = ev.size();
}

// Unmeasured closed-loop throughput (no timer reads in the loop).
template <class E>
double throughput(E& eng, const std::vector<feed::MarketEvent>& ev) {
  const auto w0 = std::chrono::steady_clock::now();
  for (const auto& e : ev) {
    eng.on_event(e);
  }
  const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count();
  return static_cast<double>(ev.size()) / s;
}

void print_dist(const char* name, const LatencyHistogram& h) {
  std::printf("| %-34s | %9" PRIu64 " | %7.0f | %7.0f | %7.0f | %7.0f | %8.0f | %8.0f |\n", name, h.count(),
              h.mean(), h.quantile(0.50), h.quantile(0.95), h.quantile(0.99), h.quantile(0.999), h.max());
}

// ---- component breakdown: a replica of hawkes::Model::step with a timer around each stage ----
enum Stage { kInterval, kCompensator, kEstInterval, kDecay, kEvent, kSolve, kServe, kDecide, kStages };
const char* kStageNames[kStages] = {
    "interval terms: Q+1 exp, psi, F (Thm 3, 7)", "compensators + CUSUM accumulation",
    "estimator interval update (Thm 7)",          "kernel decay (Thm 2) + long-gap restore",
    "event: regressor, residual, C row",          "ridge replenish + solve / refresh",
    "serve: Perron, projection, rho~, Jacobi",    "decision: forecast, toxicity, quotes"};

template <class Est>
struct Replica {
  using M = hawkes::Model<Shape, Est>;
  hawkes::KernelBank<Shape> bank;
  hawkes::KernelState<Shape> kernel{};
  Est est{};
  hawkes::RegimeDetector<kD> det{};
  hawkes::StabilityGuard<Shape> guard{};
  hawkes::Served<Shape> served{};
  hawkes::IntervalTerms<Shape> it{};
  core::FixedVector<Shape::P> z{};
  core::FixedVector<Shape::P> Z{};
  engine::DecisionLayer<Shape> decision{};
  core::FixedMatrix<hawkes::BankLayout<Shape>::S, kD> rpost{};
  engine::Signal<kD> sig{};
  core::TimestampNs t_prev = 0;
  core::DurationNs hb = 0;
  std::uint64_t steps = 0;
  double n_eff = 0.0;
  bool started = false;
  std::array<double, kStages> ns{};
  std::array<std::uint64_t, kStages> calls{};

  explicit Replica(const engine::EngineConfig<Shape>& cfg) : bank(cfg.model.beta, cfg.model.x_cut) {
    hb = cfg.model.heartbeat_ns;
    est.init(cfg.model);
    det.init(cfg.model.cusum_enabled, cfg.model.cusum_rho, cfg.model.cusum_threshold, cfg.model.kappa_slow(),
             cfg.model.kappa_fast(), cfg.model.dwell_s);
    guard.init(cfg.model, served);
    decision.init(cfg, bank);
  }

  void lap(Stage s, std::uint64_t& c) {
    const std::uint64_t n = CycleClock::now();
    ns[s] += static_cast<double>(n - c) * g_ns_per_tick - g_overhead_ns;
    ++calls[s];
    c = n;
  }

  void step(core::TimestampNs t, int ch, double g, const feed::EventMapper<kD>& mapper) {
    std::uint64_t c = CycleClock::now();
    if (!started) {
      started = true;
      t_prev = t;
    }
    const core::DurationNs dt_ns = t - t_prev;
    if (dt_ns > 0) {
      det.close_batch(t_prev);
      kernel.flush();
      hawkes::compute_interval(bank, core::ns_to_seconds(dt_ns), det.kappa(t_prev), it);
      lap(kInterval, c);
      kernel.compensator(it, Z);
      core::FixedVector<kD> dl{};
      core::matvec(served.theta, Z, dl);
      det.add_compensator(dl);
      lap(kCompensator, c);
      const bool long_gap = dt_ns > hb;
      est.interval(bank, kernel, it, long_gap);
      lap(kEstInterval, c);
      kernel.decay(it);
      n_eff *= it.d;
      if (long_gap) {
        est.full_restore();
      }
      t_prev = t;
      lap(kDecay, c);
    }
    if (ch >= 0) {
      kernel.regressor(bank, z);
      det.on_event(static_cast<std::size_t>(ch));
      est.event(static_cast<std::size_t>(ch), z);
      n_eff += 1.0;
      lap(kEvent, c);
    }
    est.replenish(steps % Shape::P);
    ++steps;
    est.finish();
    lap(kSolve, c);
    guard.serve(est.theta(), served);
    served.n_eff = n_eff;
    served.steps = steps;
    served.flags = det.fast(t) ? hawkes::Served<Shape>::kFlagFastForgetting : 0U;
    if (ch >= 0) {
      kernel.add_pending(static_cast<std::size_t>(ch), g);
    }
    lap(kServe, c);
    kernel.post_event(rpost);
    decision.decide(t, bank, rpost, served, mapper, 0.0, false, sig);
    lap(kDecide, c);
  }
};

// The replica gets its own mapper and mark normalizer, fed the same events with the engine's
// logic (the stream is valid and in order), so it takes exactly the engine's steps.
template <class Est>
bool breakdown(const engine::EngineConfig<Shape>& cfg, const std::vector<feed::MarketEvent>& ev,
               std::array<double, kStages>& mean_ns, std::array<std::uint64_t, kStages>& calls) {
  auto real = std::make_unique<engine::Engine<Shape, Est>>(cfg);
  for (const auto& e : ev) {
    real->on_event(e);
  }
  auto rep = std::make_unique<Replica<Est>>(cfg);
  auto mapper = std::make_unique<feed::EventMapper<kD>>();
  mapper->init(cfg.channels);
  hawkes::MarkNormalizer<kD> marks;
  marks.init(cfg.marks);
  for (const auto& e : ev) {
    const core::TimestampNs t = e.ts_local_ns;
    if (e.instrument < feed::kMaxInstruments && cfg.channels.instruments[e.instrument].present) {
      rep->decision.on_venue(cfg.channels.instruments[e.instrument].venue, t);
    }
    rep->decision.on_input(false);
    if (e.kind == feed::EventKind::Trade) {
      feed::TradeFeatures f{};
      if (mapper->map_trade(e, f)) {
        const double g = marks.mark(static_cast<std::size_t>(f.channel), f.size, f.depth, f.iota);
        rep->step(t, f.channel, g, *mapper);
      }
    } else if (e.kind == feed::EventKind::BookTop) {
      mapper->on_book(e, t);
    } else if (e.kind == feed::EventKind::Heartbeat) {
      rep->step(t, -1, 0.0, *mapper);
    }
  }
  for (int s = 0; s < kStages; ++s) {
    const auto u = static_cast<std::size_t>(s);
    mean_ns[u] = rep->calls[u] ? rep->ns[u] / static_cast<double>(rep->calls[u]) : 0.0;
    calls[u] = rep->calls[u];
  }
  return testkit::same_served(rep->served, real->model().served()) &&
         testkit::same_estimator(rep->est, real->model().estimator());
}

// Open-loop replay: events arrive at Poisson times at the given rate; the replay busy-waits until each
// arrival, so latency (arrival to signal) includes queueing whenever the engine falls behind.
struct Paced {
  LatencyHistogram trades;
  LatencyHistogram all;
  LatencyHistogram staleness;  // split mode: estimator steps behind the stream (+1)
};

template <class E>
void paced(E& eng, const std::vector<feed::MarketEvent>& ev, double rate, Paced& out, bool split) {
  testkit::Rng rng(11);
  const double ticks_per_ns = 1.0 / g_ns_per_tick;
  const std::uint64_t start = CycleClock::now() + static_cast<std::uint64_t>(1e6 * ticks_per_ns);
  double arrival_ns = 0.0;
  for (const auto& e : ev) {
    arrival_ns += rng.exponential(rate) * 1e9;
    const std::uint64_t target = start + static_cast<std::uint64_t>(arrival_ns * ticks_per_ns);
    while (CycleClock::now() < target) {
    }
    eng.on_event(e);
    const double ns = static_cast<double>(CycleClock::now() - target) * g_ns_per_tick;
    out.all.add(ns);
    if (e.kind == feed::EventKind::Trade) {
      out.trades.add(ns);
    }
    if constexpr (requires { eng.steps_pushed(); }) {
      if (split) {
        out.staleness.add(static_cast<double>(eng.steps_pushed() - eng.served().steps) + 1.0);
      }
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  double seconds = 600.0;
  const char* json_path = nullptr;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--seconds") == 0 && i + 1 < argc) {
      seconds = std::atof(argv[++i]);
    } else if (std::strcmp(argv[i], "--json") == 0 && i + 1 < argc) {
      json_path = argv[++i];
    }
  }
#if defined(__APPLE__)
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
  const core::FlushToZeroGuard ftz;
  const double freq = CycleClock::frequency();
  g_ns_per_tick = 1e9 / freq;
  {
    LatencyHistogram h;
    for (int k = 0; k < 2'000'000; ++k) {
      const std::uint64_t a = CycleClock::now();
      const std::uint64_t b = CycleClock::now();
      h.add(static_cast<double>(b - a) * g_ns_per_tick);
    }
    g_overhead_ns = h.mean();
  }
  const auto cfg = testkit::default_engine_config();
  const auto ev = make_stream(seconds, 20261120);
  std::size_t trades = 0;
  for (const auto& e : ev) {
    trades += e.kind == feed::EventKind::Trade ? 1U : 0U;
  }

  std::printf("# PulseLOB %s latency report\n\n", pulselob::kVersionString);
  std::printf("CPU: %s, counter %.3f MHz (resolution %.2f ns), timer overhead %.2f ns (subtracted)\n",
              cpu_name().c_str(), freq / 1e6, g_ns_per_tick, g_overhead_ns);
  std::printf(
      "Stream: %.0f s of simulated D=4 Hawkes flow, %zu events (%zu trades, the rest book updates and "
      "heartbeats)\n\n",
      seconds, ev.size(), trades);

  Dist chol;
  Dist potter;
  Dist split;
  {
    auto e = std::make_unique<engine::Engine<Shape, hawkes::CholeskyEstimator<Shape>>>(cfg);
    measure(*e, ev, chol);
  }
  {
    auto e = std::make_unique<engine::Engine<Shape, hawkes::PotterEstimator<Shape>>>(cfg);
    measure(*e, ev, potter);
  }
  double split_publications = 0.0;
  {
    auto e = std::make_unique<engine::SplitEngine<Shape>>(cfg);
    split.trade_ns.reserve(ev.size());
    const auto w0 = std::chrono::steady_clock::now();
    for (const auto& x : ev) {
      const std::uint64_t c0 = CycleClock::now();
      e->on_event(x);
      const std::uint64_t c1 = CycleClock::now();
      const double ns = std::max(0.0, static_cast<double>(c1 - c0) * g_ns_per_tick - g_overhead_ns);
      split.all.add(ns);
      if (x.kind == feed::EventKind::Trade) {
        split.trades.add(ns);
        split.trade_ns.push_back(ns);
      }
    }
    split.wall_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count();
    e->drain();
    split_publications = static_cast<double>(e->publications());
    e->stop();
  }
  double tp_chol = 0.0;
  double tp_potter = 0.0;
  double tp_split = 0.0;
  {
    auto e = std::make_unique<engine::Engine<Shape, hawkes::CholeskyEstimator<Shape>>>(cfg);
    tp_chol = throughput(*e, ev);
  }
  {
    auto e = std::make_unique<engine::Engine<Shape, hawkes::PotterEstimator<Shape>>>(cfg);
    tp_potter = throughput(*e, ev);
  }
  {
    auto e = std::make_unique<engine::SplitEngine<Shape>>(cfg);
    tp_split = throughput(*e, ev);
    e->drain();
    e->stop();
  }
  // Estimator-thread capacity: a model fed as fast as possible (steps per second).
  double est_capacity = 0.0;
  {
    auto e = std::make_unique<engine::Engine<Shape>>(cfg);
    std::vector<std::pair<core::TimestampNs, int>> steps;
    for (const auto& x : ev) {
      const std::uint64_t before = e->model().counters().steps;
      e->on_event(x);
      if (e->model().counters().steps != before) {
        steps.emplace_back(e->model().t_prev(), x.kind == feed::EventKind::Trade
                                                    ? cfg.channels.channel_of(x.instrument, x.side())
                                                    : -1);
      }
    }
    auto m = std::make_unique<hawkes::Model<Shape, hawkes::CholeskyEstimator<Shape>>>(cfg.model);
    const auto w0 = std::chrono::steady_clock::now();
    for (const auto& s : steps) {
      m->step(s.first, s.second, 1.0);
    }
    est_capacity = static_cast<double>(steps.size()) /
                   std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count();
  }

  std::printf("## Tick-to-quote service time (ns)\n\n");
  std::printf("| %-34s | %9s | %7s | %7s | %7s | %7s | %8s | %8s |\n", "path", "events", "mean", "p50", "p95",
              "p99", "p99.9", "max");
  std::printf("|%s|%s|%s|%s|%s|%s|%s|%s|\n", std::string(36, '-').c_str(), std::string(11, '-').c_str(),
              std::string(9, '-').c_str(), std::string(9, '-').c_str(), std::string(9, '-').c_str(),
              std::string(9, '-').c_str(), std::string(9, '-').c_str(), std::string(10, '-').c_str());
  print_dist("inline, Cholesky: trades", chol.trades);
  print_dist("inline, Cholesky: all events", chol.all);
  print_dist("inline, Potter: trades", potter.trades);
  print_dist("inline, Potter: all events", potter.all);
  print_dist("split, hot thread: trades", split.trades);
  print_dist("split, hot thread: all events", split.all);
  std::printf("\n## Throughput (closed loop, no timer in the loop)\n\n");
  std::printf("| path | million events/s |\n|---|---|\n");
  std::printf("| inline, Cholesky | %.2f |\n| inline, Potter | %.2f |\n| split, hot thread | %.2f |\n",
              tp_chol / 1e6, tp_potter / 1e6, tp_split / 1e6);
  std::printf("| estimator alone (model steps/s, Cholesky) | %.2f |\n", est_capacity / 1e6);

  std::array<double, kStages> bc{};
  std::array<double, kStages> bp{};
  std::array<std::uint64_t, kStages> cc{};
  std::array<std::uint64_t, kStages> cp{};
  const bool ok_c = breakdown<hawkes::CholeskyEstimator<Shape>>(cfg, ev, bc, cc);
  const bool ok_p = breakdown<hawkes::PotterEstimator<Shape>>(cfg, ev, bp, cp);
  std::printf("\n## Component breakdown (mean ns per call, in situ)\n\n");
  std::printf("Replica state equals the real model byte for byte: Cholesky %s, Potter %s.\n\n",
              ok_c ? "yes" : "NO", ok_p ? "yes" : "NO");
  std::printf("| stage | calls | Cholesky | Potter |\n|---|---|---|---|\n");
  for (int s = 0; s < kStages; ++s) {
    const auto u = static_cast<std::size_t>(s);
    std::printf("| %s | %" PRIu64 " | %.0f | %.0f |\n", kStageNames[s], cc[u], bc[u], bp[u]);
  }

  std::printf("\n## Open-loop latency (ns, scheduled arrival to signal, includes queueing)\n\n");
  std::printf(
      "Poisson arrivals at the given rate; the replay busy-waits for each arrival. Staleness is the number "
      "of estimator steps the hot thread's parameters lag the stream when its signal is produced.\n\n");
  std::printf(
      "| path | rate (events/s) | trades p50 | p99 | p99.9 | max | all p99 | staleness p50 / p99 / max |\n");
  std::printf("|---|---|---|---|---|---|---|---|\n");
  for (double rate : {1e5, 1e6}) {
    Paced pc;
    Paced pp;
    Paced ps;
    {
      auto e = std::make_unique<engine::Engine<Shape, hawkes::CholeskyEstimator<Shape>>>(cfg);
      paced(*e, ev, rate, pc, false);
    }
    {
      auto e = std::make_unique<engine::Engine<Shape, hawkes::PotterEstimator<Shape>>>(cfg);
      paced(*e, ev, rate, pp, false);
    }
    {
      auto e = std::make_unique<engine::SplitEngine<Shape>>(cfg);
      paced(*e, ev, rate, ps, true);
      e->drain();
      e->stop();
    }
    const std::pair<const char*, const Paced*> rows[] = {
        {"inline, Cholesky", &pc}, {"inline, Potter", &pp}, {"split, hot thread", &ps}};
    for (const auto& [name, d] : rows) {
      std::printf("| %s | %.0e | %.0f | %.0f | %.0f | %.0f | %.0f |", name, rate, d->trades.quantile(0.5),
                  d->trades.quantile(0.99), d->trades.quantile(0.999), d->trades.max(),
                  d->all.quantile(0.99));
      if (d == &ps) {
        std::printf(" %.0f / %.0f / %.0f |\n", d->staleness.quantile(0.5) - 1.0,
                    d->staleness.quantile(0.99) - 1.0, d->staleness.max() - 1.0);
      } else {
        std::printf(" n/a |\n");
      }
    }
  }
  std::printf("\nSplit mode published %.0f parameter snapshots in the closed-loop run.\n",
              split_publications);
  if (json_path != nullptr) {
    FILE* f = std::fopen(json_path, "w");
    if (f != nullptr) {
      auto dj = [&](const char* k, const LatencyHistogram& h, bool last) {
        std::fprintf(f,
                     "  \"%s\": {\"count\": %" PRIu64
                     ", \"mean\": %.1f, \"p50\": %.1f, \"p95\": %.1f, \"p99\": %.1f, \"p999\": %.1f, "
                     "\"max\": %.1f}%s\n",
                     k, h.count(), h.mean(), h.quantile(0.5), h.quantile(0.95), h.quantile(0.99),
                     h.quantile(0.999), h.max(), last ? "" : ",");
      };
      std::fprintf(f, "{\n  \"version\": \"%s\",\n  \"cpu\": \"%s\",\n  \"counter_resolution_ns\": %.3f,\n",
                   pulselob::kVersionString, cpu_name().c_str(), g_ns_per_tick);
      std::fprintf(f, "  \"throughput_meps\": {\"cholesky\": %.3f, \"potter\": %.3f, \"split_hot\": %.3f},\n",
                   tp_chol / 1e6, tp_potter / 1e6, tp_split / 1e6);
      dj("cholesky_trades", chol.trades, false);
      dj("potter_trades", potter.trades, false);
      dj("split_trades", split.trades, true);
      std::fprintf(f, "}\n");
      std::fclose(f);
    }
  }
  return ok_c && ok_p ? 0 : 1;
}
