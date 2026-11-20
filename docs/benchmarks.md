# PulseLOB: Benchmarks

Every number in this document comes from a run of the code on the machine described below: Sections 1–7 at version 0.3.0 (the engine), Section 8 at version 0.4.0 (the Python bindings). The raw outputs are committed in [`bench/results/`](../bench/results/). Nothing here is estimated or extrapolated.

## Environment

| Item | Value |
|---|---|
| CPU | Apple M4 (4 performance and 6 efficiency cores), 128 KB L1D per performance core |
| OS | macOS (Darwin 27.2.0), no core isolation, threads at the user-interactive QoS class |
| Compiler | AppleClang 21.0.0, CMake 4.3.2, `bench` preset: Release (`-O3`), `-mcpu=native`, `-ffp-contract=off`, no fast-math |
| Timer | `CNTVCT_EL0` read behind an `ISB`; `CNTFRQ_EL0` reports 1 GHz (1 ns resolution); the mean back-to-back read cost of 14.5 ns is subtracted from every sample |
| Floating point | flush-to-zero enabled on the measuring thread and the estimator thread |

macOS cannot pin a thread to a core or isolate a core from the scheduler, so OS preemption is visible in the tails (p99.9 and max). Following [architecture.md](architecture.md) Section 6.4, treat the tails here as indicative. The medians are stable across runs. A Linux measurement on isolated cores (`isolcpus`, `nohz_full`) is the right source for publishable tail numbers and is planned with the Phase 5 simulator.

## Workload

The stream is 600 s of a simulated $D=4$ Hawkes process. It uses the default kernel bank (1000, 200, 50 and 1 per second), self-excitation on the fast and slow exponentials, and a perp-to-spot lead-lag of 0.3 on the 5 ms hump, simulated by Ogata thinning (`tests/support/hawkes_sim.hpp`). It is converted to `MarketEvent`s with a random-walk mid, sweeps, log-normal sizes, a book update after every trade, and a heartbeat after every 100 ms of silence. In total that is 153,628 events, of which 76,801 are trades. A trade runs the full per-event algorithm of math_spec Section 9 plus the decision layer. A book update only produces a read-only signal.

To reproduce:

```bash
cmake --preset bench && cmake --build --preset bench
./build/bench/bench/pulselob_latency --seconds 600     # distributions, breakdown, threading, open loop
./build/bench/bench/pulselob_bench --benchmark_repetitions=5 --benchmark_report_aggregates_only=true
```

## 1. Tick-to-quote service time

This is the time for one `on_event` call, from receiving the event to a complete `Signal` (intensities, toxicity, reservation shift, quotes, triggers). The table is from run 1. The brackets give the range over three full runs.

| Path | p50 (ns) | p95 (ns) | p99 (ns) | p99.9 (ns) | mean (ns) |
|---|---|---|---|---|---|
| inline, information form (default): trades | 992 [992–1056] | 1088 [1088–1120] | 1120 [1120–1312] | 2112 [2112–2880] | 994 |
| inline, Potter: trades | 2624 [2624–2752] | 3264 [3264–5760] | 3328 [3328–13312] | 7552 [5504–69632] | 2694 |
| split mode, hot thread: trades | 156 [156–160] | 196 [196–368] | 196 [196–416] | 368 [368–4864] | 158 |
| inline, information form: all events | 912 | 1088 | 1120 | 1504 | 547 |
| split mode, hot thread: all events | 112 | 196 | 196 | 368 | 127 |

The worst single samples were 10 µs (inline, information form), 100 µs (Potter) and 10 µs (split) in run 1, and up to 3.8 ms in run 3. These are OS preemptions, not engine work.

## 2. Throughput

These are closed-loop replays with no timer in the loop, giving events per second over the full mix.

| Path | Million events/s (run 1) [range over 3 runs] |
|---|---|
| inline, information form | 1.87 [1.73–1.87] |
| inline, Potter | 0.73 [0.67–0.73] |
| split mode, hot thread | 8.45 [5.77–8.45] |
| estimator alone (model steps/s, information form) | 1.15 [1.14–1.17] |

The Google Benchmark suite gives the same picture as medians of 5 repetitions ([raw output](../bench/results/gbench-apple-m4.txt)): `on_event` costs 550 ns with the information form and 1,389 ns with Potter, and the split hot thread costs 123 ns.

## 3. Component breakdown

These are mean nanoseconds per call, timed in situ on the same stream. A stage-by-stage replica of `Model::step` is run with a timer around each stage. At the end, the replica's estimator and served parameters are compared field by field with the real engine's, and they are identical for both back-ends. So the breakdown measures exactly the code the engine runs.

| Stage | Information form | Potter |
|---|---|---|
| interval terms: $Q+1$ exponentials, $\psi$ values, $\mathbf F$ (Theorems 3, 7) | 116 | 121 |
| compensators and CUSUM accumulation | 21 | 18 |
| estimator interval update (Theorem 7) | 87 | 1,867 |
| kernel decay (Theorem 2) and long-gap restore | 4 | 4 |
| event: regressor, residual, $\mathbf C$ row (Potter: also $\tilde{\boldsymbol\Theta}$ row) | 6 | 160 |
| ridge replenishment, then the Cholesky solve (information) or the row refresh (Potter) | 538 | 275 |
| serving: Perron step, four row projections, criticality, Jacobi sweep | 62 | 58 |
| decision: forecasts, toxicity, quotes, triggers, throttles | 87 | 87 |

Selected state-independent kernels (Google Benchmark, median of 5):

| Kernel | Time |
|---|---|
| `compute_interval`, $\Delta$ = 1 µs / 1 ms / 100 ms | 77 / 109 / 68 ns |
| one row projection, $DQ=16$ | 69 ns |
| forecasts and toxicity | 51 ns |
| `std::exp` | 1.99 ns |

## 4. Open-loop latency

Events arrive at Poisson times at a fixed rate. The replay busy-waits until each arrival, and latency runs from the scheduled arrival to the signal, so queueing delay is included and coordinated omission is avoided. Staleness is the number of estimator steps by which the split hot thread's parameters lag the stream when it produces a signal.

| Path | Rate (events/s) | Trades p50 (ns) | Trades p99 (ns) | Staleness p50 / p99 (steps) |
|---|---|---|---|---|
| inline, information form | 1e5 | 1024 [1008–1024] | 1312 [1312–27136] | n/a |
| inline, Potter | 1e5 | 2688 [2688] | 4352 [4352–4480] | n/a |
| split mode, hot thread | 1e5 | 192 [192–196] | 440 [384–65536] | 1 / 55 [1 / 55–407] |
| inline, information form | 1e6 | 1152 [1120–1184] | 303,104 [4,096–671,744] | n/a |
| inline, Potter | 1e6 | saturated: service time (2.7 µs) exceeds the mean inter-arrival time (1 µs), so the queue grows without bound | | n/a |
| split mode, hot thread | 1e6 | 232 [224–232] | 752 [544–2,031,616] | 1 / 77 [1–2 / 26–4607] |

At 10^6 events/s the inline engine runs at about 55% utilization. A single OS preemption of a few hundred microseconds builds a queue that takes thousands of events to drain, and that is the spread in its p99 across runs. Run 2 contained a multi-millisecond stall that affected every path, which is why some ranges reach milliseconds.

## 5. Cost of the floating-point policy

The same report was built with FMA contraction allowed (`-DPULSELOB_BENCH_FP_CONTRACT=ON`, benchmark-only) and both builds were run on a 120 s stream. The information-form trade p50 was 1,056 ns in both builds, the p99 was 1,344 ns with `-ffp-contract=off` and 1,248 ns with contraction, and throughput was 1.84 M events/s in both. The difference is within run-to-run variation. The reproducibility policy of [architecture.md](architecture.md) Section 4.3 therefore costs nothing measurable here.

## 6. Decisions supported by these measurements

### Estimator back-end: information form

| | Information form | Potter (spec 0.1.0) |
|---|---|---|
| tick-to-quote p50, trades | 0.99 µs | 2.62 µs |
| throughput | 1.87 M events/s | 0.73 M events/s |
| $\tilde{\boldsymbol\Theta}$ against the batch definition (V5) | 2.9e-14 | 8.3e-12 |
| derived signals against the Python reference | 1.9e-12 | 1.3e-9 |
| long-gap handling | identical algorithm | needs the refactor path |

The information form is faster at $D=4$ because a fresh factorization ($P^3/6\approx820$ multiply-adds plus a $D$-wide solve) costs less than seven or more Potter rank-one updates of $3P^2$ each. It is also more accurate, because the covariance form's rounding grows when new information dominates a badly scaled prior, which happens at start-up. The crossover estimate in math_spec Section 5.4.1 puts the square-root form ahead only for $D$ above about 10. Potter stays available as a template argument, and every test runs against both.

### Threading: inline by default, split for live quoting below about 1 µs

* **Inline is the default** for all research, replay, backtests and the Phase 5 simulator. It is bit-for-bit reproducible, and tests check this. Its p50 is 0.99 µs and its p99 is 1.1–1.3 µs. It sustains 10^6 events/s with no headroom left for OS jitter.
* **Split mode** cuts the hot thread's p50 about 6 times, to 156 ns, and keeps the p99 at 0.2–0.4 µs while closing the loop at 8.4 M events/s. Its parameters lag by a median of one estimator step. At 10^5 events/s the p99 lag of 55 steps is about 0.5 ms of stream time, and the served parameters move on the forgetting time scale of 5 to 60 s. Its costs are non-reproducible signals, a second core, and an additional failure mode: if the estimator falls 2^16 steps behind, the kill switch trips.
* **Recommendation.** Keep inline as the default. Enable split mode for live quoting when the tick-to-quote budget is below about 1 µs or the sustained event rate exceeds about 5 × 10^5 per second. Both conditions apply to a co-located deployment, neither to research use.

## 7. Detector calibration input (for Phase 5)

`Regime.InControlFalseAlarmRate` measures false alarms per hour on a stationary $D=4$ stream (193,274 events over 30 minutes): 32 at $b_{\mathrm{cs}}=14$ (the default inherited from the $D=2$ experiment), 14 at $b_{\mathrm{cs}}=20$ and 0 at $b_{\mathrm{cs}}=30$. With $2D=8$ CUSUMs and plug-in residuals the default threshold is too low for $D=4$. It still detects the regime shift of V10 (three alarms within 60 s of the shift, against four in the 300 s before it). The threshold should be recalibrated by in-control run-length simulation in Phase 5, as math_spec Section 5.5 anticipates.

Since version 0.4.0 the default is the conservative interim value $b_{\mathrm{cs}}=30$ (math_spec 0.2.1). On the V10 shift, over seeds 2024, 7 and 11, it raised no false alarm before the shift and fired on none of the three shifts within the run. Recovery then follows the slow forgetting half-life (60–70 s to cover half of the step), against 17–68 s at $b_{\mathrm{cs}}=14$, which also produced 1–4 false alarms in the 300 s before the shift. The Phase 5 sweep will choose the operating point between false alarms and detection delay.

## 8. Python batch throughput (version 0.4.0)

`python -m pulselob.bench` writes 4,993,072 events (2,496,535 trades, plus book updates and heartbeats, from the `two_sided` scenario) to a `.plev` file. It then times, on that one file, the native C++ driver `pulselob-replay` and `pulselob.Engine.run` reading the memory-mapped file in place with the GIL released. Each figure is the best of three runs with a fresh engine ([raw output](../bench/results/python-batch-apple-m4.json)).

| Path | Million events/s |
|---|---|
| native driver, inline, information form | 1.77 |
| Python `Engine.run`, inline | 1.82 |
| Python `Engine.run`, inline, model state sampled every 1,000 steps | 1.82 |
| Python `Engine.run`, split mode | 7.86 |

The Python path runs at 1.03 times the native driver's speed, which is within run-to-run variation: the binding adds no measurable cost per event. On the same run, generating the 5 million events took 0.54 s. A Python test also checks that `Engine.run` and the native driver produce identical output bytes on the same file, for both estimators and both kernel banks.
