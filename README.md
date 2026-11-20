# PulseLOB

**Real-time cross-venue order-flow toxicity detection with an online, non-stationary, marked multivariate Hawkes process.** The engine is a header-only C++20 core with zero allocations on the hot path, verified against definition-level references to 1e-10. It has zero-copy Python bindings and a data layer for synthetic and Binance data. Next it will be validated inside an Avellaneda–Stoikov market-making simulator with realistic latency and queue modeling.

[![CI](https://github.com/CodeByRachit/PulseLOB/actions/workflows/ci.yml/badge.svg)](https://github.com/CodeByRachit/PulseLOB/actions/workflows/ci.yml)

> **Status: Phase 4 (Python bindings and data infrastructure), awaiting review.** The C++20 engine (Phase 3, `v0.3.0`) is available from Python with zero-copy batch processing. Synthetic data with ground truth, a live Binance recorder and a historical archive loader all produce the same `.plev` event files. Each phase ships as a tagged version (see [Roadmap](#roadmap)).

## The problem

When a perpetual-futures venue is swept, the spot venue usually follows a few milliseconds later. A spot market maker whose quotes are still resting at that point is filled by better-informed flow and loses money on those fills. This is called adverse selection. To defend against it, a model has to do four things:

1. capture both self-excitation and **lead-lag contagion** across venues;
2. **adapt online** as market regimes change;
3. stay **constant-time per event**, so it can run at feed speed;
4. turn all of this into **quotes**: skew the price, widen the endangered side, and pull it before being hit.

## The approach

The full derivations, proofs and trade-offs are in **[docs/math_spec.md](docs/math_spec.md)**. In brief:

- **Model.** A linear marked Hawkes process with a fixed bank of Erlang kernels. Exponential kernels capture fast self-excitation, and Erlang-2 "humps" peaking at 5–20 ms capture the lead-lag. With the bank fixed, the intensity is linear in the parameters, $\boldsymbol\lambda(t)=\boldsymbol\Theta(t)\,\mathbf z(t)$, and the regressor $\mathbf z(t)$ does not depend on the parameters. Each event's state update is an exact recursion.
- **Online estimator.** The estimator minimizes a discounted least-squares contrast. Its sufficient statistics do not depend on the parameters, so the online estimate **is** the batch estimate, exactly. Each event costs $O(r_{\max}Q^3D^2)$ regardless of how long the history is.
- **Regime adaptation.** Change detection on time-rescaled residuals switches the model to faster forgetting after a structural break.
- **Stability.** Served parameters are certified sub-critical ($\rho(\mathbf G)\le\rho_{\max}<1$) at every event, through an exact projection onto a Collatz–Wielandt polytope.
- **Toxicity.** The score $\tau^\pm\in[0,1)$ measures how anomalous and how one-sided the forecast flow is. It scores zero for two-sided volume, and it rises *before* the lagging venue reacts.
- **Control.** The Avellaneda–Stoikov quotes are re-derived for Hawkes-driven price jumps. A Glosten–Milgrom adverse-selection premium widens the endangered side, a side is pulled when its break-even distance exceeds the quoting cap, and hysteresis provably bounds message traffic.

Every closed form, recursion and certificate in the spec is checked numerically against a definition-level brute-force reference:

```bash
pip install -r research/requirements.txt
python research/phase1/validate_math_spec.py          # identity checks, about 1 minute
python research/phase1/validate_math_spec.py --full   # plus the estimator-comparison experiments
```

CI runs the identity checks on every push and pull request.

The engineering design (component boundaries, build, memory layout, concurrency and data schemas) is in **[docs/architecture.md](docs/architecture.md)**.

## The engine (Phase 3)

The engine implements the per-event algorithm of the spec in `include/pulselob/`, in the `core`, `feed`, `hawkes`, `control` and `engine` components. Each event costs constant time regardless of history length, and nothing allocates after construction. A dedicated test binary proves this by counting every `operator new` across 10^6 events, including in split mode with both threads running.

**Verification.** Every recursion is tested against a brute-force reference at a relative tolerance of 1e-10, under GCC 13, Clang 18 (ASan+UBSan, TSan) and AppleClang. A separate Python implementation computes the whole pipeline from the definitions (`research/phase3/crosscheck_reference.py`): direct sums over the history, quadrature, and the batch least-squares solution. On a 496-event stream with ties, out-of-order data, long gaps and forgetting switches, the C++ engine matches it as follows:

- the estimate, to 2.9e-14;
- every signal, to 1.9e-12;
- every quote, flag and status, exactly.

The full table of math_spec V1–V11 results is in [Section 10 of the spec](docs/math_spec.md#10-verification-obligations-for-phase-3). On a simulated regime shift with $D=4$ and the full kernel bank, the cross-venue coupling estimate goes from 0.116 to 0.515 (true values 0.10 and 0.50). The change detector halves the time to cover half the step (17 s against 60 s).

**Measured performance** (Apple M4, macOS, AppleClang 21; [docs/benchmarks.md](docs/benchmarks.md) has the method, every table and the run-to-run ranges):

| Path | Tick-to-quote p50 | p99 | Throughput |
|---|---|---|---|
| inline engine, information-form estimator (default) | 0.99 µs | 1.1–1.3 µs | 1.8 M events/s |
| inline engine, square-root (Potter) estimator | 2.6 µs | 3.3 µs | 0.73 M events/s |
| split mode: estimator on a second core, hot thread | 0.16 µs | 0.2–0.4 µs | 8.4 M events/s |

Tick-to-quote is the time from a trade arriving to a complete signal with quotes. macOS cannot isolate cores, so the tails are indicative. Inline mode is the default because it is bit-for-bit reproducible. Split mode is recommended for live quoting when the latency budget is below about 1 µs.

## Python

```bash
pip install .                       # or ".[record]" for the live recorder
```

```python
import pulselob as pl
from pulselob import synth

ds = synth.generate("lead_lag", seed=1)                # ground-truth synthetic stream
eng = pl.Engine(pl.EngineConfig.default())             # estimator="cholesky", mode="inline"
res = eng.run(ds.events, state_every=1000)             # zero-copy, GIL released
res.signals["tau_buy"], res.signals["ask_px_e8"], res.states["theta_hat"]
eng.run("data/btcusdt.plev")                           # a memory-mapped .plev file works the same way
```

`Engine.run` processes **1.82 M events/s**, the same speed as the native C++ driver on the same file (1.77 M events/s; [docs/benchmarks.md](docs/benchmarks.md) Section 8). Its output is byte-identical to the native driver's, which a test checks.

Data commands ([docs/data.md](docs/data.md) has the formats, endpoints and known gaps):

```bash
pulselob-synth flash_crash --out data/crash.plev                       # synthetic, with ground truth
pulselob-record --symbol BTCUSDT --out data/live.plev --duration 3600  # live Binance Spot + USD-M
pulselob-archive --day 2024-03-01 --out data/btc-2024-03-01.plev       # data.binance.vision archive
```

## Building

You need CMake 3.25 or newer, a C++20 compiler (GCC 13+, Clang 16+ or AppleClang 15+), and Python 3.9 or newer for the package.

```bash
cmake --preset release            # also: debug, asan, tsan, bench
cmake --build --preset release
ctest --preset release

cmake --preset bench && cmake --build --preset bench
./build/bench/bench/pulselob_latency   # latency distributions, breakdown, threading experiment

pip install ".[test]"             # builds the pybind11 module with scikit-build-core
python -m pytest
```

## Roadmap

| Phase | Deliverable | Version | Status |
|---|---|---|---|
| 1 | Mathematical specification and numerical validation | `v0.1.0` | released |
| 2 | Repository architecture, CMake, memory layout, data schemas | `v0.2.0` | released |
| 3 | C++20 engine, exactness and stress tests, micro-benchmarks | `v0.3.0` | released |
| 4 | pybind11 bindings, synthetic generator, Binance recorder and archive loader | `v0.4.0` | in review |
| 5 | Event-driven market-making simulator and comparative backtest | `v0.5.0` | planned |
| 6 | Figures, whitepaper, publication polish | `v1.0.0` | planned |

Every published number comes from a real run; the raw outputs are in [`bench/results/`](bench/results/).

## Repository layout

```text
docs/math_spec.md                      specification (derivations, proofs, verification results)
docs/architecture.md                   engineering specification (layout, memory, concurrency, schemas)
docs/benchmarks.md                     measurements and the decisions they support
docs/data.md                           data sources: synthetic, live recorder, archive; formats and gaps
include/pulselob/                      header-only C++20 engine: core/, feed/, hawkes/, control/, engine/, synth/
bindings/, tools/                      pybind11 module pulselob._core; native replay driver pulselob-replay
python/pulselob/                       Python package: Engine, config, .plev, synth, data (recorder, archive)
tests/cpp/, tests/python/              GoogleTest (incl. tests/cpp/alloc zero-allocation proof) and pytest
tests/support/, tests/fixtures/        simulator and quadrature references; Python cross-check fixture
bench/                                 Google Benchmark suite, latency report, raw results
cmake/, CMakePresets.json              build options, warnings, sanitizers, pinned dependencies
research/phase1/validate_math_spec.py  numerical verification of every claim in the spec
research/phase3/                       definition-level reference implementation of the engine
scripts/                               version and include-layering checks
.github/workflows/                     CI, weekly research run, tag-driven releases
```

## Versioning

The project follows [Semantic Versioning](https://semver.org/). [`VERSION`](VERSION) is the single source of truth, and every release has an entry in [`CHANGELOG.md`](CHANGELOG.md) in the [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format.

Each approved phase bumps the minor version. Corrections within a phase bump the patch version. `1.0.0` marks the publication release.

Pushing a tag `vX.Y.Z` that matches `VERSION` triggers the release workflow, which creates a GitHub Release from the matching changelog section. The math spec also carries its own *spec version*, which changes whenever the mathematics changes.

## License

[MIT](LICENSE) © 2026 Rachit Yadav.
