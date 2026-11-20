# PulseLOB: Architecture and Engineering Specification

| Field | Value |
|---|---|
| Document version | 0.4.0 |
| Phase | 2 (architecture), updated in 3 (engine as built) and 4 (bindings and data) |
| Implements | [math_spec.md](math_spec.md), spec version 0.2.1 |
| Status | Engine, bindings and data layer built and verified; measurements in [benchmarks.md](benchmarks.md), data sources in [data.md](data.md) |
| Date | 2026-11-20 |

This document fixes how the mathematics of the spec becomes software. Everything marked **(built)** exists and is tested on Linux (GCC 13, Clang 18, with AddressSanitizer, UBSan and ThreadSanitizer) and macOS arm64 (AppleClang). Phase 2 built the build system, the schema and layout types, the SPSC ring, the Python package skeleton and CI. Phase 3 built the model engine: the `hawkes`, `control` and `engine` components, the event mapper, the seqlock and split mode. Everything marked **(Phase n)** is specified here and implemented in that phase.

---

## 0. Decisions at a glance

1. **Header-only core, templated on a compile-time model shape.** `ModelShape<D, r_1..r_Q>` fixes the channel count and the kernel orders, so every state array has a size known at compile time. That gives zero allocation after construction by construction rather than by discipline. Decay rates and all other parameters are runtime configuration.
2. **No C++20 named modules.** Component boundaries are directories, namespaces and CMake targets. CI enforces them with an include-layering check and by compiling every public header standalone with `-fno-exceptions -fno-rtti` (Section 2.3).
3. **Own fixed-size matrices, not Eigen.** Rows are padded to 64 bytes and the padding lanes are kept at zero, so the five kernels the estimator needs run full-width SIMD loops with no remainder handling (Section 5.2).
4. **Exact integers at the edges, doubles inside.** Timestamps are `int64` nanoseconds. Prices and quantities are `int64` fixed-point with scale $10^{-8}$, which is exact for every Binance value. The model computes in IEEE double with no fast-math and no FMA contraction, so a run is reproducible (Section 4.3).
5. **One 64-byte event record everywhere.** The same `MarketEvent` bytes serve as ring slots, NumPy structured arrays and memory-mapped `.plev` files. Ingestion is zero-copy from a file all the way to the engine (Section 7).
6. **Arrays of structures for transport, structures of arrays for model state.** Events are consumed whole, while model state is swept across channels and coordinates (Section 5.1).
7. **Deterministic single-threaded engine by default.** Optionally, the estimator moves to a second core and publishes parameters through a seqlock; this resolves decision 4 of the math spec (Section 6). Backtests and tests always use the deterministic mode. Phase 3 measured both (Section 6.1): split mode cuts the hot thread's tick-to-quote time about 6 times, with parameters lagging by a median of one estimator step.
8. **128-byte false-sharing separation.** This covers the 128-byte cache lines of Apple M-series cores, measured on the development machine, and x86 adjacent-line prefetching. Data arrays use 64-byte alignment.
9. **No exceptions or RTTI.** Errors are status codes plus counters. Configuration is checked by `validate()`, which returns the first violated condition; as built, no PulseLOB code throws (Section 4.2).
10. **Python packaging via scikit-build-core and pybind11**, with the version read from `VERSION`.
11. **A data constraint found in this phase.** Binance's public archives have **no spot top-of-book data at any date**. They have futures top-of-book only for 2023-05-16 to 2024-03-30 (BTCUSDT). This affects Phases 4 and 5; Section 7.4 gives the options.
12. **Information-form estimator by default (Phase 3).** Both back-ends of math_spec Section 5.4.1 are implemented. The information form (stored $\mathbf M$, fresh Cholesky solve per event) is 2.7 times faster than the square-root (Potter) form at $D=4$ and matches the definition-level reference about 300 times more closely. Potter remains selectable as a template argument.

---

## 1. Repository layout

```text
PulseLOB/
├── CMakeLists.txt                 root build: core target, header contract, sub-projects  (built)
├── CMakePresets.json              debug / asan / tsan / release / bench presets            (built)
├── cmake/                         options, warnings + FP policy, sanitizers, pinned deps   (built)
├── include/pulselob/              public C++ headers (header-only core)
│   ├── version.hpp.in             generated from VERSION                                   (built)
│   ├── core/                      platform constants, time, fixed point, fixed matrices,
│   │                              model shape, psi, linear algebra, FTZ guard, seqlock     (built)
│   ├── feed/                      MarketEvent schema, SPSC ring, event mapper, Binance CSV (built)
│   ├── hawkes/                    kernel bank, estimator (2 back-ends), CUSUM, stability,
│   │                              forecasts and toxicity, marks, model                     (built)
│   ├── control/                   quoting law, pull triggers, kill switch, throttle        (built)
│   ├── engine/                    configuration, decision layer, inline engine, split
│   │                              engine, replay driver, Signal, counters                  (built)
│   ├── synth/                     Ogata simulator, synthetic MarketEvent streams           (built)
│   └── sim/                       market-making simulator                                  (Phase 5)
├── bindings/module.cpp            pybind11 module pulselob._core                           (built)
├── tools/replay.cpp               native replay driver pulselob-replay                     (built)
├── python/pulselob/               Python package: schema, .plev I/O, config, Engine,
│                                  synth, data (recorder, archive), bench                   (built)
│                                  simulator, analysis                                      (Phase 5–6)
├── tests/cpp/                     GoogleTest unit, equivalence, stress tests; alloc proof  (built)
├── tests/support/                 Ogata simulator, quadrature references, test configs     (built)
├── tests/fixtures/                Python-reference cross-check fixture                     (built)
├── tests/python/                  pytest tests                                             (built)
├── bench/                         Google Benchmark suite, latency report                   (built)
├── docs/                          math_spec.md (Phase 1), architecture.md (this document)
├── research/                      spec validation (Phase 1), definition-level reference (Phase 3)
├── scripts/                       version and layering checks                              (built)
├── pyproject.toml                 Python build (scikit-build-core)                         (built)
└── .github/workflows/             CI, weekly research run, tag-driven releases             (built)
```

The prompt for this project suggested `include/hawkes_toxicity/`. The headers live under `include/pulselob/` instead, to match the project and package name.

---

## 2. Component architecture

### 2.1 Components

| Component (namespace) | Responsibility | Spec sections | May depend on | Phase |
|---|---|---|---|---|
| `core` | platform constants, time, fixed point, fixed-size matrices, model shape, ψ functions, small linear-algebra kernels, FP environment, seqlock | 4.2, 4.6–4.8 | standard library only | 2 (types), 3 (kernels) |
| `feed` | `MarketEvent` schema and validation, SPSC ring, event mapper (channel and raw mark features from events and book state), stream merger | 2.2, 2.3, 4.4, 4.5 | `core` | 2 (schema, ring), 3 (mapper), 4 (merger) |
| `hawkes` | kernel bank and state recursion, compensators, least-squares estimator (information form and Potter, ridge replenishment), CUSUM, stability projection, forecasts and toxicity, mark normalization | 2.3, 3–7 | `core` | 3 |
| `control` | quoting law, pull and kill-switch state machines, message throttling | 8 | `core`, `hawkes` | 3 |
| `engine` | per-event pipeline (spec Section 9), counters, snapshot/restore, replay driver | 9 | `core`, `feed`, `hawkes`, `control` | 3 |
| `synth` | exact Ogata simulation of the model with piecewise parameters; MarketEvent streams with ground truth | 3 | `core`, `feed` | 4 |
| `sim` | latency and queue-position simulator, fills, fees, PnL, markouts | (Phase 5 spec) | `core`, `feed`, `engine` | 5 |
| bindings, tools | pybind11 module; native replay driver | – | everything | 2 (schema), 4 (engine, data) |

`hawkes` does not know about markets: it sees channels, times and marks. `feed` turns market events into those. This keeps the model testable against the spec's definitions without any market-data plumbing.

### 2.2 Data flow

```mermaid
flowchart LR
  subgraph Sources
    BV["data.binance.vision CSV (trades, bookTicker)"]
    WS["live WebSocket (Phase 4)"]
    SY["synthetic generator (Phase 4)"]
  end
  BV --> N["normalizer: decimal to e8, sweep aggregation, timestamps"]
  WS --> N
  SY --> N
  N --> F[".plev files (memory-mapped)"]
  N --> R["SPSC ring per venue"]
  F --> M["merge by ts_local"]
  R --> M
  M --> EM["feed::EventMapper: channel, mark g_k, book state"]
  EM --> HS["hawkes: state recursion"]
  HS --> ES["hawkes: estimator + CUSUM"]
  ES --> ST["hawkes: stability projection"]
  ST --> FC["hawkes: forecasts, toxicity"]
  FC --> CT["control: quotes, pull, kill switch"]
  CT --> SG["Signal record"]
  SG --> SIM["sim (Phase 5) / order gateway"]
```

### 2.3 How the boundaries are enforced (built)

* **Layering.** [`scripts/check_layering.py`](../scripts/check_layering.py) parses every `#include` under `include/pulselob/<component>/` and fails CI if a component includes a component not listed for it in Section 2.1, or if a public header includes third-party code.
* **Self-contained, exception-free headers.** CMake generates one translation unit per public header, containing only that `#include`, and compiles all of them as the `pulselob_header_check` target with `-fno-exceptions -fno-rtti` and the full warning set. A header that relies on a transitive include, throws, or uses RTTI fails the build.
* **Why not C++20 named modules.** Module support needs CMake 3.28+ with the Ninja generator, and dependency scanning, which AppleClang has supported unevenly. pybind11 and GoogleTest are not modular, so the build would have to mix modules and headers. Since the engine is templated on the model shape, it must stay header-visible anyway. Every benefit modules would give here (no leaking includes, explicit dependencies) is delivered by the two checks above.

---

## 3. Build system (built)

### 3.1 Targets

| Target | Kind | Contents |
|---|---|---|
| `pulselob::pulselob` | INTERFACE | public headers, C++20, FP policy (Section 4.3) |
| `pulselob::warnings` | INTERFACE | warning set for PulseLOB's own code only |
| `pulselob_header_check` | OBJECT | one TU per public header, `-fno-exceptions -fno-rtti` |
| `pulselob_tests` | executable | GoogleTest unit tests, registered with CTest |
| `pulselob_bench` | executable | Google Benchmark micro-benchmarks |
| `_core` | Python module | pybind11 extension, installed as `pulselob/_core` |

### 3.2 Options and presets

| Option | Default | Meaning |
|---|---|---|
| `PULSELOB_BUILD_TESTS` | ON | GoogleTest unit tests |
| `PULSELOB_BUILD_BENCHMARKS` | OFF | Google Benchmark |
| `PULSELOB_BUILD_PYTHON` | OFF | pybind11 module (ON under pip) |
| `PULSELOB_WARNINGS_AS_ERRORS` | OFF | `-Werror` (ON in CI) |
| `PULSELOB_NATIVE_ARCH` | OFF | `-march=native` or `-mcpu=native`; benchmarks only |
| `PULSELOB_SANITIZERS` | empty | `address;undefined` or `thread` |

The presets are `debug`, `asan` (Debug with AddressSanitizer and UBSan), `tsan` (RelWithDebInfo with ThreadSanitizer), `release` and `bench` (Release, benchmarks, native tuning). Usage: `cmake --preset release && cmake --build --preset release && ctest --preset release`.

### 3.3 Pinned dependencies

Each dependency is fetched only when its option is on, by URL and SHA-256. The core has no dependencies.

| Dependency | Version | SHA-256 of the GitHub tag archive |
|---|---|---|
| GoogleTest | 1.15.2 | `7b42b4d6ed48810c5362c265a17faebe90dc2373c885e5216439d37927f02926` |
| Google Benchmark | 1.9.1 | `32131c08ee31eeff2c8968d7e874f3cb648034377dfc32a4c377fa8796d84981` |
| pybind11 | 2.13.6 | `e08cb87f4773da97fa7b5f035de8763abc656d87d5773e62f6da0587d1f0ec20` (pip builds use the `pybind11` build requirement instead) |

Third-party code is included with `SYSTEM`, so the project's warning set and `-Werror` never apply to it.

### 3.4 Compiler and platform matrix

| CI job | OS | Compiler | Configuration |
|---|---|---|---|
| C++ gcc-13 release | Ubuntu 24.04 x86-64 | GCC 13 | Release, `-Werror` |
| C++ clang-18 asan+ubsan | Ubuntu 24.04 x86-64 | Clang 18 | Debug, AddressSanitizer + UBSan |
| C++ clang-18 tsan | Ubuntu 24.04 x86-64 | Clang 18 | RelWithDebInfo, ThreadSanitizer |
| C++ appleclang arm64 release | macOS, latest GitHub arm64 image | AppleClang | Release, `-Werror` |
| Benchmarks compile | Ubuntu 24.04 | GCC 13 | bench preset (built, not run: CI timings are not meaningful) |
| Python | Ubuntu (3.9, 3.12), macOS latest arm64 (3.12) | platform default | `pip install .[test]` (module and `pulselob-replay`), pytest: bindings parity, generator statistics, recorder (mock and local WebSocket server; never Binance), archive loader |

The CMake minimum is 3.25 (presets v6, `FetchContent ... SYSTEM`). Development was done with CMake 4.3 and AppleClang 21 on an Apple M4.

---

## 4. Configuration and numerical policy

### 4.1 Compile-time versus runtime configuration

| Compile-time (`ModelShape`) | Runtime (`Config`, validated at construction) |
|---|---|
| channel count $D$ | decay rates $\beta_q$ |
| kernel orders $r_q$ (hence $Q$, $P$, $M+1$) | forgetting half-lives, dwell, CUSUM $\varrho$ and $b_{\mathrm{cs}}$ |
| | ridge $T_\omega$, heartbeat $T_{\mathrm{hb}}$, prior $\boldsymbol\Theta_0$ |
| | $\rho_{\max}$, $\chi$, $\epsilon_v$, $\rho_{\mathrm{crit}}$, $\mu_{\min}$, $\mu_{\max}$ |
| | mark coefficients and caps, normalizer rates |
| | forecast horizon $H$, latency $\mathcal L_{\mathrm{rt}}$, control parameters |
| | channel map (instrument, side) to channel; impact weights $\eta_i$, $J^b_i$, $J^I_i$ |

`DefaultShape = ModelShape<4, 1, 2, 2, 1>` gives $P=17$ and $M+1=7$, matching spec Section 3.1. In Phase 3 the Python binding will instantiate `DefaultShape` and the $Q=3$ all-exponential fallback `ModelShape<4, 1, 1, 1>` ($P=13$), and select between them by name. Erlang orders above 2 are rejected at compile time: they would need $\psi_n$ with $n>3$, which is outside the range validated in Phase 1.

### 4.2 Construction-time validation (Phase 3)

`Config::validate()` rejects any configuration that breaks an assumption of the spec:

* $\kappa_{\max}<\beta_{\min}$ (I2);
* slowest kernel mean $\times10\le$ slow half-life (I3, warning only);
* $\rho_{\max}\chi\bar g<1$ (S4);
* $H\ge\mathcal L_{\mathrm{rt}}$ (Section 7.3);
* $0<\mu_{\min}<\mu_{\max}$; $c_\iota<1$; all rates positive and finite;
* every instrument in the channel map present in the instrument table.

As built, no PulseLOB code throws: every public header compiles with `-fno-exceptions`. `engine::validate(config)` returns `nullptr` or a static string naming the first violated condition (it checks every component's `validate` and their consistency), and the engine's constructor has the precondition that validation passed. The Python binding will turn a non-null result into `ValueError` (Phase 4).

### 4.3 Floating point

* **No fast-math.** The root `CMakeLists.txt` refuses to configure if the flags contain `-ffast-math` or `-Ofast`. These would break the NaN guards and the reassociation-sensitive 1e-10 equivalence tests.
* **No FMA contraction:** `-ffp-contract=off` on every consumer of the core. GCC contracts across statements by default and Clang within expressions, so allowing it would make the same source produce different bits on different compilers. Phase 3 benchmarks will report the cost of this, and a contraction-enabled build is a benchmark-only option.
* **Reproducibility guarantee.** Same platform, same standard math library, same input bytes and same configuration give bitwise-identical output, in single-threaded mode, at any optimization level. Across platforms (for example glibc against Apple libm) `exp` may differ by an ulp, so results agree to rounding, not bitwise.
* **Denormals.** Flush-to-zero is set by an RAII guard on the engine thread: MXCSR FTZ and DAZ on x86-64, FPCR.FZ on AArch64. The Python binding sets it for the duration of a run and restores the interpreter's FP environment afterwards. Phase 3 tests run with and without the guard and require equal results, since the spec's cutoff ($X_{\mathrm{cut}}=50$) keeps denormals out of every meaningful computation.
* **Time.** Differences of `int64` nanosecond stamps convert to seconds by division, not by multiplying by `1e-9` (which is inexact), so the conversion is correctly rounded. **(built: `core/time.hpp`, tested)**

---

## 5. Memory layout

### 5.1 Principles, and SoA versus AoS

| Data | Layout | Reason |
|---|---|---|
| `MarketEvent` in rings, NumPy arrays and `.plev` files | AoS, 64-byte records | consumed whole, one record per event; one layout for all transports means zero copies |
| Model state ($R$, $\boldsymbol\Theta$, $\mathbf U$, $\mathbf C$, vectors) | SoA, padded rows | the recursions sweep one quantity across all channels or coordinates (for example "multiply row $(q,n)$ of $R$ by $e_q$ for all $j$"), which vectorizes only if those values are contiguous |
| `Signal` output | AoS, 128-byte records (for $D=4$) | one record per event for NumPy |
| Cross-thread fields (ring counters, seqlock) | each on its own 128-byte block | no false sharing |

### 5.2 Fixed-size containers (built: `core/fixed_matrix.hpp`)

`FixedMatrix<R, C>` and `FixedVector<N>` are `alignas(64)` structs holding a `std::array<double, ...>`.

* Every row starts on a 64-byte boundary.
* Rows are padded to `stride = round_up(C, 8)` doubles.
* **Padding lanes are zero** at construction, and every kernel must keep them zero. Full-width SIMD loops over `stride` lanes are then exact, because padded products contribute $0\cdot0=0$.
* Nothing allocates. Unit tests check the sizes, the row alignment and the padding invariant.

Eigen was considered and rejected for this role:

* **Padding.** With $P=17$, an Eigen fixed-size matrix neither pads rows nor aligns them for SIMD.
* **Over-alignment pitfalls.** Fixed-size vectorizable Eigen types carry known pitfalls when stored in structs that are themselves heap-allocated.
* **Small surface.** The estimator needs only five kernels: row-wise axpy, matrix-vector product, transposed matrix-vector product, rank-one update and scaling. They are written once in Phase 3 and tested against long-double references.

### 5.3 Engine state for `DefaultShape` (built)

The Phase 2 plan budgeted about 12 KB. The engine as built is larger, for two measured reasons: the moment matrix $\mathbf M$ is stored (for the long-gap refactorization and the information-form estimator, math_spec 5.4.1), and the engine owns a copy of its configuration. Sizes from `sizeof` on AppleClang 21, arm64; `tests/cpp/test_engine.cpp` asserts the totals stay within a 24 KB (information form) and 28 KB (Potter) L1 budget.

| Component | Contents | Bytes |
|---|---|---|
| `hawkes::KernelState` | $R^{q,(n)}_j$ (6 rows × 4 channels, padded) and the pending tie buffer $\boldsymbol\pi$ | 448 |
| `hawkes::IntervalTerms` | per-interval decay factors, compensator coefficients, $\mathbf F$ (7 × 7) | 576 |
| `hawkes::LsStatistics` | $\mathbf M$ (17 × 17), $\mathbf C$, $\boldsymbol\Theta_0$, $\boldsymbol\omega$, $\boldsymbol\omega^{\mathrm{eff}}$, interval scratch $\mathbf V$ and $\mathbf F\mathbf V$ | 7,872 |
| `hawkes::CholeskyEstimator` | statistics, $\tilde{\boldsymbol\Theta}$, factorization scratch | 12,928 |
| `hawkes::PotterEstimator` | statistics, $\mathbf U$, $\tilde{\boldsymbol\Theta}$, rank-one and refactor scratch | 16,704 |
| `hawkes::RegimeDetector` | CUSUM statistics, residual accumulators, forgetting schedule | 320 |
| `hawkes::Served` | $\hat{\boldsymbol\Theta}$, $\hat{\mathbf G}$, $\boldsymbol\lambda^\infty$, $\mathbf v$, diagnostics (the seqlock payload in split mode) | 1,216 |
| `hawkes::Model` (information form) | the above plus regressor and compensator scratch | 16,256 |
| `engine::DecisionLayer` | forecaster constants, quote law, triggers, kill switch, throttles, venue clocks | 1,216 |
| `engine::EngineConfig` | the configuration copy (cold) | 1,792 |
| **`engine::Engine` (information form)** | everything, including mapper, marks and the current `Signal` | **20,992** |
| `engine::Engine` (Potter) | | 24,768 |

This fits L1D on every target: 128 KB per performance core on the development M4 (`hw.perflevel0.l1dcachesize`) and 48 KB on recent x86 cores. Hot data (kernel state, served parameters, decision layer) come first in the object; the statistics follow; the configuration copy is cold. In split mode the estimator-side `Model` lives on the estimator thread, and the hot thread's working set is the kernel state, the served snapshot and the decision layer, about 3 KB.

### 5.4 The zero-allocation contract

* **Fixed size.** Every engine member is a fixed-size container, so construction performs one allocation (the engine object, via aligned `new`) or none (static or stack storage).
* **Large rings.** These are allocated once at start-up through `std::make_unique`, which honors over-alignment.
* **Enforcement (built).** `tests/cpp/alloc/test_zero_alloc.cpp` is its own binary that replaces every global `operator new` and `operator delete` (plain, array, aligned, nothrow, sized) with counting versions. After construction it replays $10^6$ market events (2 × 10^5 in debug and sanitizer builds) plus 1,000 heartbeats through the inline engine with each estimator back-end, and a stream through the split engine with both threads running, and asserts that zero allocations occurred. It runs in every CI configuration, including ASan and TSan.

---

## 6. Concurrency

### 6.1 Topology

```mermaid
flowchart LR
  subgraph "feed threads (one per venue, Phase 4)"
    A["perp WebSocket decoder"]
    B["spot WebSocket decoder"]
  end
  A -->|"SPSC ring of MarketEvent, capacity 2^16"| E
  B -->|"SPSC ring of MarketEvent, capacity 2^16"| E
  subgraph "engine thread (pinned, busy-polling)"
    E["merge by ts_local, then engine pipeline"]
  end
  E -.->|"split mode only: events"| X["estimator thread"]
  X -.->|"seqlock: served parameters"| E
  E --> O["Signal to gateway or simulator"]
```

| Mode | Threads | Deterministic | Used for |
|---|---|---|---|
| **Inline (default)** | one engine thread; the estimator runs inside each event | yes, bitwise | all tests, replays, backtests, the Phase 5 simulator |
| Split (built) | engine thread for state, inference and control; estimator thread fed by a second SPSC ring, publishing $\hat{\boldsymbol\Theta}$ by seqlock | no: parameter freshness depends on timing | live quoting when the latency budget is below about 1 µs |

**Measured (Phase 3, [benchmarks.md](benchmarks.md)).** On the M4, inline tick-to-quote for a trade has a p50 of 0.99 µs and a p99 of 1.1–1.3 µs, at 1.8 M events/s. The split hot thread has a p50 of 0.16 µs and a p99 of 0.2–0.4 µs, at 8.4 M events/s, with parameters lagging by a median of one estimator step. The estimator thread alone sustains 1.15 M steps/s. After `drain()` the split engine's served parameters equal the inline engine's bit for bit (tested), because the estimator thread runs the same model on the same steps.

Split mode is safe because parameters move on a scale of seconds (spec Section 6.6). A served parameter set that is a few microseconds stale has no measurable effect. As built, the estimator publishes whenever its ring is drained and at least every 32 steps during a backlog. The hot thread checks the sequence number with one load per event and copies the parameters (1,216 bytes) only when that number has changed. Signals computed with parameters more than 64 steps old carry the `kStaleParameters` flag.

### 6.2 SPSC ring buffer (built: `feed/spsc_ring.hpp`)

* **Layout.** The head counter (written by the producer), the producer's cached copy of the tail, the tail counter (written by the consumer), the consumer's cached copy of the head, and the slot array each start on their own 128-byte block. Capacity is a compile-time power of two. Counters are monotonically increasing `uint64` values, which cannot wrap in 584 years at $10^9$ events per second, and the slot index is `counter & (capacity - 1)`.
* **Algorithm.** Push: load the head (relaxed). If the ring looks full from the cached tail, reload the tail (acquire). If it is still full, return false. Otherwise write the slot and store the head + 1 (release). Pop is symmetric. `pop_batch` drains up to $n$ slots with a single release store.
* **Correctness.** The producer's release store of the head, paired with the consumer's acquire load, makes the slot write happen before the slot read. The consumer's release store of the tail, paired with the producer's acquire load, makes the slot read happen before the slot is overwritten. With the cached counters, a push or pop in steady state touches no cache line written by the other core.
* **Tests.** Full and empty detection across several counter wrap-arounds, batch pops across the wrap point, alignment, and a two-thread stress test of $2\times10^6$ events that checks order and absence of loss. The stress test runs under ThreadSanitizer in CI.
* **Capacity.** $2^{16}$ events (4 MiB) per venue. That absorbs 0.65 s of a $10^5$ events/s burst, which comfortably exceeds an OS preemption stall on a non-isolated core.
* **Overflow policy.** The producer never blocks. When the ring is full, it counts the dropped event and marks an overflow. On its next successful push it first inserts a `Status` event with code `RingOverflow`. The engine then raises the feed-health kill switch (spec Section 8.6), because a model that silently lost events is not trustworthy.
* **Measured** (Apple M4, AppleClang 21, Release, native tuning, median of 5): push plus pop on one thread costs 7.8 ns. That is the ring's own overhead. The cross-core cost needs pinned threads and is measured in Phase 3.

### 6.3 Seqlock parameter publication (built: `core/seqlock.hpp`)

A plain seqlock reads non-atomic data during a concurrent write, which is a data race and undefined behavior in the C++ memory model. The construction used here, following Boehm (2012), avoids that:

```text
writer:  s = seq.load(relaxed); seq.store(s + 1, relaxed); atomic_thread_fence(release);
         for each double x: atomic_ref(x).store(value, relaxed);
         seq.store(s + 2, release);
reader:  do { s1 = seq.load(acquire); if (s1 is odd) continue;
              for each double x: copy = atomic_ref(x).load(relaxed);
              atomic_thread_fence(acquire); s2 = seq.load(relaxed);
         } while (s1 != s2);
```

`std::atomic_ref<std::uint64_t>` (C++20) over the payload's words makes every payload access atomic, so there is no race; ThreadSanitizer runs the torn-read test and the split-engine test in CI. The fences give the reader a consistent snapshot whenever `s1 == s2` and `s1` is even. The sequence number sits on its own 128-byte block.

### 6.4 Thread placement

* **Linux.** The engine and feed threads are pinned to isolated cores (`isolcpus`, `nohz_full`) and busy-poll.
* **macOS.** It has no hard affinity API, so threads get the user-interactive QoS class, and macOS latency numbers are reported as indicative only.
* **Denormals.** Every engine thread sets flush-to-zero at start-up (Section 4.3).

---

## 7. Data schemas

### 7.1 Units and encodings

| Quantity | Encoding |
|---|---|
| Absolute time | `int64` nanoseconds since the Unix epoch |
| Time differences | `int64` nanoseconds, converted to double seconds only as a difference |
| Price, quantity, notional | `int64` fixed point, scale $10^{-8}$ ("e8"); exact for Binance, range $\pm9.2\times10^{10}$ units |
| Enumerations | `uint8` (`EventKind`, `Side`, `TsResolution`), `uint32` (`StatusCode`) |
| Byte order | little-endian (checked at compile time in the binding) |

Decimal strings are parsed into e8 exactly by `core::parse_decimal_e8` **(built, tested on the Binance sample strings)**. It rejects malformed input, overflow, and any nonzero digit beyond the 8th decimal. It never passes through a float, so a 16-significant-digit quantity such as `12345678.12345678` survives unchanged.

### 7.2 `MarketEvent`, 64 bytes (built: `feed/market_event.hpp`)

| Offset | Size | Type | Field | Trade | BookTop |
|---|---|---|---|---|---|
| 0 | 8 | int64 | `ts_exchange_ns` | matching-engine time | matching-engine time |
| 8 | 8 | int64 | `ts_local_ns` | time available to us | time available to us |
| 16 | 8 | int64 | `px0_e8` | last (worst) price of the sweep | best bid price |
| 24 | 8 | int64 | `qty0_e8` | total quantity | best bid quantity |
| 32 | 8 | int64 | `px1_e8` | first (best) price of the sweep | best ask price |
| 40 | 8 | int64 | `qty1_e8` | notional in quote units | best ask quantity |
| 48 | 8 | uint64 | `seq` | first trade id | book update id |
| 56 | 4 | uint32 | `count` | prints merged, at least 1 | 0 (Status: `StatusCode`) |
| 60 | 2 | uint16 | `instrument` | index into the instrument table | same |
| 62 | 1 | uint8 | `kind` | `Trade` = 1 | `BookTop` = 2 (Heartbeat 0, Status 3) |
| 63 | 1 | uint8 | `flags` | see below | see below |

`flags`:

* bits 0–1: aggressor side (0 none, 1 buy, 2 sell);
* bits 2–3: resolution of the exchange timestamp (0 ns, 1 µs, 2 ms, 3 s);
* bit 4: timestamp adjusted by the normalizer;
* bit 5: synthetic;
* bit 6: snapshot;
* bit 7: reserved, must be 0.

The record is trivially copyable, standard layout, `sizeof == 64` and `alignof == 8`, and every field offset is asserted at compile time. Natural (8-byte) alignment rather than `alignas(64)` is deliberate: NumPy buffers are only guaranteed natural alignment, and reinterpreting one as an over-aligned type would be undefined behavior. Rings place records on 64-byte boundaries through their slot array instead.

The validator `feed::validate` is **(built)** and costs 1.75 ns per event as measured above. It checks:

* the reserved bit is 0;
* the kind is known;
* trades have a buy or sell aggressor, and quotes have none;
* prices and quantities are positive;
* sweep direction is consistent: a buy sweep's last price is at least its first, and a sell sweep's at most;
* quotes are not crossed or locked;
* heartbeats and status events have zero payload.

This implements step 1 of the spec's per-event algorithm.

### 7.3 Instrument table (JSON sidecar)

Each instrument has:

* `id` (`uint16`, matching `MarketEvent::instrument`);
* `venue` (for example `binance-usdm`, `binance-spot`) and `symbol`;
* `tick_size_e8` and `lot_size_e8`;
* `contract_multiplier`;
* `maker_fee_bps` and `taker_fee_bps`;
* `role` (`lead` or `quote`).

The table travels with every dataset as `<file>.plev.json`, together with provenance: source URLs, the SHA-256 of each source file, and the normalizer version.

### 7.4 Binance sources (verified against downloaded samples)

The dataset inventory comes from the public bucket listing of data.binance.vision. The formats come from one daily file each, downloaded on 2026-11-20.

| Dataset | Available | Header row | Columns | Timestamp |
|---|---|---|---|---|
| Spot `trades` | yes, since 2017 | **no** | id, price, qty, quoteQty, time, isBuyerMaker (`True`/`False`), isBestMatch | **ms** (2024-03-01 sample), **µs** (2025-06-01 sample) |
| Spot `bookTicker` | **not published** | – | – | – |
| USD-M `trades` | yes | yes | id, price, qty, quote_qty, time, is_buyer_maker (`true`/`false`) | ms |
| USD-M `bookTicker` | **2023-05-16 to 2024-03-30 only** (BTCUSDT) | yes | update_id, best_bid_price, best_bid_qty, best_ask_price, best_ask_qty, transaction_time, event_time | ms (both) |

Mapping to `MarketEvent`:

* **Aggressor:** `isBuyerMaker = true` means the aggressor sold (`Side::Sell`).
* **Sweeps:** consecutive prints with equal instrument, aggressor and exchange time are merged. The USD-M sample already shows multiple prints at the same millisecond. `px1` is the first price, `px0` the last, `qty0` the summed quantity, `qty1` the summed notional, `count` the number of prints, and `seq` the first trade id.
* **Timestamp units:** detected per file by magnitude: below $10^{14}$ is ms, below $10^{17}$ is µs, otherwise ns. The detected unit is stored in the flags.
* **Two clocks:** `ts_exchange_ns` is `time`, or `transaction_time` for `bookTicker`. `ts_local_ns` is `event_time` plus a modeled network latency for `bookTicker`, and the exchange time plus modeled latency for trades. The latency model is a per-venue configuration in Phases 4 and 5.

**The engine clock is `ts_local_ns`.** The merged stream is ordered by availability time, with ties broken by instrument and then `seq`. A backtest therefore never acts on an event before it could have arrived. This is one of the naive-backtest biases Phase 5 must avoid. Research runs that want to study market dynamics on exchange time can select `ts_exchange_ns` explicitly.

**Resolution caveat.** Millisecond timestamps make cross-venue ties common, and Section 4.4 of the spec handles them with batch semantics. They also leave the 1 ms fast kernel close to the resolution limit, so Phase 4 will re-check the default kernel bank against the data's resolution.

**Spot top-of-book: decision needed before Phase 4.** The spec's imbalance mark $\iota$ for spot channels, and the simulator's queue-position model on spot (the quoting venue chosen in Phase 1), both need the spot best bid and offer. The options are:

| Option | Realism | Cost | Time to data |
|---|---|---|---|
| (a) Record live spot and perp `bookTicker` plus trades with a Phase 4 recorder | full, current market | free | days to weeks of recording |
| (b) Licensed historical data from a vendor with Binance spot top-of-book | full, long history | paid | immediate |
| (c) 2023-05 to 2024-03 window: perp book plus spot trades, spot $\iota=0$, simulate quoting on perp | partial: spot book and queue modeling are unavailable | free | immediate |

The recommendation is (a), started early in Phase 4, with (c) as the immediately available validation set and the synthetic generator as the main controlled experiment. Option (b) is worth it if a long real-data history becomes a requirement for the paper.

### 7.5 Channel events (built)

`feed::EventMapper` turns `Trade` events into raw channel features (channel $d_k$, size, sweep depth $L=\lvert\texttt{px0}-\texttt{px1}\rvert/\text{tick}$, and $\iota=s_{d_k}\,\mathrm{OBI}$) using the configured map from (instrument, side) to channel and the last `BookTop` of the trade's instrument. A missing book gives $\iota=0$ and is counted. The mark normalization (log-median tracker and EWMA of math_spec 2.3) lives in `hawkes::MarkNormalizer`, because it is model math; the engine computes the mark with `peek` and applies the statistics update with `commit` only if the model accepts the step. `BookTop` events update the book and produce a read-only signal. `Heartbeat` events step the model. `Status` events trip the kill switch. These records never cross a module boundary.

### 7.6 `Signal`, output record (built: `engine/signal.hpp`)

Size is `round_up(16 + 8D + 72, 64)` bytes, which is 128 for $D=4$.

| Offset ($D=4$) | Type | Field |
|---|---|---|
| 0 | int64 | `t_ns` (engine clock of the event that produced it) |
| 8 | uint32 | `flags`: warming up, ask pulled, bid pulled, kill switch, fast forgetting, projection active, no quote book, stale parameters (split mode) |
| 12 | uint16 | `status` (input `EventCheck` or engine status) |
| 14 | uint16 | reserved |
| 16 | double[4] | `lambda` (served intensities at $t+$) |
| 48 | double | `tau_buy` |
| 56 | double | `tau_sell` |
| 64 | double | `criticality` ($\tilde\rho$) |
| 72 | double | `reservation_shift` ($\hat\Delta$) |
| 80 | double | `adverse_ask` ($\varepsilon^+$) |
| 88 | double | `adverse_bid` ($\varepsilon^-$) |
| 96 | double | `sigma2` |
| 104 | int64 | `bid_px_e8` (rounded down to tick; 0 means not quoted) |
| 112 | int64 | `ask_px_e8` (rounded up to tick; 0 means not quoted) |
| 120 | – | padding to 128 |

### 7.7 `.plev` event files (built: `python/pulselob/plev.py`)

A `.plev` file is a 64-byte header followed by `n` raw `MarketEvent` records starting at offset 64.

| Offset | Size | Field |
|---|---|---|
| 0 | 4 | magic `PLEV` |
| 4 | 2 | schema version |
| 6 | 2 | record size (64) |
| 8 | 4 | header size (64) |
| 16 | 8 | number of records |
| 24 | 8 | creation time (Unix ns) |
| 12, 32 | 4, 32 | reserved, zero |

`open_plev` returns a read-only `numpy.memmap`, which the native code consumes without a copy (tested). The reader checks the magic, schema version, record size and exact file size. The sidecar holds the instrument table and provenance.

### 7.8 Schema versioning

* `kMarketEventSchemaVersion` (currently 1) changes whenever any byte of the layout or the meaning of a field changes.
* The Python mirror in `schema.py` is written independently and compared field by field with the layout the compiled module reports, **at import time**. A C++ change without the matching Python change, or the reverse, makes `import pulselob` fail.
* `.plev` readers reject other schema versions instead of guessing.
* The schema version is separate from the project version: a minor release may leave the schema unchanged.

### 7.9 Zero-copy contract for NumPy

The native code accepts an array as events if and only if it is 1-D, has itemsize 64 (the `MARKET_EVENT_DTYPE`), is C-contiguous, and is 8-byte aligned. Otherwise it raises `ValueError` with the reason (tested). The buffer is then read in place with the GIL released. Reading `malloc`'ed NumPy memory as `MarketEvent` objects relies on C++20 implicit object creation for implicit-lifetime types (P0593). For memory-mapped files the strictly conforming tool is C++23's `std::start_lifetime_as`, which will be used once the toolchain floor allows it. Every major compiler already treats both cases as defined.

---

## 8. Errors and observability

* **No logging on the hot path.** Every guard of the spec increments a named counter: rejected events by `EventCheck` code, clamped and rejected out-of-order events, cutoff activations, projection activations, CUSUM alarms, kill-switch causes and ring overflows. Counters live in the cold block and are exposed to Python.
* **Status, not exceptions.** `on_event` returns a status (accepted, clamped, rejected with a reason), and the same code is copied into `Signal::status`.
* **Snapshot and restore (built).** The engine is trivially copyable (asserted at compile time). A snapshot is a copy, and the tests check that a restored copy continues bit for bit like the uninterrupted run. State comparisons in tests are field by field: the 64-byte-aligned types contain padding, whose bytes are indeterminate, so `memcmp` over whole objects is not a valid equality test.

Engine interface (built: `engine/engine.hpp`, `engine/split_engine.hpp`):

```cpp
template <class Shape>
const char* validate(const EngineConfig<Shape>& config) noexcept;  // nullptr or the violated condition

template <class Shape, class Estimator = hawkes::CholeskyEstimator<Shape>>
class Engine {  // deterministic, single-threaded, trivially copyable
 public:
  explicit Engine(const EngineConfig<Shape>& config) noexcept;  // precondition: validate() == nullptr
  Status on_event(const feed::MarketEvent& e) noexcept;          // math_spec Section 9
  Status on_heartbeat(core::TimestampNs t) noexcept;
  void set_inventory(double q) noexcept;                         // non-finite: ignored, counted, kill switch
  const Signal<Shape::D>& signal() const noexcept;
  const EngineCounters& counters() const noexcept;
  const hawkes::Model<Shape, Estimator>& model() const noexcept;
};

template <class Shape, class Estimator = hawkes::CholeskyEstimator<Shape>>
class SplitEngine;  // same on_event; estimator thread fed by an SPSC ring, parameters by seqlock

template <class EngineT, class Sink>  // Sink: noexcept on_signal(const Signal&)
std::size_t replay(EngineT&, std::span<const feed::MarketEvent>, Sink&) noexcept;
```

`Status` is `Accepted`, `Clamped` (out of order within $\varepsilon_{\mathrm{ooo}}$), `RejectedInvalid`, `RejectedOutOfOrder`, `Ignored` (unmapped instrument or side) or `RejectedNonFinite`. Rejections leave every model-side state unchanged (tested field by field).

---

## 9. Testing and benchmarking architecture

### 9.1 Test layers

| Layer | Framework | Location | CI | Phase |
|---|---|---|---|---|
| Layout and schema (sizes, offsets, alignment, padding, validation) | GoogleTest + `static_assert` | `tests/cpp` | all C++ jobs | 2 (built) |
| Concurrency (SPSC) | GoogleTest under ThreadSanitizer | `tests/cpp` | tsan job | 2 (built) |
| Python schema mirror, zero-copy, `.plev` | pytest | `tests/python` | python jobs | 2 (built) |
| Equivalence V1–V9 (recursions against brute force, 1e-10) | GoogleTest with direct-sum and quadrature references, and a 50-digit table for $\psi$ | `tests/cpp` | all C++ jobs | 3 (built) |
| Whole-pipeline cross-check against a definition-level Python reference | GoogleTest reading `tests/fixtures/phase3_crosscheck.txt`; a CI job regenerates the fixture | `tests/cpp`, `research/phase3` | all C++ jobs, reference-fixture | 3 (built) |
| Statistical V10 (regime-shift recovery, false-alarm rate) | GoogleTest, seeded | `tests/cpp` | all C++ jobs | 3 (built) |
| Stress V11 (bursts, extreme gaps, NaN/Inf, disorder), determinism, snapshot/restore, FTZ invariance | GoogleTest, typed over both estimator back-ends | `tests/cpp` | all C++ jobs | 3 (built) |
| Zero allocation after construction | separate binary with counting `operator new` | `tests/cpp/alloc` | all C++ jobs | 3 (built) |
| Concurrency (seqlock, split engine) | GoogleTest under ThreadSanitizer | `tests/cpp` | tsan job | 3 (built) |
| Math-spec identities | Python research script | `research/` | math-validation | 1 (built) |

### 9.2 Benchmark methodology (built: `bench/latency_report.cpp`)

* **Service time per event.** Each event is timed with the platform counter (`cntvct_el0` behind an `isb` on AArch64, which runs at 1 GHz on the M4; `rdtscp` on x86) into a fixed log-linear histogram with 32 sub-buckets per octave, which allocates nothing. The mean back-to-back timer cost is subtracted. The report gives p50, p95, p99, p99.9 and the maximum, plus closed-loop throughput measured without timers in the loop.
* **Component breakdown.** A stage-by-stage replica of `Model::step` is timed in situ, and its final state is checked field by field against the real engine, so the breakdown measures the code that runs.
* **Open-loop latency.** Poisson arrivals at $10^5$ and $10^6$ events/s. The replay busy-waits for each arrival and measures from scheduled arrival to signal, so queueing delay is included (no coordinated omission).
* **Environment.** Report CPU, compiler, flags (contraction on and off), and pinning. Run on isolated cores on Linux; mark macOS numbers indicative.

### 9.3 Baseline measurements (Phase 2, real runs; Phase 3 results are in [benchmarks.md](benchmarks.md))

These numbers are from `build/bench/bench/pulselob_bench` on an Apple M4 with AppleClang 21, Release with `-mcpu=native`, as the median of 5 repetitions. They are inputs to the Phase 3 cost model, not engine results.

| Benchmark | Median | What it tells us |
|---|---|---|
| `std::exp`, argument in $[-50,-10^{-6}]$ | 1.99 ns | $Q+1=5$ exponentials per event cost about 10 ns, so a fast-exp approximation is not needed at the default bank size |
| SPSC push + pop, same thread | 7.8 ns | the ring's own overhead per event |
| `validate(MarketEvent)` | 1.75 ns | input validation is negligible |

These are throughput figures for independent calls. They are not latency distributions.

---

## 10. Python package (built)

```text
python/pulselob/
├── __init__.py        public API, native_replay_path()
├── schema.py          MARKET_EVENT_DTYPE, SIGNAL_DTYPE, enums, exact decimal_to_e8, event builders;
│                      both dtypes checked against the native layouts at import
├── plev.py            .plev writer and memory-mapped reader, streaming PlevWriter, recover_plev
├── config.py          EngineConfig: every parameter under the C++ codec's key names; presets for the
│                      default and exp3 banks; to_text() for the native tool
├── engine.py          Engine (run, step, heartbeat, set_inventory, state, counters, drain)
├── synth/             scenarios, generate(), write() with ground truth; pulselob-synth
├── data/recorder.py   live Binance recorder; pulselob-record
├── data/binance_archive.py  data.binance.vision loader; pulselob-archive
└── bench.py           python -m pulselob.bench
```

**Engine bridge.** `pulselob._core.Engine` is one type-erased class over model shape (`default`, `exp3`), estimator (`cholesky`, `potter`) and mode (`inline`, `split`).

* **Configuration.** It arrives as `{key: [values]}`, is applied through `engine/config_io.hpp` (the same codec the native tool and the cross-check test use), and is validated. A violation raises `ValueError` with the condition.
* **`Engine.run(events, state_every=0)`.** The input is a MarketEvent array, a memory-mapped `.plev` file or a path, and it is read in place. Output arrays are allocated before the call: one Signal per event and, if requested, the model state every k-th model step (served and unconstrained parameters, branching matrix, stationary intensities, criticality, effective sample size, CUSUM statistics). The GIL is released for the whole loop, with flush-to-zero set for the duration.
* **`Engine.step(event)`.** The same path for one event, for live loops. In split mode it does not wait for the estimator thread; `drain()` does.

**Native tool.** `pulselob-replay` (`tools/replay.cpp`) is installed in the wheel and built in every C++ configuration. It reads a `.plev` file and the text configuration, and writes the raw Signal records. The Python tests require its output to equal `Engine.run` byte for byte.

**Packaging.** scikit-build-core builds the module and the tool. The `record` extra adds `websockets`. The console scripts are `pulselob-record`, `pulselob-archive` and `pulselob-synth`. The license is MIT.

## 11. Assumptions and edge cases

| Case | Handling |
|---|---|
| Host is big-endian | compile-time error in the binding; the formats are little-endian |
| `size_t` narrower than 64 bits | compile-time error in the ring (counter differences must convert losslessly) |
| NumPy array not contiguous, wrong dtype, misaligned, not 1-D | `ValueError` with the reason; no silent copy inside the binding |
| `.plev` with another schema version, bad magic, or truncated | `PlevFormatError` |
| Decimal with more than 8 nonzero decimals, or out of range | `parse_decimal_e8` returns nothing; the loader counts and rejects the row |
| Spot timestamps changing from ms to µs between files | unit detected per file by magnitude and recorded in the flags |
| Ring full | producer drops, counts, and injects a `RingOverflow` status event; the engine kill switch trips |
| Toolchain without `std::atomic_ref` | split mode unavailable at compile time; inline mode unaffected |
| `-ffast-math` or `-Ofast` in the flags | configuration fails |
| Erlang order above 2 | compile-time error |

---

## 12. Decisions

Phase 2 review (settled): header-only core without named modules; inline engine by default; fixed-point e8; `ts_local_ns` as the engine clock; the Phase 4 data source is a live recorder plus the 2023–2024 archive window, option (a) plus (c) of Section 7.4.

For the Phase 3 review:

1. **The information-form estimator as the default back-end**, with Potter kept as an alternative (math_spec 5.4.1, [benchmarks.md](benchmarks.md) Section 6). This changes the spec 0.1.0 choice, on measured speed and accuracy.
2. **Inline mode stays the default; split mode is for live quoting** below about 1 µs of tick-to-quote budget or above about 5 × 10^5 events/s.
3. **The long-gap rule and the other spec 0.2.0 changes** (math_spec Appendix C).
4. **The CUSUM threshold.** The inherited $b_{\mathrm{cs}}=14$ gives 32 false alarms per hour at $D=4$ on stationary data. Settled for now: interim default 30 (math_spec 0.2.1), with a calibration sweep in Phase 5.
5. **License.** Settled: MIT.

## References

* Boehm, H.-J. (2012). Can seqlocks get along with programming language memory models? *Proceedings of the ACM SIGPLAN Workshop on Memory Systems Performance and Correctness (MSPC)*.
* Smith, R. (2019). *Implicit creation of objects for low-level object manipulation*. WG21 paper P0593R6 (adopted in C++20).
* The data.binance.vision public archive, bucket listing and sample files retrieved 2026-11-20.
