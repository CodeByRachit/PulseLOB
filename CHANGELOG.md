# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-11-20

Phase 3: the C++20 engine, its verification suite and real measurements.

### Added

- Model engine, header-only, zero allocation after construction, no exceptions or RTTI:
  - `hawkes/kernel.hpp`: exact Erlang-1/2 state recursion with tie batches and the underflow cutoff, psi-form compensators, and the interval Gram matrix of Theorem 7.
  - `hawkes/estimator.hpp`: the discounted least-squares estimator with two back-ends. The information form (stored moment matrix, fresh Cholesky solve per event) is the default; the square-root covariance form (Potter updates) is the spec 0.1.0 design. Both use ridge replenishment and the long-gap full restore.
  - `hawkes/regime.hpp`: the time-rescaling CUSUM with variable forgetting.
  - `hawkes/stability.hpp`: Perron-vector tracking, the exact Collatz–Wielandt row projection, the criticality index and the stationary intensity.
  - `hawkes/forecast.hpp`: horizon forecasts, stationary references, the directional toxicity score and the excess-cascade vector.
  - `hawkes/marks.hpp`, `feed/event_mapper.hpp`: marks and their online normalization; the mapping from trades and book updates to channel features.
  - `control/quoting.hpp`: the Hawkes-adjusted Avellaneda–Stoikov quotes with the Glosten–Milgrom premium, derived pull thresholds, Schmitt triggers with cooldown, the kill switch, requote throttling and tick rounding.
  - `engine/engine.hpp`: the deterministic inline engine and the replay driver. `engine/split_engine.hpp`: the estimator on a second thread, fed through the SPSC ring, with parameters published through `core/seqlock.hpp`.
  - `core/psi.hpp`, `core/linalg.hpp`, `core/fp_env.hpp`: psi functions, deterministic fixed-order linear-algebra kernels, and the flush-to-zero guard.
- Verification: about 60 new GoogleTest cases covering math_spec V1–V11. These include a definition-level Python reference of the whole pipeline (`research/phase3/crosscheck_reference.py`) whose fixture the C++ engine must match, a zero-allocation proof binary, a regime-shift recovery test, stress tests and determinism tests. All run under GCC 13, Clang 18 (ASan+UBSan, TSan) and AppleClang.
- Benchmarks: a Google Benchmark suite and `pulselob_latency`, which reports tick-to-quote distributions, an in-situ component breakdown, the inline-versus-split experiment and open-loop latency. Results are in `docs/benchmarks.md`.
- CI: a job that regenerates the reference fixture and compares it with the committed one.

### Changed

- `docs/math_spec.md` is now spec version 0.2.0 (see its Appendix C). It adds the long-gap rule, heartbeat replenishment, an overflow-free form of one Gram entry, the iteration-free projection, the CUSUM batch rule, E[J^2], and the information-form back-end. It also fixes the exponent in the general stationary-forecast formula.
- `docs/architecture.md` is now document version 0.3.0: memory layout and interface as built, measured threading decision.

## [0.2.0] - 2026-11-20

Phase 2: repository architecture, build system, memory layout and data schemas.

### Added

- `docs/architecture.md`: component boundaries and layering, build system, configuration and floating-point policy, memory layout with per-field byte budgets, threading model (inline or split estimator, SPSC rings, seqlock), data schemas verified against Binance archive samples, `.plev` file format, testing and benchmark methodology.
- CMake build (minimum 3.25) with presets `debug`, `asan`, `tsan`, `release` and `bench`. Dependencies are pinned by SHA-256 (GoogleTest 1.15.2, Google Benchmark 1.9.1, pybind11 2.13.6). Strict warnings, `-ffp-contract=off`, and a refusal to configure with fast-math.
- Header contract: every public header compiles standalone with `-fno-exceptions -fno-rtti`, and `scripts/check_layering.py` enforces component dependencies.
- C++ headers: `core/platform.hpp`, `core/time.hpp`, `core/fixed_point.hpp` (exact e8 decimal parsing), `core/fixed_matrix.hpp` (64-byte-aligned padded containers), `core/model_shape.hpp`, `feed/market_event.hpp` (64-byte `MarketEvent` schema v1 with validation), `feed/spsc_ring.hpp` (wait-free SPSC ring).
- pybind11 module `pulselob._core` (version, schema layout, zero-copy `validate_events`) and Python package `pulselob` (NumPy dtype mirror checked at import, `.plev` writer and memory-mapped reader), built with scikit-build-core from `pyproject.toml`.
- Tests: 21 GoogleTest cases (also run under ASan+UBSan and TSan) and 7 pytest cases. Baseline micro-benchmarks for `std::exp`, the SPSC ring and event validation.
- CI jobs: C++ on GCC 13, Clang 18 (ASan+UBSan, TSan) and AppleClang arm64; benchmark compile; Python 3.9 and 3.12 on Linux and macOS; clang-format; include layering.

## [0.1.0] - 2026-11-20

Phase 1: mathematical research and formulation.

### Added

- `docs/math_spec.md` (spec version 0.1.0), the mathematical specification. It covers:
  - a marked multivariate Hawkes model with an Erlang kernel bank for cross-venue lead-lag, and bounded log-concave marks;
  - exact constant-time state recursions and compensators in series-protected form, with tie, out-of-order, underflow and timestamp-precision semantics;
  - an online discounted least-squares estimator that provably equals its batch counterpart, using low-rank interval updates, square-root inverse updates and ridge replenishment to prevent wind-up;
  - time-rescaling CUSUM for adaptive forgetting;
  - a Collatz–Wielandt sub-criticality certificate with an exact row projection, and a non-explosion theorem;
  - a directional toxicity score and Avellaneda–Stoikov quotes re-derived with adverse-selection jumps, a derived quote-pull threshold, and bounds on message rate.
- `research/phase1/validate_math_spec.py`, numerical verification of the spec's claims plus the estimator-comparison experiments.
- CI: markdown lint, offline link check, version consistency, and math validation. A tag-driven release workflow.

[Unreleased]: https://github.com/CodeByRachit/PulseLOB/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/CodeByRachit/PulseLOB/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/CodeByRachit/PulseLOB/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/CodeByRachit/PulseLOB/releases/tag/v0.1.0
