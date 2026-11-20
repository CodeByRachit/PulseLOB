# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/CodeByRachit/PulseLOB/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/CodeByRachit/PulseLOB/releases/tag/v0.1.0
