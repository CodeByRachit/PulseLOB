# PulseLOB

**Real-time cross-venue order-flow toxicity detection with an online, non-stationary, marked multivariate Hawkes process.** The planned engine is a C++20 core with zero allocations on the hot path and Python bindings. It will be validated inside an Avellaneda–Stoikov market-making simulator with realistic latency and queue modeling.

[![CI](https://github.com/CodeByRachit/PulseLOB/actions/workflows/ci.yml/badge.svg)](https://github.com/CodeByRachit/PulseLOB/actions/workflows/ci.yml)

> **Status: Phase 1 (mathematical specification), awaiting review.** There is no engine code yet. The project is built in reviewed phases, and each phase ships as a tagged version (see [Roadmap](#roadmap)).

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

## Roadmap

| Phase | Deliverable | Version | Status |
|---|---|---|---|
| 1 | Mathematical specification and numerical validation | `v0.1.0` | in review |
| 2 | Repository architecture, CMake, memory layout, data schemas | `v0.2.0` | planned |
| 3 | C++20 engine, exactness and stress tests, micro-benchmarks | `v0.3.0` | planned |
| 4 | pybind11 bindings, synthetic generator, Binance data loader | `v0.4.0` | planned |
| 5 | Event-driven market-making simulator and comparative backtest | `v0.5.0` | planned |
| 6 | Figures, whitepaper, publication polish | `v1.0.0` | planned |

Benchmark numbers will be published only from real runs of the Phase 3 and Phase 5 code.

## Repository layout

```text
docs/math_spec.md                      Phase 1 specification (derivations, proofs, design decisions)
research/phase1/validate_math_spec.py  numerical verification of every claim in the spec
research/requirements.txt              pinned Python dependencies for research scripts
scripts/check_version.sh               VERSION / CHANGELOG / tag consistency check
.github/workflows/                     CI (lint, links, version, math validation) and releases
```

## Versioning

The project follows [Semantic Versioning](https://semver.org/). [`VERSION`](VERSION) is the single source of truth, and every release has an entry in [`CHANGELOG.md`](CHANGELOG.md) in the [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format.

Each approved phase bumps the minor version. Corrections within a phase bump the patch version. `1.0.0` marks the publication release.

Pushing a tag `vX.Y.Z` that matches `VERSION` triggers the release workflow, which creates a GitHub Release from the matching changelog section. The math spec also carries its own *spec version*, which changes whenever the mathematics changes.

## License

Not yet chosen.
