"""Synthetic cross-venue data with ground truth (Phase 4).

The process is the model of math_spec Section 3 with the default kernel bank (orders 1, 2, 2, 1;
rates 1000, 200, 50, 1 per second) and piecewise-constant parameters, simulated exactly by Ogata
thinning in C++ (include/pulselob/synth/hawkes_simulator.hpp). Channels: 0 perp buy, 1 perp sell,
2 spot buy, 3 spot sell; instrument 0 is the perpetual (venue A, leading), instrument 1 spot
(venue B, quoting).

Every scenario is a list of segments (start time, Theta). Generated datasets carry their ground
truth: the true left-limit intensities lambda(t_k-) of all channels at every event, the parameter
segments, and the times at which the regime changes ("jumps").
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .. import _core
from ..config import DEFAULT_BETA, D, coord, n_params
from ..plev import sidecar_path, write_plev
from ..schema import MARKET_EVENT_DTYPE, EventKind

BETA = DEFAULT_BETA["default"]
P = n_params("default")
FAST, HUMP_5MS, HUMP_20MS, SLOW = 0, 1, 2, 3
PERP_BUY, PERP_SELL, SPOT_BUY, SPOT_SELL = 0, 1, 2, 3


def theta(mu: tuple[float, float, float, float], self_fast: float = 0.20, self_slow: float = 0.10,
          lead_5ms: float = 0.0, lead_20ms: float = 0.0) -> np.ndarray:
    """Theta (D x P): baselines, self-excitation on the 1 ms and 1 s exponentials, and perp -> spot
    same-side lead-lag on the 5 ms and 20 ms Erlang-2 humps."""
    th = np.zeros((D, P))
    th[:, 0] = mu
    for i in range(D):
        th[i, coord("default", i, FAST)] = self_fast
        th[i, coord("default", i, SLOW)] = self_slow
    for src, dst in ((PERP_BUY, SPOT_BUY), (PERP_SELL, SPOT_SELL)):
        th[dst, coord("default", src, HUMP_5MS)] = lead_5ms
        th[dst, coord("default", src, HUMP_20MS)] = lead_20ms
    return th


def branching(th: np.ndarray) -> np.ndarray:
    """G_ij = sum_q alpha^q_ij."""
    q_n = len(BETA)
    return np.array([[th[i, 1 + j * q_n:1 + (j + 1) * q_n].sum() for j in range(D)] for i in range(D)])


def stationary_rates(th: np.ndarray) -> np.ndarray:
    """lambda_inf = (I - G)^{-1} mu (requires rho(G) < 1)."""
    return np.linalg.solve(np.eye(D) - branching(th), th[:, 0])


@dataclass
class Scenario:
    name: str
    description: str
    horizon_s: float
    segment_starts: list[float]
    thetas: list[np.ndarray]
    jumps: list[float] = field(default_factory=list)  # times of regime changes and bursts

    def theta_at(self, t_s: float) -> np.ndarray:
        k = int(np.searchsorted(np.asarray(self.segment_starts), t_s, side="right")) - 1
        return self.thetas[max(k, 0)]

    def metadata(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "horizon_s": self.horizon_s,
            "beta": list(BETA),
            "orders": [1, 2, 2, 1],
            "segments": [{"t_start_s": float(t), "theta": th.tolist(),
                          "spectral_radius": float(max(abs(np.linalg.eigvals(branching(th)))))}
                         for t, th in zip(self.segment_starts, self.thetas)],
            "jumps_s": [float(j) for j in self.jumps],
        }


def calm(horizon_s: float = 600.0) -> Scenario:
    th = theta((10.0, 10.0, 6.0, 6.0), lead_5ms=0.10, lead_20ms=0.05)
    return Scenario("calm", "stationary baseline flow with a weak perp -> spot lead-lag", horizon_s, [0.0], [th])


def two_sided(horizon_s: float = 300.0) -> Scenario:
    th = theta((80.0, 80.0, 80.0, 80.0), self_fast=0.25, lead_5ms=0.10, lead_20ms=0.05)
    return Scenario("two_sided", "high, symmetric volume on both sides of both venues: toxicity should stay near "
                    "zero", horizon_s, [0.0], [th])


def lead_lag(horizon_s: float = 300.0, every_s: float = 5.0, burst_s: float = 0.1, burst_rate: float = 1500.0) -> Scenario:
    """Exogenous perp bursts (alternating buy and sell) that propagate to spot through the 5 ms and
    20 ms humps."""
    base = theta((10.0, 10.0, 6.0, 6.0), lead_5ms=0.35, lead_20ms=0.20)
    starts, thetas, jumps = [0.0], [base], []
    t, k = every_s, 0
    while t + burst_s < horizon_s:
        b = base.copy()
        b[PERP_BUY if k % 2 == 0 else PERP_SELL, 0] = burst_rate
        starts += [t, t + burst_s]
        thetas += [b, base]
        jumps.append(t)
        t += every_s
        k += 1
    return Scenario("lead_lag", "perp bursts propagating to spot through the Erlang-2 humps", horizon_s, starts,
                    thetas, jumps)


def regime_shift(horizon_s: float = 600.0, before: float = 0.10, after: float = 0.50) -> Scenario:
    """The cross-venue coupling G(spot <- perp) steps from `before` to `after` at mid-horizon (the A1 -> A2
    experiment of math_spec V10)."""
    a1 = theta((20.0, 20.0, 12.0, 12.0), self_fast=0.25, lead_5ms=before)
    a2 = theta((20.0, 20.0, 12.0, 12.0), self_fast=0.25, lead_5ms=after)
    t = horizon_s / 2.0
    return Scenario("regime_shift", f"coupling {before} -> {after} at t = {t:g} s", horizon_s, [0.0, t], [a1, a2], [t])


def flash_crash(horizon_s: float = 120.0, at_s: float = 60.0, duration_s: float = 2.0) -> Scenario:
    """A liquidation cascade: heavy aggressive selling on both venues with near-critical
    self-excitation for `duration_s`, then recovery."""
    base = theta((10.0, 10.0, 6.0, 6.0), lead_5ms=0.10, lead_20ms=0.05)
    crash = theta((10.0, 3000.0, 6.0, 1500.0), self_fast=0.60, self_slow=0.20, lead_5ms=0.10, lead_20ms=0.05)
    return Scenario("flash_crash", "directional sell cascade with tight inter-arrivals", horizon_s,
                    [0.0, at_s, at_s + duration_s], [base, crash, base], [at_s, at_s + duration_s])


SCENARIOS: dict[str, Callable[..., Scenario]] = {
    "calm": calm,
    "two_sided": two_sided,
    "lead_lag": lead_lag,
    "regime_shift": regime_shift,
    "flash_crash": flash_crash,
}


@dataclass
class Dataset:
    scenario: Scenario
    seed: int
    events: np.ndarray          # MARKET_EVENT_DTYPE (trades, book updates, heartbeats)
    t_s: np.ndarray             # channel-event times (seconds from the start)
    channel: np.ndarray         # channel of each channel event
    lambda_true: np.ndarray     # (n, D) true intensities lambda(t_k-) at each channel event
    trade_index: np.ndarray     # position in `events` of each channel event's Trade record
    options: dict

    def metadata(self) -> dict:
        return {
            "generator": "pulselob.synth",
            "version": _core.__version__,
            "seed": self.seed,
            "scenario": self.scenario.metadata(),
            "stream_options": self.options,
            "instruments": [
                {"id": 0, "venue": "perp", "role": "lead", "tick_size_e8": self.options["tick_e8"][0]},
                {"id": 1, "venue": "spot", "role": "quote", "tick_size_e8": self.options["tick_e8"][1]},
            ],
            "channels": ["perp buy", "perp sell", "spot buy", "spot sell"],
            "ground_truth": "lambda(t-) per channel event in <file>.truth.npz",
        }


def generate(scenario: Scenario | str, seed: int = 0, *, constant_marks: bool = False,
             latency_ns: tuple[int, int] = (0, 0), t0_ns: int = 1_700_000_000_000_000_000,
             heartbeat_ns: int = 100_000_000, max_events: int = 50_000_000, **scenario_kwargs) -> Dataset:
    """Simulate a scenario and build its MarketEvent stream with ground truth.

    constant_marks=True gives unit sizes, no sweeps and balanced books, so the engine's mark is
    exactly 1 and the data follow exactly the model the engine estimates.
    """
    if isinstance(scenario, str):
        scenario = SCENARIOS[scenario](**scenario_kwargs)
    thetas = np.stack([np.asarray(t, dtype=float) for t in scenario.thetas])
    t_s, channel, lam = _core.simulate_hawkes(list(BETA), [float(t) for t in scenario.segment_starts], thetas,
                                              float(scenario.horizon_s), int(seed), int(max_events), True)
    options = {
        "t0_ns": int(t0_ns),
        "tick_e8": [10_000_000, 1_000_000],
        "mid_e8": [6_000_000_000_000, 6_000_000_000_000],
        "heartbeat_ns": int(heartbeat_ns),
        "constant_marks": bool(constant_marks),
        "latency_ns": [int(latency_ns[0]), int(latency_ns[1])],
    }
    raw = _core.build_market_events(t_s, channel, int(seed) ^ 0x9E3779B97F4A7C15, options)
    events = raw.view(MARKET_EVENT_DTYPE)
    trades = np.flatnonzero(events["kind"] == EventKind.TRADE)
    trade_index = trades[np.argsort(events["seq"][trades], kind="stable")]  # channel-event order
    return Dataset(scenario, int(seed), events, t_s, channel, lam, trade_index, options)


def truth_path(path: str | Path) -> Path:
    path = Path(path)
    return path.with_name(path.name + ".truth.npz")


def write(dataset: Dataset, path: str | Path) -> Path:
    """Write <path> (.plev), <path>.json (provenance and scenario) and <path>.truth.npz."""
    path = Path(path)
    write_plev(path, dataset.events, dataset.metadata())
    np.savez_compressed(truth_path(path), t_s=dataset.t_s, channel=dataset.channel,
                        lambda_true=dataset.lambda_true, trade_index=dataset.trade_index,
                        segment_starts=np.asarray(dataset.scenario.segment_starts),
                        thetas=np.stack(dataset.scenario.thetas), jumps=np.asarray(dataset.scenario.jumps))
    return path


def load_truth(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(truth_path(path)) as z:
        return {k: z[k] for k in z.files}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pulselob-synth", description="Generate a synthetic .plev dataset with ground truth.")
    ap.add_argument("scenario", choices=sorted(SCENARIOS))
    ap.add_argument("--out", required=True, help="output .plev path")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--horizon", type=float, default=None, help="seconds (default: the scenario's)")
    ap.add_argument("--constant-marks", action="store_true")
    args = ap.parse_args(argv)
    kwargs = {} if args.horizon is None else {"horizon_s": args.horizon}
    ds = generate(args.scenario, args.seed, constant_marks=args.constant_marks, **kwargs)
    write(ds, args.out)
    meta = {"events": int(ds.events.size), "channel_events": int(ds.t_s.size), "out": str(args.out),
            "sidecar": str(sidecar_path(args.out)), "truth": str(truth_path(args.out))}
    print(json.dumps(meta))
    return 0
