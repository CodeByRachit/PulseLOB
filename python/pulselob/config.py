"""Engine configuration (docs/architecture.md Section 4.1).

`EngineConfig` holds every engine parameter under the key names of the C++ codec
(`include/pulselob/engine/config_io.hpp`), so `to_entries()` feeds the extension module directly and
`to_text()` writes the file read by the native `pulselob-replay` tool. The C++ side validates the
configuration when an engine is built and raises ValueError naming the violated condition.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

D = 4  # channels: 0 perp buy, 1 perp sell, 2 spot buy, 3 spot sell

SHAPES: dict[str, tuple[int, ...]] = {
    "default": (1, 2, 2, 1),  # Erlang orders of the default bank (math_spec Section 3.1)
    "exp3": (1, 1, 1),        # fallback: three exponentials
}
DEFAULT_BETA: dict[str, tuple[float, ...]] = {
    "default": (1000.0, 200.0, 50.0, 1.0),
    "exp3": (1000.0, 50.0, 1.0),
}

PERP, SPOT = 0, 1  # instrument ids of the default market
VENUE_PERP, VENUE_SPOT = 0, 1


def n_params(shape: str) -> int:
    return 1 + D * len(SHAPES[shape])


def coord(shape: str, j: int, q: int) -> int:
    """Regressor coordinate of source channel j and basis q (coordinate 0 is the baseline)."""
    return 1 + j * len(SHAPES[shape]) + q


def default_prior(shape: str = "default", mu: tuple[float, ...] = (20.0, 20.0, 12.0, 12.0), self_fast: float = 0.25,
                  self_slow: float = 0.10, cross: float = 0.15, floor: float = 0.01) -> np.ndarray:
    """Prior Theta_0 (D x P): self-excitation on the fastest and slowest bases, perp -> spot lead-lag on
    the second basis (the 5 ms Erlang-2 hump in the default bank), and a small floor elsewhere."""
    q_n = len(SHAPES[shape])
    th = np.full((D, n_params(shape)), floor)
    th[:, 0] = mu
    for i in range(D):
        th[i, coord(shape, i, 0)] = self_fast
        th[i, coord(shape, i, q_n - 1)] = self_slow
    th[2, coord(shape, 0, 1)] = cross  # perp buy -> spot buy
    th[3, coord(shape, 1, 1)] = cross  # perp sell -> spot sell
    return th


def scale_free_ridge(theta0: np.ndarray, beta: tuple[float, ...], orders: tuple[int, ...],
                     t_omega_s: float = 1.0) -> np.ndarray:
    """omega_p = T_omega E[z_p^2] under the prior, Poisson (Campbell) approximation (math_spec 5.4).
    Mirrors hawkes::set_scale_free_ridge."""
    q_n = len(orders)
    G = np.zeros((D, D))
    for i in range(D):
        for j in range(D):
            G[i, j] = theta0[i, 1 + j * q_n:1 + (j + 1) * q_n].sum()
    lam = np.linalg.solve(np.eye(D) - G, theta0[:, 0])
    omega = np.zeros(theta0.shape[1])
    omega[0] = t_omega_s
    for j in range(D):
        for q in range(q_n):
            h2 = beta[q] / 2.0 if orders[q] == 1 else beta[q] / 4.0
            omega[1 + j * q_n + q] = t_omega_s * (lam[j] ** 2 + lam[j] * h2)
    return omega


@dataclass
class EngineConfig:
    """All engine parameters. Field names are the keys of the C++ configuration codec."""

    shape: str = "default"
    # Model (math_spec Sections 3-6, Appendix B).
    beta: tuple[float, ...] = DEFAULT_BETA["default"]
    half_life_slow_s: float = 60.0
    half_life_fast_s: float = 5.0
    dwell_s: float = 10.0
    cusum_enabled: bool = True
    cusum_rho: float = 1.5
    cusum_threshold: float = 30.0  # interim conservative default (spec 0.2.1)
    heartbeat_ns: int = 100_000_000
    rho_max: float = 0.97
    chi: float = 1.01
    eps_v: float = 1e-6
    rho_crit: float = 1.0
    mu_min: float = 1e-4
    mu_max: float = 1e6
    v_ref_from_prior: bool = True
    x_cut: float = 50.0
    pivot_eta: float = 1e-15
    warmup_events: float = 200.0
    theta0: np.ndarray = field(default_factory=lambda: default_prior("default"))
    omega: np.ndarray | None = None  # None: scale-free ridge with T_omega = 1 s
    # Marks (Section 2.3).
    c_u: float = 0.5
    c_L: float = 0.5
    c_iota: float = 0.5
    u_max: float = 1e4
    L_max: float = 50.0
    eta_g: float = 0.01
    eta_nu: float = 0.01
    # Forecasts and toxicity (Section 7).
    horizon_s: float = 0.05
    neumann_terms: int = 3
    direction: tuple[int, ...] = (1, -1, 1, -1)
    impact: tuple[float, ...] = (1.0, 1.0, 0.6, 0.6)
    # Control (Section 8).
    gamma: float = 0.05
    k: float = 2.0
    horizon_inv_s: float = 1.0
    sigma_w2: float = 0.5
    delta_max: float = 2.0
    q_max: float = 10.0
    jump_benign: tuple[float, ...] = (0.3, 0.3, 0.2, 0.2)
    jump_informed: tuple[float, ...] = (1.5, 1.5, 1.0, 1.0)
    buy_channel: int = 2
    sell_channel: int = 3
    hysteresis: float = 0.1
    cooldown_s: float = 0.25
    requote_s: float = 0.005
    latency_rt_s: float = 0.01
    # Market: (id, venue, tick_e8) and (instrument, side 1 buy / 2 sell, channel).
    instruments: tuple[tuple[int, int, int], ...] = ((PERP, VENUE_PERP, 10_000_000), (SPOT, VENUE_SPOT, 1_000_000))
    channel_map: tuple[tuple[int, int, int], ...] = ((PERP, 1, 0), (PERP, 2, 1), (SPOT, 1, 2), (SPOT, 2, 3))
    quote_instrument: int = SPOT
    ooo_tolerance_ns: int = 100_000
    stale_ns: int = 2_000_000_000
    reject_rate_eta: float = 0.01
    reject_rate_max: float = 0.05

    @classmethod
    def default(cls, shape: str = "default", **overrides: Any) -> EngineConfig:
        """Default configuration for `shape` ("default" or "exp3"); keyword overrides are applied last."""
        if shape not in SHAPES:
            raise ValueError(f"shape must be one of {sorted(SHAPES)}")
        cfg = cls(shape=shape, beta=DEFAULT_BETA[shape], theta0=default_prior(shape))
        return dataclasses.replace(cfg, **overrides) if overrides else cfg

    def replace(self, **changes: Any) -> EngineConfig:
        return dataclasses.replace(self, **changes)

    @property
    def orders(self) -> tuple[int, ...]:
        return SHAPES[self.shape]

    def ridge(self) -> np.ndarray:
        if self.omega is not None:
            return np.asarray(self.omega, dtype=float)
        return scale_free_ridge(np.asarray(self.theta0, dtype=float), tuple(self.beta), self.orders)

    def to_entries(self) -> dict[str, list[float]]:
        """{key: [values]} for the C++ codec."""
        p = n_params(self.shape)
        theta0 = np.asarray(self.theta0, dtype=float)
        if theta0.shape != (D, p):
            raise ValueError(f"theta0 must have shape {(D, p)} for shape {self.shape!r}")
        out: dict[str, list[float]] = {}
        for f in dataclasses.fields(self):
            name = f.name
            if name == "shape":
                continue
            value = getattr(self, name)
            if name == "theta0":
                out[name] = theta0.ravel().tolist()
            elif name == "omega":
                out[name] = self.ridge().tolist()
            elif name in ("instruments", "channel_map"):
                flat = [float(len(value))]
                for row in value:
                    flat.extend(float(x) for x in row)
                out[name] = flat
            elif isinstance(value, (tuple, list, np.ndarray)):
                out[name] = [float(x) for x in value]
            else:
                out[name] = [float(value)]
        return out

    def to_text(self) -> str:
        """The text form read by `pulselob-replay --config` (one `key values...` per line)."""
        lines = [f"# PulseLOB engine configuration, shape {self.shape}"]
        for key, values in self.to_entries().items():
            lines.append(key + " " + " ".join(repr(v) if not float(v).is_integer() or abs(v) > 2**53 else
                                              str(int(v)) for v in values))
        return "\n".join(lines) + "\n"

    @classmethod
    def from_text(cls, text: str, shape: str = "default") -> EngineConfig:
        cfg = cls.default(shape)
        updates: dict[str, Any] = {}
        for raw in text.splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            key, *vals = line.split()
            nums = [float(v) for v in vals]
            if key == "theta0":
                updates[key] = np.array(nums).reshape(D, n_params(shape))
            elif key == "omega":
                updates[key] = np.array(nums)
            elif key in ("instruments", "channel_map"):
                n = int(nums[0])
                updates[key] = tuple(tuple(int(x) for x in nums[1 + 3 * r:4 + 3 * r]) for r in range(n))
            elif key in ("beta", "impact", "jump_benign", "jump_informed"):
                updates[key] = tuple(nums)
            elif key == "direction":
                updates[key] = tuple(int(x) for x in nums)
            else:
                (v,) = nums
                current = getattr(cfg, key)
                updates[key] = bool(v) if isinstance(current, bool) else (int(v) if isinstance(current, int) else v)
        return dataclasses.replace(cfg, **updates)

    def kappa(self) -> tuple[float, float]:
        """(kappa_slow, kappa_fast) in 1/s."""
        return math.log(2.0) / self.half_life_slow_s, math.log(2.0) / self.half_life_fast_s
