#!/usr/bin/env python3
"""Definition-level reference for the Phase 3 engine and generator of its cross-check fixture.

This is an independent implementation of math_spec.md (spec version 0.2.0), written from the
definitions rather than from the C++ recursions:

* Erlang states, intensities, regressors z(t-) and forecasts are direct O(N) sums over the history.
* Compensator increments and interval moment matrices are computed by adaptive quadrature
  (scipy quad_vec) of the directly summed regressor.
* The estimate is the batch discounted least-squares solution Theta~ = C M^{-1}, with M and C built
  from their batch definitions (math_spec Section 5.4, "batch reference"): interval Gram matrices
  discounted along the recorded forgetting path, discounted event cross-moments, and the ridge
  diag(omega_p e^{-K(tau_p, t)}) with the recorded replenishment times.
* The projection uses the exact sorting algorithm.
* Quantities that are recursive *by definition* (the mark normalizer, the CUSUM, the Perron step,
  the Jacobi sweep, the pull triggers, the kill switch and the requote throttle) are re-implemented
  here from the spec text.

Usage:
    python research/phase3/crosscheck_reference.py --write   # regenerate tests/fixtures/phase3_crosscheck.txt
    python research/phase3/crosscheck_reference.py --check   # regenerate in memory and compare with the file
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import integrate

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "phase3_crosscheck.txt"
FIXTURE_VERSION = 1

# ----------------------------------------------------------------------------------------------
# Configuration (written into the fixture; the C++ test reads it back)
# ----------------------------------------------------------------------------------------------
D = 4
ORDERS = (1, 2, 2, 1)
Q = len(ORDERS)
P = 1 + D * Q
BETA = (1000.0, 200.0, 50.0, 5.0)


def coord(j: int, q: int) -> int:
    return 1 + j * Q + q


def make_config() -> dict:
    theta0 = np.zeros((D, P))
    mu0 = (20.0, 20.0, 12.0, 12.0)
    for i in range(D):
        theta0[i, 0] = mu0[i]
        theta0[i, 1:] = 0.03
        theta0[i, coord(i, 0)] = 0.25
        theta0[i, coord(i, 3)] = 0.10
    theta0[2, coord(0, 1)] = 0.15
    theta0[3, coord(1, 1)] = 0.15
    omega = np.zeros(P)
    omega[0] = 0.5
    for j in range(D):
        for q in range(Q):
            omega[coord(j, q)] = 0.5 * (30.0 + 3.0 * BETA[q])
    return dict(
        beta=list(BETA),
        half_life_slow_s=2.0,
        half_life_fast_s=0.5,
        dwell_s=0.4,
        cusum_enabled=1,
        cusum_rho=1.5,
        cusum_threshold=4.0,
        heartbeat_ns=100_000_000,
        rho_max=0.97,
        chi=1.01,
        eps_v=1e-6,
        rho_crit=1.0,
        mu_min=1e-4,
        mu_max=1e6,
        v_ref_from_prior=1,
        x_cut=50.0,
        pivot_eta=1e-15,
        warmup_events=40.0,
        omega=omega.tolist(),
        theta0=theta0.ravel().tolist(),
        c_u=0.5,
        c_L=0.5,
        c_iota=0.5,
        u_max=1e4,
        L_max=50.0,
        eta_g=0.05,
        eta_nu=0.05,
        horizon_s=0.05,
        neumann_terms=3,
        direction=[1, -1, 1, -1],
        impact=[1.0, 1.0, 0.6, 0.6],
        gamma=0.05,
        k=2.0,
        horizon_inv_s=1.0,
        sigma_w2=0.5,
        delta_max=2.5,
        q_max=10.0,
        jump_benign=[0.3, 0.3, 0.2, 0.2],
        jump_informed=[1.5, 1.5, 1.0, 1.0],
        buy_channel=2,
        sell_channel=3,
        hysteresis=0.1,
        cooldown_s=0.05,
        requote_s=0.005,
        latency_rt_s=0.01,
        # instruments: id venue tick_e8; channel map: instrument side(1 buy, 2 sell) channel
        instruments=[[0, 0, 10_000_000], [1, 1, 1_000_000]],
        channel_map=[[0, 1, 0], [0, 2, 1], [1, 1, 2], [1, 2, 3]],
        quote_instrument=1,
        ooo_tolerance_ns=100_000,
        stale_ns=2_000_000_000,
        reject_rate_eta=0.01,
        reject_rate_max=0.05,
        inventory=1.5,
    )


# ----------------------------------------------------------------------------------------------
# Input stream
# ----------------------------------------------------------------------------------------------
TRADE, BOOK, HEARTBEAT, STATUS = 1, 2, 0, 3


@dataclass
class Event:
    ts_exchange_ns: int
    ts_local_ns: int
    px0: int
    qty0: int
    px1: int
    qty1: int
    seq: int
    count: int
    instrument: int
    kind: int
    flags: int


def make_events(seed: int = 20261120) -> list[Event]:
    """About 360 events over ~3.5 s: bursts, cross-channel ties, book updates, heartbeats, long gaps,
    out-of-order jitter and lateness, one invalid record, one unmapped instrument and a status event."""
    rng = np.random.default_rng(seed)
    t = 1_700_000_000_000_000_000
    mid = {0: 6_000_000_000_000, 1: 6_000_000_000_000}
    tick = {0: 10_000_000, 1: 1_000_000}
    seq = 1
    ev: list[Event] = []

    def book(inst: int, ts: int) -> None:
        nonlocal seq
        qb = int(rng.integers(10_000_000, 300_000_000))
        qa = int(rng.integers(10_000_000, 300_000_000))
        ev.append(Event(ts, ts, mid[inst] - tick[inst], qb, mid[inst] + tick[inst], qa, seq, 0, inst, BOOK, 0))
        seq += 1

    def trade(inst: int, buy: bool, ts: int, levels: int = 0, instrument_override: int | None = None) -> None:
        nonlocal seq
        first = mid[inst] + tick[inst] if buy else mid[inst] - tick[inst]
        last = first + levels * tick[inst] if buy else first - levels * tick[inst]
        qty = int(math.exp(-3.0 + 1.0 * rng.normal()) * 1e8) + 1
        flags = 1 if buy else 2
        ev.append(Event(ts, ts, last, qty, first, qty * first // 100_000_000, seq, 1 + levels,
                        inst if instrument_override is None else instrument_override, TRADE, flags))
        seq += 1

    book(0, t)
    book(1, t)
    for k in range(300):
        gap = int(rng.exponential(6_000_000))
        if k % 40 == 20:
            gap = int(rng.integers(1, 1000))  # nanosecond-scale gaps
        if k == 100:
            gap = 250_000_000  # long gap without heartbeat: full ridge restore
        if k == 200:
            gap = 600_000_000
        if 150 <= k < 170:
            gap = int(rng.integers(50_000, 400_000))  # a burst
        t += gap
        if k % 23 == 7:
            # cross-channel tie batch on both venues
            trade(0, True, t)
            trade(1, True, t)
            trade(1, False, t, levels=1)
        else:
            ch = int(rng.integers(0, 4))
            inst, buy = (0 if ch < 2 else 1), ch % 2 == 0
            if 150 <= k < 170:
                inst, buy = 0, True  # perp-buy burst
            trade(inst, buy, t, levels=int(rng.integers(0, 3)) if rng.uniform() < 0.2 else 0)
        if k % 3 == 0:
            mid[1] += tick[1] * int(rng.integers(-1, 2))
            book(1, t)
        if k % 5 == 0:
            mid[0] += tick[0] * int(rng.integers(-1, 2))
            book(0, t)
        if k == 60:
            for _ in range(3):  # explicit heartbeats through a quiet stretch
                t += 80_000_000
                ev.append(Event(t, t, 0, 0, 0, 0, 0, 0, 0, HEARTBEAT, 0))
        if k == 80:
            late = ev[-1]
            j = Event(**late.__dict__)
            j.ts_local_ns = t - 40_000  # within epsilon_ooo: clamped
            j.seq = seq
            seq += 1
            ev.append(j)
            j2 = Event(**late.__dict__)
            j2.ts_local_ns = t - 5_000_000  # too late: rejected
            j2.seq = seq
            seq += 1
            ev.append(j2)
        if k == 90:
            bad = Event(**ev[-1].__dict__)
            bad.kind, bad.flags, bad.qty0 = TRADE, 1, -5  # invalid: non-positive quantity
            ev.append(bad)
            trade(1, True, t, instrument_override=5)  # unmapped instrument
        if k == 250:
            ev.append(Event(t, t, 0, 0, 0, 0, 0, 1, 0, STATUS, 0))  # FeedGap
    return ev


# ----------------------------------------------------------------------------------------------
# Feed validation (architecture Section 7.2)
# ----------------------------------------------------------------------------------------------
def validate(e: Event) -> int:
    if e.flags & 0x80:
        return 2
    side = e.flags & 0x03
    if e.kind == TRADE:
        if side not in (1, 2):
            return 3
        if e.px0 <= 0 or e.px1 <= 0:
            return 4
        if e.qty0 <= 0:
            return 5
        if e.qty1 < 0:
            return 10
        if e.count == 0:
            return 6
        if (side == 1 and e.px0 < e.px1) or (side == 2 and e.px0 > e.px1):
            return 7
        return 0
    if e.kind == BOOK:
        if side != 0:
            return 3
        if e.px0 <= 0 or e.px1 <= 0:
            return 4
        if e.qty0 <= 0 or e.qty1 <= 0:
            return 5
        if e.count != 0:
            return 6
        if e.px0 >= e.px1:
            return 8
        return 0
    if e.kind in (HEARTBEAT, STATUS):
        if side != 0:
            return 3
        if e.px0 or e.qty0 or e.px1 or e.qty1:
            return 9
        if e.kind == HEARTBEAT and e.count != 0:
            return 6
        return 0
    return 1


# ----------------------------------------------------------------------------------------------
# Kernel definitions
# ----------------------------------------------------------------------------------------------
def h(q: int, a: np.ndarray) -> np.ndarray:
    b = BETA[q]
    return b * np.exp(-b * a) if ORDERS[q] == 1 else b * b * a * np.exp(-b * a)


def survival(q: int, a: np.ndarray) -> np.ndarray:
    b = BETA[q]
    return np.exp(-b * a) if ORDERS[q] == 1 else (1.0 + b * a) * np.exp(-b * a)


def r_state(q: int, n: int, a: np.ndarray) -> np.ndarray:
    return (a ** n if n else 1.0) * np.exp(-BETA[q] * a)


class History:
    """Accepted model steps as arrays (time in ns, channel, mark)."""

    def __init__(self) -> None:
        self.t: list[int] = []
        self.ch: list[int] = []
        self.g: list[float] = []

    def arrays(self, t_ns: int, strict: bool):
        t = np.array(self.t, dtype=np.int64)
        ch = np.array(self.ch, dtype=np.int64)
        g = np.array(self.g, dtype=float)
        m = (t < t_ns) if strict else (t <= t_ns)
        m &= ch >= 0
        return (t_ns - t[m]) / 1e9, ch[m], g[m]

    def z(self, t_ns: int, u: float = 0.0, strict: bool = True) -> np.ndarray:
        """Regressor at t_ns + u seconds from events strictly before t_ns (or at t_ns if not strict)."""
        age, ch, g = self.arrays(t_ns, strict)
        age = age + u
        z = np.zeros(P)
        z[0] = 1.0
        for q in range(Q):
            hv = g * h(q, age)
            for j in range(D):
                z[coord(j, q)] = hv[ch == j].sum()
        return z


# ----------------------------------------------------------------------------------------------
# Projection (sorting reference) and serving
# ----------------------------------------------------------------------------------------------
def project_sorted(at: np.ndarray, w: np.ndarray, c: float) -> tuple[np.ndarray, float]:
    pos = np.maximum(at, 0.0)
    if float(w @ pos) <= c:
        return pos, 0.0
    order = np.argsort(-(at / w), kind="stable")
    sa = sw = 0.0
    nu = 0.0
    for k, m in enumerate(order):
        if at[m] <= 0:
            break
        sa += w[m] * at[m]
        sw += w[m] * w[m]
        nu = (sa - c) / sw
        nxt = at[order[k + 1]] / w[order[k + 1]] if k + 1 < len(order) else -1.0
        if nu >= nxt:
            break
    a = np.maximum(at - nu * w, 0.0)
    s = float(w @ a)
    if s > c:
        a *= c / s
    return a, nu


def branching(th: np.ndarray, clamp: bool) -> np.ndarray:
    G = np.zeros((D, D))
    for i in range(D):
        for j in range(D):
            vals = th[i, [coord(j, q) for q in range(Q)]]
            G[i, j] = np.maximum(vals, 0.0).sum() if clamp else vals.sum()
    return G


@dataclass
class Served:
    theta: np.ndarray
    G: np.ndarray
    lam_inf: np.ndarray
    v: np.ndarray
    crit: float = 0.0
    n_eff: float = 0.0
    projected: int = 0
    fast: bool = False


class Guard:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.lo, self.hi = 1.0 / math.sqrt(cfg["chi"]), math.sqrt(cfg["chi"])
        theta0 = np.array(cfg["theta0"]).reshape(D, P)
        v = np.ones(D)
        if cfg["v_ref_from_prior"]:
            G0 = branching(theta0, False)
            for _ in range(1000):
                y = cfg["eps_v"] + G0 @ v
                v = y / y.max()
        self.v_ref = v

    def serve(self, tt: np.ndarray, prev: Served, perron: bool = True) -> Served:
        c = self.cfg
        th = np.zeros((D, P))
        th[:, 0] = np.clip(tt[:, 0], c["mu_min"], c["mu_max"])
        v = prev.v.copy()
        if perron:
            y = c["eps_v"] * self.v_ref + prev.G @ prev.v
            v = np.minimum(np.maximum(y / y.max(), self.lo * self.v_ref), self.hi * self.v_ref)
        w = np.repeat(v, Q)
        projected = 0
        for i in range(D):
            a, nu = project_sorted(tt[i, 1:], w, c["rho_max"] * v[i])
            th[i, 1:] = a
            if nu > 0:
                projected |= 1 << i
        G = branching(th, False)
        Gt = branching(tt, True)
        crit = float(np.max((Gt @ v) / v))
        lam_inf = th[:, 0] + G @ prev.lam_inf
        return Served(th, G, lam_inf, v, crit, 0.0, projected)


# ----------------------------------------------------------------------------------------------
# The reference engine
# ----------------------------------------------------------------------------------------------
@dataclass
class Side:
    pulled: bool = False
    t_pull: int = 0


@dataclass
class Kill:
    active: bool = True
    t_last: int = 0


@dataclass
class Throttle:
    working: int = 0
    t_change: int = 0


@dataclass
class Record:
    status: int
    t: int = 0
    flags: int = 0
    tau_b: float = 0.0
    tau_s: float = 0.0
    crit: float = 0.0
    shift: float = 0.0
    adv_a: float = 0.0
    adv_b: float = 0.0
    sigma2: float = 0.0
    bid: int = 0
    ask: int = 0
    lam: list = field(default_factory=lambda: [0.0] * D)
    model: list | None = None  # theta~, theta^, lambda_inf, v, cusum, n_eff (after a model step)


class Reference:
    def __init__(self, cfg: dict) -> None:
        self.c = cfg
        self.theta0 = np.array(cfg["theta0"]).reshape(D, P)
        self.omega = np.array(cfg["omega"])
        self.k_slow = math.log(2.0) / cfg["half_life_slow_s"]
        self.k_fast = math.log(2.0) / cfg["half_life_fast_s"]
        self.guard = Guard(cfg)
        th0 = self.theta0
        init = Served(np.zeros((D, P)), np.zeros((D, D)), np.zeros(D), self.guard.v_ref.copy())
        s = self.guard.serve(th0, init, perron=False)
        s.lam_inf = np.linalg.solve(np.eye(D) - s.G, s.theta[:, 0])
        self.served = s
        self.hist = History()
        self.started = False
        self.t_prev = 0
        self.step_count = 0
        self.intervals: list[tuple[int, int, float, np.ndarray]] = []  # (start, end, kappa, gram)
        self.tau_p = np.full(P, -1, dtype=np.int64)  # -1: initial ridge, never restored
        self.t_origin = None
        # CUSUM
        self.since = np.zeros(D)
        self.cusum = np.zeros((2, D))
        self.alarm_pending = False
        self.fast_until = -(2**63)
        self.dwell_ns = int(round(cfg["dwell_s"] * 1e9))
        # marks
        self.nu = np.zeros(D)
        self.gbar = np.zeros(D)
        self.seen = np.zeros(D, dtype=bool)
        c = cfg
        self.gmin = 1.0 - c["c_iota"]
        self.gmax = (1 + c["c_u"] * math.log1p(c["u_max"])) * (1 + c["c_L"] * math.log1p(c["L_max"])) * (1 + c["c_iota"])
        # feed / decision
        self.inst = {i: (v, tk) for (i, v, tk) in cfg["instruments"]}
        self.chmap = {(i, s): ch for (i, s, ch) in cfg["channel_map"]}
        self.books: dict[int, tuple[int, int, int, int]] = {}
        self.venue_last: dict[int, int] = {}
        self.venues_used = {v for (v, _) in self.inst.values()}
        self.clock = 0
        self.ref_started = False
        self.reject_rate = 0.0
        self.ask, self.bid, self.kill = Side(), Side(), Kill()
        self.tha, self.thb = Throttle(), Throttle()
        self.inventory = cfg["inventory"]
        self.last = Record(0)
        H = cfg["horizon_s"]
        self.kbar = [integrate.quad(lambda u, q=q: float(survival(q, np.array(u))), 0.0, H, epsabs=0, epsrel=1e-13)[0]
                     for q in range(Q)]

    # ---- forgetting path ----
    def K(self, s_ns: int, t_ns: int) -> float:
        tot = 0.0
        for (a, b, kap, _) in self.intervals:
            lo, hi = max(a, s_ns), min(b, t_ns)
            if hi > lo:
                tot += kap * (hi - lo) / 1e9
        return tot

    # ---- model step: math_spec Section 9, steps 2-9, from definitions ----
    def step(self, t: int, ch: int, g: float) -> None:
        if not self.started:
            self.started = True
            self.t_prev = t
            self.t_origin = t
        dt_ns = t - self.t_prev
        if dt_ns > 0:
            if self.alarm_pending:  # batch closes: alarm takes effect
                self.alarm_pending = False
                self.cusum[:] = 0.0
                self.fast_until = self.t_prev + self.dwell_ns
            kappa = self.k_fast if self.t_prev < self.fast_until else self.k_slow
            a_ns = self.t_prev
            dt = dt_ns / 1e9
            # Compensator increments with the served parameters, by quadrature of direct sums.
            Zq = integrate.quad_vec(lambda u: self.hist.z(a_ns, u, strict=False), 0.0, dt,
                                    epsabs=0, epsrel=1e-13, norm="max")[0]
            self.since += self.served.theta @ Zq
            gram = integrate.quad_vec(
                lambda u: math.exp(-kappa * (dt - u)) * np.outer(*(2 * [self.hist.z(a_ns, u, strict=False)])),
                0.0, dt, epsabs=0, epsrel=1e-13, norm="max")[0]
            self.intervals.append((a_ns, t, kappa, gram))
            if dt_ns > self.c["heartbeat_ns"]:
                self.tau_p[:] = t  # full ridge restore at the end of a long gap
            self.t_prev = t
        if ch >= 0:
            z = self.hist.z(t, strict=True)
            xi = self.since[ch]
            self.since[ch] = 0.0
            if self.c["cusum_enabled"]:
                rho = self.c["cusum_rho"]
                up = max(0.0, self.cusum[0, ch] + math.log(rho) - (rho - 1.0) * xi)
                dn = max(0.0, self.cusum[1, ch] - math.log(rho) - (1.0 / rho - 1.0) * xi)
                self.cusum[0, ch], self.cusum[1, ch] = up, dn
                if up > self.c["cusum_threshold"] or dn > self.c["cusum_threshold"]:
                    self.alarm_pending = True
            self.hist.t.append(t)
            self.hist.ch.append(ch)
            self.hist.g.append(g)
            self.zcache = getattr(self, "zcache", [])
            self.zcache.append(z)
        else:
            self.hist.t.append(t)
            self.hist.ch.append(-1)
            self.hist.g.append(0.0)
            self.zcache = getattr(self, "zcache", [])
            self.zcache.append(None)
        self.tau_p[self.step_count % P] = t
        self.step_count += 1
        # Batch statistics from the definitions.
        M = np.zeros((P, P))
        for (a, b, kap, gram) in self.intervals:
            M += math.exp(-self.K(b, t)) * gram
        ridge = np.array([self.omega[p] * math.exp(-self.K(int(self.tau_p[p]) if self.tau_p[p] >= 0 else self.t_origin, t))
                          for p in range(P)])
        M += np.diag(ridge)
        C = self.theta0 * ridge[None, :]
        n_eff = 0.0
        for k, (tk, chk) in enumerate(zip(self.hist.t, self.hist.ch)):
            if chk >= 0:
                wk = math.exp(-self.K(tk, t))
                C[chk] += wk * self.zcache[k]
                n_eff += wk
        theta_tilde = np.linalg.solve(M, C.T).T
        s = self.guard.serve(theta_tilde, self.served)
        s.n_eff = n_eff
        s.fast = t < self.fast_until
        self.served = s
        self.theta_tilde = theta_tilde

    # ---- marks (Section 2.3) ----
    def raw_mark(self, u: float, L: float, iota: float) -> float:
        c = self.c
        return ((1 + c["c_u"] * math.log1p(min(u, c["u_max"]))) * (1 + c["c_L"] * math.log1p(min(L, c["L_max"])))
                * (1 + c["c_iota"] * iota))

    def mark(self, j: int, size: float, depth: float, iota: float) -> float:
        ell = math.log(size)
        nu, gb = self.nu[j], self.gbar[j]
        if not self.seen[j]:
            nu = ell
            gb = min(max(self.raw_mark(1.0, depth, iota), self.gmin), self.gmax)
        u = size / math.exp(nu)
        gt = self.raw_mark(u, depth, iota)
        g = gt / gb
        self.seen[j] = True
        self.nu[j] = nu + self.c["eta_nu"] * ((1.0 if ell > nu else 0.0) - (1.0 if ell < nu else 0.0))
        self.gbar[j] = min(max(gb + self.c["eta_g"] * (gt - gb), self.gmin), self.gmax)
        return g

    # ---- signals (Section 7 and 8), definitions ----
    def decide(self, t: int, fault: bool, rec: Record) -> None:
        c, s = self.c, self.served
        H = c["horizon_s"]
        age, ch, g = self.hist.arrays(t, strict=False) if self.started else (np.zeros(0), np.zeros(0, int), np.zeros(0))
        lam = s.theta[:, 0].copy()
        m0 = s.theta[:, 0] * H
        mb = s.theta[:, 0] * H
        y = np.zeros(D)
        for j in range(D):
            sel = ch == j
            for q in range(Q):
                a = age[sel]
                gj = g[sel]
                x_now = float((gj * h(q, a)).sum())
                mass_h = float((gj * (survival(q, a) - survival(q, a + H))).sum())
                mass_inf = float((gj * survival(q, a)).sum())
                alpha = s.theta[:, coord(j, q)]
                lam += alpha * x_now
                m0 += alpha * mass_h
                mb += alpha * s.lam_inf[j] * self.kbar[q]
                y += alpha * (mass_inf - s.lam_inf[j] * ORDERS[q] / BETA[q])
        x = y.copy()
        for _ in range(c["neumann_terms"]):
            x = y + s.G @ x
        dirs, eta = np.array(c["direction"]), np.array(c["impact"])
        Mp, Mm = float((eta * m0)[dirs > 0].sum()), float((eta * m0)[dirs < 0].sum())
        Bp, Bm = float((eta * mb)[dirs > 0].sum()), float((eta * mb)[dirs < 0].sum())
        Ep, Em = max(Mp - Bp, 0.0), max(Mm - Bm, 0.0)
        tot = Ep + Em
        tau_b = (Ep / Mp if Mp > 0 else 0.0) * (max(Ep - Em, 0.0) / tot if tot > 0 else 0.0)
        tau_s = (Em / Mm if Mm > 0 else 0.0) * (max(Em - Ep, 0.0) / tot if tot > 0 else 0.0)
        # Theorem 19 / Section 8.3.
        jb, ji = np.array(c["jump_benign"]), np.array(c["jump_informed"])
        tau_i = np.where(dirs > 0, tau_b, np.where(dirs < 0, tau_s, 0.0))
        shift = float((dirs * (jb + tau_i * ji) * x).sum())
        sigma2 = c["sigma_w2"] + float((m0 * ((1 - tau_i) * jb * jb + tau_i * (jb + ji) ** 2)).sum()) / H
        risk = 0.5 * c["gamma"] * sigma2 * c["horizon_inv_s"]
        a_ch, b_ch = c["buy_channel"], c["sell_channel"]
        eps_a = jb[a_ch] + tau_b * ji[a_ch]
        eps_b = jb[b_ch] + tau_s * ji[b_ch]
        q_inv = self.inventory
        base_a = shift - risk * (2 * q_inv - 1)
        base_b = -shift + risk * (2 * q_inv + 1)
        spread = math.log(1 + c["gamma"] / c["k"]) / c["gamma"]
        dist_a = min(eps_a + base_a + spread, c["delta_max"])
        dist_b = min(eps_b + base_b + spread, c["delta_max"])

        def thr(c0: float, jI: float) -> float:
            if jI <= 0:
                return 0.0 if c0 >= c["delta_max"] else 1.0
            return min(max((c["delta_max"] - c0) / jI, 0.0), 1.0)

        th_a, th_b = thr(jb[a_ch] + base_a, ji[a_ch]), thr(jb[b_ch] + base_b, ji[b_ch])
        cool = int(round(c["cooldown_s"] * 1e9))
        stale = any((v not in self.venue_last) or (t - self.venue_last[v] > c["stale_ns"]) for v in self.venues_used)
        cond = fault or stale or s.crit >= c["rho_crit"] or self.reject_rate > c["reject_rate_max"]
        k = self.kill
        if cond:
            k.active, k.t_last = True, t
        elif k.active and t - k.t_last >= cool:
            k.active = False

        def trig(sd: Side, tau: float, th: float) -> bool:
            if not sd.pulled:
                if tau >= th:
                    sd.pulled, sd.t_pull = True, t
            elif tau <= th - c["hysteresis"] and t - sd.t_pull >= cool:
                sd.pulled = False
            return sd.pulled

        ask_pulled = trig(self.ask, tau_b, th_a)
        bid_pulled = trig(self.bid, tau_s, th_b)
        qi = c["quote_instrument"]
        tick = self.inst[qi][1]
        want_a = want_b = 0
        book = self.books.get(qi)
        if book is not None and not k.active:
            bp, bq, ap, aq = book
            mid = 0.5 * (bp / 1e8 + ap / 1e8)
            if not ask_pulled and q_inv > -c["q_max"]:
                want_a = int(math.ceil((mid + dist_a) * 1e8 / tick - 1e-9)) * tick
                want_a = max(want_a, bp + tick)
            if not bid_pulled and q_inv < c["q_max"]:
                want_b = int(math.floor((mid - dist_b) * 1e8 / tick + 1e-9)) * tick
                want_b = max(min(want_b, ap - tick), 0)
        rq = int(round(c["requote_s"] * 1e9))

        def throttle(th: Throttle, want: int) -> int:
            if want == 0 or th.working == 0:
                if want != th.working:
                    th.working, th.t_change = want, t
            elif want != th.working and t - th.t_change >= rq:
                th.working, th.t_change = want, t
            return th.working

        rec.ask = throttle(self.tha, want_a)
        rec.bid = throttle(self.thb, want_b)
        rec.t = t
        rec.lam = lam.tolist()
        rec.tau_b, rec.tau_s, rec.crit = tau_b, tau_s, s.crit
        rec.shift, rec.adv_a, rec.adv_b, rec.sigma2 = shift, eps_a, eps_b, sigma2
        flags = 0
        flags |= 1 if s.n_eff < c["warmup_events"] else 0
        flags |= 2 if ask_pulled else 0
        flags |= 4 if bid_pulled else 0
        flags |= 8 if k.active else 0
        flags |= 16 if s.fast else 0
        flags |= 32 if s.projected else 0
        flags |= 0 if book is not None else 64
        rec.flags = flags

    def on_event(self, e: Event) -> Record:
        chk = validate(e)
        if chk != 0:
            self.reject_rate += self.c["reject_rate_eta"] * (1.0 - self.reject_rate)
            r = Record(**{**self.last.__dict__, "status": 2 | (chk << 8), "model": None})
            self.last = r
            return r
        t = e.ts_local_ns
        status = 0
        if self.ref_started and t < self.clock:
            if self.clock - t > self.c["ooo_tolerance_ns"]:
                self.reject_rate += self.c["reject_rate_eta"] * (1.0 - self.reject_rate)
                r = Record(**{**self.last.__dict__, "status": 3, "model": None})
                self.last = r
                return r
            t, status = self.clock, 1
        self.ref_started = True
        self.clock = t
        self.reject_rate += self.c["reject_rate_eta"] * (0.0 - self.reject_rate)
        if e.instrument in self.inst:
            self.venue_last[self.inst[e.instrument][0]] = t
        fault = False
        stepped = False
        if e.kind == TRADE:
            side = e.flags & 3
            ch = self.chmap.get((e.instrument, side), -1)
            if ch < 0:
                status = 4
            else:
                tick = self.inst[e.instrument][1]
                size = e.qty0 / 1e8
                depth = abs(e.px0 - e.px1) / tick
                book = self.books.get(e.instrument)
                iota = 0.0
                if book is not None:
                    bq, aq = float(book[1]), float(book[3])
                    iota = self.c["direction"][ch] * (bq - aq) / (bq + aq)
                g = self.mark(ch, size, depth, iota)
                self.step(t, ch, g)
                stepped = True
        elif e.kind == BOOK:
            self.books[e.instrument] = (e.px0, e.qty0, e.px1, e.qty1)
        elif e.kind == HEARTBEAT:
            self.step(t, -1, 0.0)
            stepped = True
        else:
            fault = True
        rec = Record(status)
        self.decide(t, fault, rec)
        if stepped and (self.step_count % 3 == 0 or self.step_count < 5):
            rec.model = (self.theta_tilde.ravel().tolist() + self.served.theta.ravel().tolist()
                         + self.served.lam_inf.tolist() + self.served.v.tolist() + self.cusum.ravel().tolist()
                         + [self.served.n_eff])
        self.last = rec
        return rec


# ----------------------------------------------------------------------------------------------
# Fixture I/O
# ----------------------------------------------------------------------------------------------
def fmt(x) -> str:
    return repr(float(x)) if isinstance(x, (float, np.floating)) else str(int(x))


def render(cfg: dict, events: list[Event], records: list[Record]) -> str:
    out = [f"# PulseLOB Phase 3 cross-check fixture, version {FIXTURE_VERSION}.",
           "# Generated by research/phase3/crosscheck_reference.py; do not edit by hand.",
           "[config]"]
    for k, v in cfg.items():
        if isinstance(v, list) and v and isinstance(v[0], list):
            out.append(f"{k} {len(v)} " + " ".join(" ".join(fmt(x) for x in row) for row in v))
        elif isinstance(v, list):
            out.append(f"{k} " + " ".join(fmt(x) for x in v))
        else:
            out.append(f"{k} {fmt(v)}")
    out.append("[events]")
    out.append(str(len(events)))
    for e in events:
        out.append(" ".join(str(x) for x in (e.ts_exchange_ns, e.ts_local_ns, e.px0, e.qty0, e.px1, e.qty1,
                                              e.seq, e.count, e.instrument, e.kind, e.flags)))
    out.append("[expect]")
    for r in records:
        vals = [r.status, r.t, r.flags, r.tau_b, r.tau_s, r.crit, r.shift, r.adv_a, r.adv_b, r.sigma2, r.bid, r.ask]
        vals += r.lam
        line = " ".join(fmt(x) for x in vals)
        if r.model is not None:
            line += " | " + " ".join(fmt(x) for x in r.model)
        out.append(line)
    return "\n".join(out) + "\n"


def generate() -> str:
    cfg = make_config()
    events = make_events()
    ref = Reference(cfg)
    records = [ref.on_event(e) for e in events]
    stats = dict(events=len(events), steps=ref.step_count, intervals=len(ref.intervals),
                 alarms_effect=int(sum(1 for (a, b, k, _) in ref.intervals if k == ref.k_fast)),
                 long_gaps=int(sum(1 for (a, b, k, _) in ref.intervals if b - a > cfg["heartbeat_ns"])))
    print("reference:", stats, file=sys.stderr)
    return render(cfg, events, records)


def numeric_compare(a: str, b: str, tol: float) -> float:
    worst = 0.0
    la, lb = a.splitlines(), b.splitlines()
    if len(la) != len(lb):
        return math.inf
    for x, y in zip(la, lb):
        ta, tb = x.split(), y.split()
        if len(ta) != len(tb):
            return math.inf
        for u, v in zip(ta, tb):
            if u == v:
                continue
            try:
                fu, fv = float(u), float(v)
            except ValueError:
                return math.inf
            worst = max(worst, abs(fu - fv) / max(abs(fv), 1.0))
    return worst


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    args = ap.parse_args()
    text = generate()
    if args.write:
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE.write_text(text)
        print(f"wrote {FIXTURE.relative_to(ROOT)} ({len(text)} bytes)")
        return 0
    worst = numeric_compare(text, FIXTURE.read_text(), 1e-9)
    print(f"fixture reproducibility: worst relative difference {worst:.3e}")
    return 0 if worst < 1e-9 else 1


if __name__ == "__main__":
    sys.exit(main())
