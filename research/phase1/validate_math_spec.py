#!/usr/bin/env python3
"""Numerical validation of the claims in docs/math_spec.md (Phase 1).

This is research-verification code, NOT the production engine. It checks every
closed form, recursion and certificate in the spec against definition-level
brute-force references, and (with --full) reruns the estimator-comparison
experiments whose numbers are quoted in Section 5.8 of the spec.

    python research/phase1/validate_math_spec.py          # identity checks (~1 min)
    python research/phase1/validate_math_spec.py --full   # + experiments (~5-10 min)

Exit code is non-zero if any check fails. All randomness is seeded.
"""
from __future__ import annotations

import argparse
import math
import sys
import time

import mpmath as mp
import numpy as np
from scipy import integrate

mp.mp.dps = 50
FAILED: list[str] = []


def check(name: str, err: float, tol: float) -> None:
    ok = err <= tol
    if not ok:
        FAILED.append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: err={err:.3e} (tol {tol:.0e})")


# --------------------------------------------------------------------------
# psi_n(x) = int_0^1 s^n e^{-x s} ds   (spec eq. psi-def)
# --------------------------------------------------------------------------
PSI_SERIES_RADIUS = 1.0
PSI_SERIES_TERMS = 22


def psi(n: int, x: float, emx: float | None = None) -> float:
    """Series for |x| < 1, upward recurrence otherwise (amplification n/x <= n). emx = e^{-x} if known."""
    if abs(x) < PSI_SERIES_RADIUS:
        s, term = 0.0, 1.0
        for m in range(PSI_SERIES_TERMS):
            s += term / (n + m + 1)
            term *= -x / (m + 1)
        return s
    if emx is None:
        emx = math.exp(-x)
    p = (1.0 - emx) / x
    for k in range(1, n + 1):
        p = (k * p - emx) / x
    return p


def psi_ref(n: int, x: float) -> float:
    return float(mp.quad(lambda s: s**n * mp.e ** (-mp.mpf(x) * s), [0, 1]))


# --------------------------------------------------------------------------
# Q1. Erlang-r state recursion with tie batching (Theorem 2, Prop. 4)
# --------------------------------------------------------------------------
def q1_erlang_recursion() -> None:
    print("Q1  Erlang-r recursion vs O(N^2) brute force (strict left limits, ties)")
    rng = np.random.default_rng(11)
    beta = 37.0
    t = np.sort(rng.uniform(0.0, 2.0, 500))
    t[50:56] = t[50]
    t[300:302] = t[300]
    g = rng.uniform(0.5, 2.0, t.size)
    for r in (1, 2, 3):
        R = np.zeros(r)
        pending, t_last, worst = 0.0, 0.0, 0.0
        for k, tk in enumerate(t):
            d = tk - t_last
            if d > 0:
                R[0] += pending
                pending = 0.0
                e = math.exp(-beta * d)
                newR = np.zeros(r)
                for m in range(r):
                    newR[m] = e * sum(d ** (m - l) / math.factorial(m - l) * R[l] for l in range(m + 1))
                R, t_last = newR, tk
            mask = t < tk
            s = tk - t[mask]
            for m in range(r):
                ref = np.sum(g[mask] * s**m / math.factorial(m) * np.exp(-beta * s))
                worst = max(worst, abs(R[m] - ref) / max(1.0, ref))
            pending += g[k]
        check(f"Erlang-{r} states R^(0..{r-1}) over 500 events", worst, 1e-12)


# --------------------------------------------------------------------------
# Q2/Q3. Compensator closed forms and psi accuracy (Theorem 3)
# --------------------------------------------------------------------------
def q2_compensator() -> None:
    print("Q2  Compensator increments: psi-form vs 50-digit reference; naive form")
    beta = 37.0
    worst_psi, worst_naive = 0.0, 0.0
    for (R0, R1, D) in [(0.2, 0.001, 1e-9), (0.2, 0.001, 2e-7), (3.0, 0.01, 1e-5), (1.0, 1.0, 0.5), (5.0, 0.3, 4.0)]:
        x = beta * D
        X = mp.mpf(beta) * D
        ref1 = R0 * (1 - mp.e ** (-X))
        ref2 = mp.mpf(beta) * R1 * (1 - mp.e ** (-X)) + R0 * (1 - mp.e ** (-X) * (1 + X))
        c1 = R0 * x * psi(0, x)
        c2 = beta * R1 * x * psi(0, x) + R0 * x * x * psi(1, x)
        e = math.exp(-x)
        n2 = beta * R1 * (1 - e) + R0 * (1 - e * (1 + x))
        worst_psi = max(worst_psi, abs(c1 - float(ref1)) / float(ref1), abs(c2 - float(ref2)) / float(ref2))
        worst_naive = max(worst_naive, abs(n2 - float(ref2)) / float(ref2))
    check("psi-form Erlang-1/2 compensator (rel)", worst_psi, 1e-14)
    print(f"  [INFO] naive 1-e^-x(1+x) form worst rel err = {worst_naive:.3e}  (why the psi-form is mandatory)")
    print("Q3  psi_n accuracy, n=0..3, x in [0, 200] (and x<0 for n=0 only, the sole negative case)")
    worst = 0.0
    for n in range(4):
        xs = (1e-12, 1e-6, 1e-3, 0.2, 0.5, 0.999, 1.001, 2.0, 3.0, 7.0, 60.0, 200.0)
        for x in xs + ((-0.5, -0.01) if n == 0 else ()):
            ref = psi_ref(n, x)
            worst = max(worst, abs(psi(n, x) - ref) / ref)
    check("psi_n relative accuracy", worst, 5e-14)


# --------------------------------------------------------------------------
# Kernel bank used by the estimator checks: per basis (order r, rate beta)
# --------------------------------------------------------------------------
class Bank:
    def __init__(self, D: int, bases: list[tuple[int, float]]):
        self.D, self.bases = D, bases
        self.Q = len(bases)
        self.P = 1 + D * self.Q
        # time-basis functions f_m(u) = u^n e^{-c u}: list of (n, c)
        self.f = [(0, 0.0)]
        self.f_index = {}
        for q, (r, b) in enumerate(bases):
            for n in range(r):
                self.f_index[(q, n)] = len(self.f)
                self.f.append((n, b))

    def coord(self, j: int, q: int) -> int:
        return 1 + j * self.Q + q

    def V(self, R: list[np.ndarray]) -> np.ndarray:
        """Coefficient matrix V (M+1 x P) s.t. z(t+u) = sum_m f_m(u) V[m]."""
        V = np.zeros((len(self.f), self.P))
        V[0, 0] = 1.0
        for q, (r, b) in enumerate(self.bases):
            for j in range(self.D):
                c = self.coord(j, q)
                if r == 1:
                    V[self.f_index[(q, 0)], c] = b * R[q][j, 0]
                else:  # Erlang-2: beta^2 (R1 + u R0) e^{-beta u}
                    V[self.f_index[(q, 0)], c] = b * b * R[q][j, 1]
                    V[self.f_index[(q, 1)], c] = b * b * R[q][j, 0]
        return V

    def z_left_limit(self, R: list[np.ndarray]) -> np.ndarray:
        """z(t-) = sum_m f_m(0) V[m]; f_m(0) = 1 iff n_m = 0."""
        f0 = np.array([1.0 if n == 0 else 0.0 for (n, _) in self.f])
        return self.V(R).T @ f0

    def z_direct(self, s: float, t: np.ndarray, ch: np.ndarray, g: np.ndarray) -> np.ndarray:
        """Definition-level regressor z(s) = (1, x_j^q(s)) with strict t_l < s."""
        z = np.zeros(self.P)
        z[0] = 1.0
        m = t < s
        dt, cj, gj = s - t[m], ch[m], g[m]
        for q, (r, b) in enumerate(self.bases):
            h = b**r * dt ** (r - 1) / math.factorial(r - 1) * np.exp(-b * dt)
            for j in range(self.D):
                z[self.coord(j, q)] = np.sum(gj[cj == j] * h[cj == j])
        return z

    def F(self, D: float, kappa: float) -> np.ndarray:
        """F_ab = int_0^D e^{-kappa (D-u)} f_a(u) f_b(u) du = e^{-kD} D^{n+1} psi_n((c-k)D)."""
        M = len(self.f)
        F = np.zeros((M, M))
        dk = math.exp(-kappa * D)
        for a, (na, ca) in enumerate(self.f):
            for b_, (nb, cb) in enumerate(self.f):
                n, c = na + nb, ca + cb
                F[a, b_] = dk * D ** (n + 1) * psi(n, (c - kappa) * D)
        return F


def pivoted_cholesky(F: np.ndarray, rel_tol: float = 1e-15) -> np.ndarray:
    F = F.copy()
    tr = np.trace(F)
    rows = []
    for _ in range(len(F)):
        p = int(np.argmax(np.diag(F)))
        if F[p, p] <= rel_tol * tr:
            break
        l = F[:, p] / math.sqrt(F[p, p])
        rows.append(l)
        F -= np.outer(l, l)
    return np.array(rows).reshape(-1, len(F))


def q4_lowrank_gram() -> None:
    print("Q4  Exact low-rank interval Gram (Theorem 7) vs adaptive quadrature")
    rng = np.random.default_rng(5)
    bank = Bank(2, [(1, 900.0), (2, 60.0), (1, 1.5)])
    kappa = 0.05
    worst, ranks = 0.0, []
    for D in (1e-7, 1e-5, 1e-3, 0.02, 0.5, 5.0):
        Vm = rng.normal(size=(len(bank.f), bank.P))
        L = pivoted_cholesky(bank.F(D, kappa))
        ranks.append(L.shape[0])
        W = L @ Vm
        G_lr = W.T @ W
        fvec = lambda u: np.array([u**n * math.exp(-c * u) for (n, c) in bank.f])
        G_q = integrate.quad_vec(lambda u: math.exp(-kappa * (D - u)) * np.outer(fvec(u) @ Vm, fvec(u) @ Vm),
                                 0, D, epsabs=0, epsrel=1e-13, norm="max")[0]
        worst = max(worst, np.max(np.abs(G_lr - G_q)) / np.max(np.abs(G_q)))
    check("rank-(M+1) interval Gram vs quadrature (rel)", worst, 1e-12)
    print(f"  [INFO] numerical rank of F for D=1e-7..5s: {ranks} (M+1={len(bank.f)})")


# --------------------------------------------------------------------------
# Q5. Online discounted LS (Potter square root + ridge replenishment)
#     == batch discounted LS rebuilt from definitions (Theorem 10)
# --------------------------------------------------------------------------
def potter_update(S: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """S S^T = G^{-1}; returns (S', k) with S'S'^T = (G + w w^T)^{-1}, k = (G+ww^T)^{-1} w."""
    a = S.T @ w
    s2 = float(a @ a)
    Sa = S @ a
    root = math.sqrt(1.0 + s2)
    c = 1.0 / (root * (root + 1.0))
    return S - c * np.outer(Sa, a), Sa / (1.0 + s2)


def q5_online_equals_batch() -> None:
    print("Q5  Online recursive LS == batch discounted LS from definitions")
    rng = np.random.default_rng(23)
    bank = Bank(2, [(1, 30.0), (2, 8.0)])
    D, P = bank.D, bank.P
    N = 260
    t = np.cumsum(rng.exponential(0.02, N))
    t[40:43] = t[40]
    t[150:152] = t[150]
    ch = rng.integers(0, D, N)
    g = rng.uniform(0.5, 2.0, N)
    k_slow, k_fast, switch_at = 1 / 3.0, 1 / 0.4, 130
    omega = np.full(P, 0.05)
    theta0 = np.abs(rng.normal(0.3, 0.1, size=(D, P)))

    # ---- online recursion ----
    R = [np.zeros((D, r)) for (r, _) in bank.bases]
    pending = np.zeros(D)
    S = np.diag(1.0 / np.sqrt(omega))
    Theta = theta0.copy()
    Om_eff = omega.copy()
    t_prev, kappa, rr = 0.0, k_slow, 0
    kappa_path = []  # (start, end, kappa)
    repl_time = np.zeros(P)
    online = {}
    for k in range(N):
        d = t[k] - t_prev
        if d > 0:
            for q in range(bank.Q):
                R[q][:, 0] += pending
            pending[:] = 0
            dk = math.exp(-kappa * d)
            S /= math.sqrt(dk)
            Om_eff *= dk
            V = bank.V(R)
            for l in pivoted_cholesky(bank.F(d, kappa)):
                w = V.T @ l
                S, kv = potter_update(S, w)
                Theta -= np.outer(Theta @ w, kv)
            for q, (r, b) in enumerate(bank.bases):
                e = math.exp(-b * d)
                if r == 1:
                    R[q][:, 0] *= e
                else:
                    R[q][:, 1] = e * (R[q][:, 1] + d * R[q][:, 0])
                    R[q][:, 0] *= e
            kappa_path.append((t_prev, t[k], kappa))
            t_prev = t[k]
        z = bank.z_left_limit(R)
        i = ch[k]
        Theta[i] += S @ (S.T @ z)
        pending[i] += g[k]
        p = rr % P
        rr += 1
        amt = omega[p] - Om_eff[p]
        if amt > 0:
            S, _ = potter_update(S, math.sqrt(amt) * np.eye(P)[p])
            Theta += amt * np.outer(theta0[:, p] - Theta[:, p], S @ (S.T @ np.eye(P)[p]))
        Om_eff[p] = omega[p]
        repl_time[p] = t[k]
        if k == switch_at:
            kappa = k_fast
        if k % 20 == 0 or k == N - 1:
            online[k] = (Theta.copy(), repl_time.copy())

    # ---- batch reference from definitions ----
    def K(s: float, tt: float) -> float:  # int_s^tt kappa(u) du over the recorded path
        tot = 0.0
        for (a, b, kap) in kappa_path:
            lo, hi = max(a, s), min(b, tt)
            if hi > lo:
                tot += kap * (hi - lo)
        return tot

    interval_grams = []
    for (a, b, kap) in kappa_path:
        Gi = integrate.quad_vec(lambda s: math.exp(-kap * (b - s)) * np.outer(*(2 * [bank.z_direct(s, t, ch, g)])),
                                a, b, epsabs=0, epsrel=1e-13, norm="max")[0]
        interval_grams.append((b, Gi))
    worst = 0.0
    for k, (Th, rtime) in online.items():
        tk = t[k]
        Om = np.array([omega[p] * math.exp(-K(rtime[p], tk)) for p in range(P)])
        G = np.diag(Om) + sum(math.exp(-K(b, tk)) * Gi for (b, Gi) in interval_grams if b <= tk)
        B = theta0 * Om[None, :]
        for l in range(k + 1):
            B[ch[l]] += math.exp(-K(t[l], tk)) * bank.z_direct(t[l], t, ch, g)
        Th_ref = np.linalg.solve(G, B.T).T
        worst = max(worst, np.max(np.abs(Th - Th_ref)) / np.max(np.abs(Th_ref)))
    check(f"Theta_online vs Theta_batch at {len(online)} checkpoints (N={N}, ties, kappa switch)", worst, 1e-10)


def q6_potter_stability() -> None:
    print("Q6  Square-root (Potter) vs Sherman-Morrison inverse drift, ill-conditioned stream")
    rng = np.random.default_rng(9)
    P = 17
    scales = 10.0 ** np.linspace(-3, 3, P)
    G = np.eye(P) * 1e-2
    S = np.eye(P) * 10.0
    Gi = np.linalg.inv(G)
    dk = math.exp(-1e-4)
    for _ in range(100_000):
        w = rng.normal(size=P) * scales * rng.uniform(0, 1)
        G = dk * G + np.outer(w, w)
        S /= math.sqrt(dk)
        S, _ = potter_update(S, w)
        Gi /= dk
        u = Gi @ w
        Gi -= np.outer(u, u) / (1 + w @ u)
    ref = np.linalg.inv(G)
    e_pot = np.max(np.abs(S @ S.T - ref)) / np.max(np.abs(ref))
    e_sm = np.max(np.abs(Gi - ref)) / np.max(np.abs(ref))
    min_eig_sm = np.min(np.linalg.eigvalsh(0.5 * (Gi + Gi.T)))
    print(f"  [INFO] cond(G)={np.linalg.cond(G):.1e}  Potter rel err={e_pot:.2e}  SM rel err={e_sm:.2e}  SM min eig={min_eig_sm:.2e}")
    check("Potter factor stays accurate", e_pot, 1e-6)


# --------------------------------------------------------------------------
# Q7. Collatz-Wielandt certificate and exact row projection (Lemma 13, Prop. 14)
# --------------------------------------------------------------------------
def project_row(a_t: np.ndarray, w: np.ndarray, cap: float) -> tuple[np.ndarray, float]:
    a_plus = np.maximum(a_t, 0.0)
    if w @ a_plus <= cap:
        return a_plus, 0.0
    bp = a_t / w
    order = np.argsort(-bp)
    A = W2 = 0.0
    nu = 0.0
    for idx, m in enumerate(order):
        if a_t[m] <= 0:
            break
        A += w[m] * a_t[m]
        W2 += w[m] ** 2
        nu = (A - cap) / W2
        nxt = bp[order[idx + 1]] if idx + 1 < len(order) else -np.inf
        if nu >= max(nxt, 0.0):
            break
    return np.maximum(a_t - nu * w, 0.0), nu


def q7_cw_and_projection() -> None:
    print("Q7  Collatz-Wielandt bound >= spectral radius; exact projection satisfies KKT")
    rng = np.random.default_rng(3)
    worst_cw = 0.0
    for _ in range(3000):
        n = int(rng.integers(2, 10))
        A = rng.uniform(0, 1, (n, n)) * (rng.uniform(size=(n, n)) < 0.6)
        v = rng.uniform(0.05, 2, n)
        worst_cw = max(worst_cw, max(abs(np.linalg.eigvals(A))) - np.max(A @ v / v))
    check("max(rho(A) - CW(v)) over 3000 random nonnegative A (must be <= 0)", max(worst_cw, 0.0), 1e-12)
    worst_kkt = 0.0
    for _ in range(3000):
        n = 16
        a_t = rng.normal(0.2, 0.3, n)
        w = rng.uniform(0.3, 2.0, n)
        cap = rng.uniform(0.05, 1.5)
        a, nu = project_row(a_t, w, cap)
        viol = [max(0.0, -a.min()), max(0.0, w @ a - cap)]
        if nu > 0:
            viol.append(abs(w @ a - cap))
        pos = a > 0
        viol.append(np.max(np.abs(a[pos] - (a_t[pos] - nu * w[pos]))) if pos.any() else 0.0)
        viol.append(max(0.0, np.max(a_t[~pos] - nu * w[~pos])) if (~pos).any() else 0.0)
        worst_kkt = max(worst_kkt, max(viol))
    check("row projection KKT residual over 3000 random rows", worst_kkt, 1e-12)


# --------------------------------------------------------------------------
# Q8. Campbell stationary means E[R0]=lam/beta, E[R1]=lam/beta^2 for a
#     stationary Hawkes with Erlang-2 kernel (exact cluster simulation)
# --------------------------------------------------------------------------
def simulate_cluster(mu: np.ndarray, A: np.ndarray, beta: float, order: int, T: float, rng) -> tuple[np.ndarray, np.ndarray]:
    D = len(mu)
    times, chans = [], []
    gen_t, gen_c = [], []
    for i in range(D):
        n = rng.poisson(mu[i] * (T + 50.0))
        gen_t.append(rng.uniform(-50.0, T, n))
        gen_c.append(np.full(n, i))
    gt, gc = np.concatenate(gen_t), np.concatenate(gen_c)
    while gt.size:
        times.append(gt)
        chans.append(gc)
        nt, nc = [], []
        for i in range(D):
            for j in range(D):
                par = gt[gc == j]
                k = rng.poisson(A[i, j], par.size)
                if k.sum() == 0:
                    continue
                pt = np.repeat(par, k) + rng.gamma(order, 1.0 / beta, k.sum())
                nt.append(pt)
                nc.append(np.full(pt.size, i))
        gt = np.concatenate(nt) if nt else np.array([])
        gc = np.concatenate(nc) if nc else np.array([], dtype=int)
        keep = gt < T
        gt, gc = gt[keep], gc[keep]
    t, c = np.concatenate(times), np.concatenate(chans)
    o = np.argsort(t)
    return t[o], c[o]


def q8_campbell() -> None:
    print("Q8  Stationary means of Erlang-2 states under a stationary Hawkes (Campbell)")
    rng = np.random.default_rng(17)
    mu = np.array([20.0, 10.0])
    A = np.array([[0.3, 0.1], [0.4, 0.2]])
    beta = 25.0
    T = 8000.0
    t, c = simulate_cluster(mu, A, beta, 2, T, rng)
    lam_inf = np.linalg.solve(np.eye(2) - A, mu)
    probes = rng.uniform(100.0, T, 40000)
    worst = 0.0
    for j in range(2):
        tj = t[c == j]
        R0 = R1 = 0.0
        for p in probes:
            lo = np.searchsorted(tj, p - 40.0 / beta)
            hi = np.searchsorted(tj, p)
            s = p - tj[lo:hi]
            R0 += np.sum(np.exp(-beta * s))
            R1 += np.sum(s * np.exp(-beta * s))
        R0 /= probes.size
        R1 /= probes.size
        worst = max(worst, abs(R0 - lam_inf[j] / beta) / (lam_inf[j] / beta),
                    abs(R1 - lam_inf[j] / beta**2) / (lam_inf[j] / beta**2))
    check("Monte-Carlo E[R0], E[R1] vs lam_inf/beta, lam_inf/beta^2 (rel)", worst, 0.03)


# --------------------------------------------------------------------------
# Full-mode experiments (Section 5.8): estimator bias / noise / lag frontier
# --------------------------------------------------------------------------
EXP_MU = np.array([12.0, 8.0])
EXP_BETA = 40.0
EXP_T = 1200.0
EXP_A1 = np.array([[0.30, 0.05], [0.10, 0.20]])
EXP_A2 = np.array([[0.30, 0.05], [0.55, 0.20]])


def simulate_shift(seed: int) -> list[tuple[float, int]]:
    """Ogata thinning, exp kernel, parameters evaluated at current time, A1->A2 at T/2."""
    r = np.random.default_rng(seed)
    t, R, ev = 0.0, np.zeros(2), []
    Amax = np.maximum(EXP_A1, EXP_A2)
    while t < EXP_T:
        A = EXP_A1 if t < EXP_T / 2 else EXP_A2
        lb = np.sum(EXP_MU + Amax @ (EXP_BETA * R))
        w = r.exponential(1 / lb)
        R *= math.exp(-EXP_BETA * w)
        t += w
        A = EXP_A1 if t < EXP_T / 2 else EXP_A2
        lam = EXP_MU + A @ (EXP_BETA * R)
        u = r.uniform() * lb
        if u < lam.sum():
            i = 0 if u < lam[0] else 1
            ev.append((t, i))
            R[i] += 1.0
    return ev


def est_em(ev, hl):
    kap, bk = math.log(2) / hl, EXP_BETA
    mu, A = np.ones(2), np.full((2, 2), 0.1)
    S0, S, Ex, Td, R, tl, out = np.zeros(2), np.zeros((2, 2)), np.zeros(2), 0.0, np.zeros(2), 0.0, []
    for tk, i in ev:
        d = tk - tl
        dsc = math.exp(-kap * d)
        Ex = dsc * (Ex + bk * R * d * psi(0, (bk - kap) * d))
        Td = dsc * Td + d * psi(0, kap * d)
        S0 *= dsc
        S *= dsc
        R *= math.exp(-bk * d)
        x = bk * R
        li = mu[i] + A[i] @ x
        S0[i] += mu[i] / li
        S[i] += A[i] * x / li
        R[i] += 1
        tl = tk
        mu = (S0 + 0.1) / (Td + 1.0)
        A = (S + 0.1) / (Ex[None, :] + 1.0)
        out.append((tk, A[1, 0]))
    return np.array(out)


def est_ls(ev, hl, hl_fast=None, dwell=0.0, h=np.inf, rho_up=1.5):
    ks = math.log(2) / hl
    kf = math.log(2) / hl_fast if hl_fast else ks
    bk, P = EXP_BETA, 3
    G, B, R, tl, out = np.zeros((P, P)), np.zeros((2, P)), np.zeros(2), 0.0, []
    th = np.array([[1, 0.1, 0.1]] * 2, float)
    Lam, last, C, fast_until, alarms = np.zeros(2), np.zeros(2), np.zeros((2, 2)), -1.0, 0
    for tk, i in ev:
        d = tk - tl
        kap = kf if tk < fast_until else ks
        dsc = math.exp(-kap * d)
        c = bk * R
        I00 = d * psi(0, kap * d)
        I01 = dsc * d * psi(0, (bk - kap) * d)
        I11 = dsc * d * psi(0, (2 * bk - kap) * d)
        M = np.array([[I00, c[0] * I01, c[1] * I01],
                      [c[0] * I01, c[0] ** 2 * I11, c[0] * c[1] * I11],
                      [c[1] * I01, c[0] * c[1] * I11, c[1] ** 2 * I11]])
        ths = np.maximum(th, [1e-3, 0, 0])
        e = math.exp(-bk * d)
        Lam += ths @ np.array([d, R[0] * (1 - e), R[1] * (1 - e)])
        G = dsc * G + M
        B *= dsc
        R *= e
        z = np.array([1, bk * R[0], bk * R[1]])
        B[i] += z
        R[i] += 1
        tl = tk
        th = np.linalg.solve(G + 1e-3 * np.eye(P), B.T).T
        xi = Lam[i] - last[i]
        last[i] = Lam[i]
        for s_, rr in enumerate((rho_up, 1 / rho_up)):
            C[i, s_] = max(0.0, C[i, s_] + math.log(rr) - (rr - 1) * xi)
        if C.max() > h:
            fast_until, alarms = tk + dwell, alarms + 1
            C[:] = 0
        out.append((tk, th[1, 1]))
    return np.array(out), alarms


def est_rml(ev, hl):
    kap, bk, P = math.log(2) / hl, EXP_BETA, 3
    th = np.array([[1, 0.1, 0.1]] * 2, float)
    H = np.array([np.eye(P)] * 2)
    R, tl, Zacc, tlast, out = np.zeros(2), 0.0, np.zeros((2, P)), np.zeros(2), []
    for tk, i in ev:
        d = tk - tl
        e = math.exp(-bk * d)
        Zacc += np.array([d, R[0] * (1 - e), R[1] * (1 - e)])
        R *= e
        z = np.array([1, bk * R[0], bk * R[1]])
        li = max(th[i] @ z, 1e-6)
        H[i] = math.exp(-kap * (tk - tlast[i])) * H[i] + np.outer(z, z) / li**2
        tlast[i] = tk
        th[i] = np.maximum(th[i] + np.linalg.solve(H[i], z / li - Zacc[i]), [1e-3, 0, 0])
        Zacc[i] = 0
        R[i] += 1
        tl = tk
        out.append((tk, th[1, 1]))
    return np.array(out)


def frontier_metrics(paths: list[np.ndarray]) -> tuple[float, float, float]:
    grid = np.arange(300.0, 599.0, 1.0)
    pre, lag = [], []
    mid = 0.5 * (EXP_A1[1, 0] + EXP_A2[1, 0])
    for o in paths:
        pre.append(o[np.searchsorted(o[:, 0], grid), 1])
        post = o[o[:, 0] > EXP_T / 2]
        cr = post[post[:, 1] >= mid]
        lag.append(cr[0, 0] - EXP_T / 2 if len(cr) else np.nan)
    pre = np.array(pre)
    return pre.mean() - EXP_A1[1, 0], pre.std(), float(np.nanmean(lag))


def full_experiments() -> None:
    print("\nF1  Estimator frontier on a step change alpha_{1<-0}: 0.10 -> 0.55 at t=600s")
    print("    (D=2, exp kernel beta=40/s, mu=(12,8)/s, T=1200s, 8 seeds; calm metrics on t in [300,599])")
    evs = [simulate_shift(100 + s) for s in range(8)]
    print(f"    events per run: mean {np.mean([len(e) for e in evs]):.0f}")
    print("    | estimator | half-life | bias | sd | RMSE | lag to 50% of step |")
    print("    |---|---|---|---|---|---|")
    for name, fn, hls in (("online EM", est_em, (3, 6, 12, 24)),
                          ("discounted LS", lambda e, h: est_ls(e, h)[0], (6, 12, 24, 48)),
                          ("recursive Newton ML", est_rml, (6, 12, 24, 48))):
        for hl in hls:
            b, sd, lag = frontier_metrics([fn(e, hl) for e in evs])
            print(f"    | {name} | {hl} s | {b:+.4f} | {sd:.4f} | {math.hypot(b, sd):.4f} | {lag:.1f} s |")
    print("\nF2  Discounted LS with residual-CUSUM adaptive forgetting (rho_up=1.5, two-sided)")
    print("    | slow half-life | fast half-life | dwell | threshold h | RMSE (calm) | lag to 50% | alarms / run |")
    print("    |---|---|---|---|---|---|---|")
    for (hs, hf, dw, h) in ((24, None, 0, np.inf), (24, 3, 6, 14), (48, None, 0, np.inf), (48, 4, 8, 14)):
        res = [est_ls(e, hs, hf, dw, h) for e in evs]
        b, sd, lag = frontier_metrics([r[0] for r in res])
        al = np.mean([r[1] for r in res])
        print(f"    | {hs} s | {hf if hf else '-'} | {dw if hf else '-'} | {h if hf else '-'} | {math.hypot(b, sd):.4f} | {lag:.1f} s | {al:.1f} |")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="also run the Section 5.8 experiments")
    args = ap.parse_args()
    t0 = time.time()
    for fn in (q1_erlang_recursion, q2_compensator, q4_lowrank_gram, q5_online_equals_batch,
               q6_potter_stability, q7_cw_and_projection, q8_campbell):
        fn()
    if args.full:
        full_experiments()
    print(f"\n{'ALL CHECKS PASSED' if not FAILED else 'FAILED: ' + ', '.join(FAILED)}  ({time.time() - t0:.0f}s)")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
