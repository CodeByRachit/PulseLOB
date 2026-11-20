import math

import numpy as np
import pytest

import pulselob as pl
from pulselob import synth
from pulselob.schema import EventKind, SignalFlags

BETA = np.array(synth.BETA)
ORDERS = (1, 2, 2, 1)
Q = 4


def replay_truth(sc: synth.Scenario, t_s: np.ndarray, ch: np.ndarray):
    """Independent Python recursion: left-limit intensities at each event and, per channel, the
    compensator between its consecutive events (time-rescaling residuals), with the scenario's
    piecewise parameters. Interval integrals are closed forms of the Erlang kernels."""
    starts = np.asarray(sc.segment_starts)
    R0 = np.zeros((4, Q))
    R1 = np.zeros((4, Q))
    lam_left = np.empty((t_s.size, 4))
    since = np.zeros(4)
    residuals = [[] for _ in range(4)]
    t = 0.0
    for k, (tk, c) in enumerate(zip(t_s, ch)):
        while t < tk:  # integrate piece by piece across segment boundaries
            seg = int(np.searchsorted(starts, t, side="right")) - 1
            end = min(tk, starts[seg + 1]) if seg + 1 < starts.size else tk
            th = sc.thetas[seg]
            dt = end - t
            x = BETA * dt
            e = np.exp(-x)
            integ = np.empty((4, Q))
            for q in range(Q):
                if ORDERS[q] == 1:
                    integ[:, q] = R0[:, q] * (1 - e[q])
                else:
                    integ[:, q] = BETA[q] * R1[:, q] * (1 - e[q]) + R0[:, q] * (1 - e[q] * (1 + x[q]))
            since += th[:, 0] * dt + (th[:, 1:].reshape(4, 4, Q) * integ[None, :, :]).sum(axis=(1, 2))
            R1 = e * (R1 + dt * R0)
            R0 = e * R0
            t = end
        seg = int(np.searchsorted(starts, tk, side="right")) - 1
        th = sc.thetas[seg]
        x_now = np.where(np.array(ORDERS) == 1, BETA * R0, BETA**2 * R1)
        lam_left[k] = th[:, 0] + (th[:, 1:].reshape(4, 4, Q) * x_now[None, :, :]).sum(axis=(1, 2))
        residuals[c].append(since[c])
        since[c] = 0.0
        R0[c] += 1.0
    return lam_left, [np.array(r[1:]) for r in residuals]  # first residual includes the start-up


def ks_exp1(x: np.ndarray) -> float:
    x = np.sort(x)
    n = x.size
    cdf = 1.0 - np.exp(-x)
    return float(max(np.max(np.arange(1, n + 1) / n - cdf), np.max(cdf - np.arange(n) / n)))


def test_seed_determinism():
    a = synth.generate("lead_lag", seed=9, horizon_s=30.0)
    b = synth.generate("lead_lag", seed=9, horizon_s=30.0)
    c = synth.generate("lead_lag", seed=10, horizon_s=30.0)
    assert a.events.tobytes() == b.events.tobytes()
    assert np.array_equal(a.lambda_true, b.lambda_true)
    assert a.events.tobytes() != c.events.tobytes()
    assert (pl.validate_events(a.events) == 0).all()
    assert np.all(np.diff(a.events["ts_local_ns"]) >= 0)


@pytest.mark.parametrize("name,kwargs", [("calm", {"horizon_s": 200.0}), ("regime_shift", {"horizon_s": 120.0}),
                                         ("lead_lag", {"horizon_s": 60.0})])
def test_ground_truth_and_time_rescaling(name, kwargs):
    ds = synth.generate(name, seed=4, **kwargs)
    lam, res = replay_truth(ds.scenario, ds.t_s, ds.channel)
    # Recorded ground-truth intensities equal an independent recursion.
    assert np.max(np.abs(lam - ds.lambda_true) / lam) < 1e-9
    # Time-rescaling theorem: compensator increments between events are Exp(1).
    pooled = np.concatenate(res)
    n = pooled.size
    assert abs(pooled.mean() - 1.0) < 5.0 / math.sqrt(n)
    assert ks_exp1(pooled) < 1.95 / math.sqrt(n)  # 0.1% critical value of the KS statistic


def test_counts_match_stationary_rates():
    sc = synth.calm(horizon_s=600.0)
    ds = synth.generate(sc, seed=2)
    th = sc.thetas[0]
    lam_inf = synth.stationary_rates(th)
    rho = max(abs(np.linalg.eigvals(synth.branching(th))))
    counts = np.bincount(ds.channel, minlength=4)
    expected = lam_inf * sc.horizon_s
    # Clustering inflates the count variance by about 1 / (1 - rho)^2.
    assert np.all(np.abs(counts - expected) < 5.0 * np.sqrt(expected) / (1.0 - rho))


def test_write_plev_sidecar_and_truth(tmp_path):
    ds = synth.generate("flash_crash", seed=1, horizon_s=70.0)
    path = synth.write(ds, tmp_path / "crash.plev")
    ev, meta = pl.open_plev(path)
    assert ev.tobytes() == ds.events.tobytes()
    assert meta["sidecar"]["scenario"]["jumps_s"] == [60.0, 62.0]
    assert meta["sidecar"]["scenario"]["segments"][1]["spectral_radius"] < 1.0
    truth = synth.load_truth(path)
    assert truth["lambda_true"].shape == (ds.t_s.size, 4)
    # trade_index points at the Trade record of every channel event, in order.
    trades = ds.events[truth["trade_index"]]
    assert (trades["kind"] == EventKind.TRADE).all()
    assert np.array_equal(trades["instrument"], (truth["channel"] >= 2).astype(trades["instrument"].dtype))


def test_constant_marks_give_unit_marks():
    ds = synth.generate("calm", seed=1, horizon_s=20.0, constant_marks=True)
    trades = ds.events[ds.events["kind"] == EventKind.TRADE]
    assert (trades["qty0_e8"] == 100_000_000).all() and (trades["px0_e8"] == trades["px1_e8"]).all()


# --- Scenarios through the engine (Phase 5 benchmark behavior) ---------------------------------
def _tau(name, **kw):
    ds = synth.generate(name, seed=1, **kw)
    s = pl.Engine().run(ds.events).signals
    tr = ds.events["kind"] == EventKind.TRADE
    return ds, s, np.maximum(s["tau_buy"], s["tau_sell"])[tr]


def test_two_sided_noise_does_not_look_toxic():
    _, _, tau = _tau("two_sided", horizon_s=60.0)
    assert np.quantile(tau, 0.99) < 0.05


def test_lead_lag_bursts_are_toxic():
    _, _, tau = _tau("lead_lag", horizon_s=120.0)
    assert np.mean(tau > 0.5) > 0.2


def test_flash_crash_raises_sell_toxicity_and_pulls_the_bid():
    ds, s, _ = _tau("flash_crash")
    t = (s["t_ns"] - ds.events["ts_local_ns"][0]) / 1e9
    w = (t >= 60.0) & (t < 62.0) & (ds.events["kind"] == EventKind.TRADE)
    assert s["tau_sell"][w].mean() > 0.5
    assert np.mean((s["flags"][w] & SignalFlags.BID_PULLED) > 0) > 0.9
    assert (s["tau_buy"][w] == 0.0).mean() > 0.9


def test_regime_shift_is_tracked():
    ds = synth.generate("regime_shift", seed=1, constant_marks=True)
    st = pl.Engine().run(ds.events, state_every=200).states
    t = (st["t_ns"] - ds.events["ts_local_ns"][0]) / 1e9
    g = st["G"][:, 2, 0]  # served G(spot buy <- perp buy); truth 0.10 -> 0.50 at 300 s
    assert abs(g[(t > 200) & (t < 300)].mean() - 0.10) < 0.08
    assert abs(g[t > 500].mean() - 0.50) < 0.10
