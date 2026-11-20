import subprocess
import threading
import time

import numpy as np
import pytest

import pulselob as pl
from pulselob import synth
from pulselob.config import EngineConfig


@pytest.fixture(scope="module")
def stream():
    return synth.generate("lead_lag", seed=3, horizon_s=40.0).events


def test_config_text_round_trip_and_validation():
    cfg = EngineConfig.default(cusum_threshold=25.0, rho_max=0.95)
    back = EngineConfig.from_text(cfg.to_text())
    assert back.to_entries() == cfg.to_entries()
    with pytest.raises(ValueError, match="S4"):
        pl.Engine(EngineConfig.default(rho_max=0.999))
    with pytest.raises(ValueError, match="I2"):
        pl.Engine(EngineConfig.default(half_life_fast_s=0.5))  # kappa_fast above beta_min = 1
    with pytest.raises(ValueError):
        pl.Engine(estimator="newton")
    with pytest.raises(ValueError):
        EngineConfig.default("cubic")


def test_run_shapes_and_dtypes(stream):
    eng = pl.Engine()
    res = eng.run(stream, state_every=50)
    assert res.signals.dtype == pl.SIGNAL_DTYPE and res.signals.shape == stream.shape
    st = res.states
    assert st["theta_hat"].shape[1:] == (4, 17) and st["G"].shape[1:] == (4, 4)
    assert len(st["t_ns"]) == eng.counters()["model_steps"] // 50
    assert np.all(np.diff(st["index"]) > 0)
    assert np.isfinite(st["theta_tilde"]).all()
    assert np.all(st["theta_hat"][:, :, 1:] >= 0.0)
    assert (res.signals["tau_buy"] * res.signals["tau_sell"] == 0.0).all()


def same_fields(a: np.ndarray, b: np.ndarray) -> bool:
    """Bitwise equality of every named field (records built from np.void lists leave padding unset)."""
    return all(a[n].tobytes() == b[n].tobytes() for n in a.dtype.names)


def test_step_matches_run_bit_for_bit(stream):
    a = pl.Engine().run(stream).signals
    eng = pl.Engine()
    b = np.array([eng.step(e) for e in stream[:3000]], dtype=pl.SIGNAL_DTYPE)
    assert same_fields(a[:3000], b)


def test_memory_mapped_plev_is_read_in_place(stream, tmp_path):
    path = pl.write_plev(tmp_path / "s.plev", stream)
    mm, _ = pl.open_plev(path)
    assert isinstance(mm, np.memmap) and not mm.flags.writeable
    assert pl.Engine().run(mm).signals.tobytes() == pl.Engine().run(stream).signals.tobytes()
    assert pl.Engine().run(path).signals.tobytes() == pl.Engine().run(stream).signals.tobytes()


@pytest.mark.parametrize("shape,estimator,mode", [("default", "cholesky", "inline"), ("default", "potter", "inline"),
                                                  ("exp3", "cholesky", "inline")])
def test_native_replay_parity(stream, tmp_path, shape, estimator, mode):
    """The same .plev through the native C++ driver and through Python: identical bytes."""
    exe = pl.native_replay_path()
    assert exe.exists(), "pulselob-replay is installed with the wheel"
    cfg = EngineConfig.default(shape)
    (tmp_path / "cfg.txt").write_text(cfg.to_text())
    path = pl.write_plev(tmp_path / "s.plev", stream)
    out = tmp_path / "native.sig"
    subprocess.run([str(exe), "--config", str(tmp_path / "cfg.txt"), "--input", str(path), "--output", str(out),
                    "--shape", shape, "--estimator", estimator, "--mode", mode], check=True, capture_output=True)
    native = np.fromfile(out, dtype=pl.SIGNAL_DTYPE)
    py = pl.Engine(cfg, estimator=estimator, mode=mode).run(path).signals
    assert native.shape == py.shape
    assert native.tobytes() == py.tobytes()


def test_split_mode_parameters_match_inline_after_run(stream):
    inline = pl.Engine()
    split = pl.Engine(mode="split")
    inline.run(stream)
    res = split.run(stream, state_every=100)
    a, b = inline.state(), split.state()
    assert np.array_equal(a["theta_hat"], b["theta_hat"]) and np.array_equal(a["lambda_inf"], b["lambda_inf"])
    assert np.isnan(res.states["theta_tilde"]).all()  # not available on the hot thread
    assert split.counters()["ring_overflows"] == 0


def test_gil_is_released_during_run():
    big = synth.generate("two_sided", seed=5, horizon_s=120.0).events
    eng = pl.Engine()
    done = threading.Event()
    ticks = 0

    def work():
        eng.run(big)
        done.set()

    th = threading.Thread(target=work)
    t0 = time.perf_counter()
    th.start()
    while not done.is_set():
        ticks += 1
    th.join()
    elapsed = time.perf_counter() - t0
    assert elapsed > 0.05
    # With the GIL held for the whole run this loop could not advance.
    assert ticks > 10_000


def test_input_validation(stream):
    eng = pl.Engine()
    with pytest.raises(TypeError):
        eng.run(np.zeros(4, dtype=np.float64))
    with pytest.raises(ValueError, match="contiguous"):
        eng.run(stream[::2])


def test_inventory_heartbeat_and_counters(stream):
    eng = pl.Engine()
    eng.run(stream)
    t = int(stream["ts_local_ns"][-1])
    eng.set_inventory(10.0)
    sig = eng.heartbeat(t + 1_000_000)
    assert sig["bid_px_e8"] == 0  # at the inventory limit the bid is withdrawn
    eng.set_inventory(float("nan"))
    c = eng.counters()
    assert c["nonfinite_inputs"] == 1 and c["model_heartbeats"] >= 1
    assert c["factorization_failures"] == 0
