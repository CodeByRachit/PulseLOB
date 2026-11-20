"""Python interface to the C++ engine (docs/architecture.md Section 10).

The batch path is zero-copy: `Engine.run` hands the event buffer (a NumPy array or a memory-mapped
.plev file) to C++ without copying, releases the GIL for the whole loop, and fills arrays that were
allocated before the call.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np

from . import _core
from .config import D, EngineConfig, n_params
from .plev import open_plev
from .schema import MARKET_EVENT_DTYPE, SIGNAL_DTYPE, heartbeat

EventsLike = Union[np.ndarray, str, os.PathLike]

ESTIMATORS = ("cholesky", "potter")
MODES = ("inline", "split")


@dataclass
class RunResult:
    """Output of Engine.run: one Signal per input event and, optionally, sampled model states."""

    signals: np.ndarray
    states: dict[str, np.ndarray] | None = None


def _as_events(events: EventsLike) -> np.ndarray:
    if isinstance(events, (str, os.PathLike)):
        arr, _meta = open_plev(Path(events))
        return arr
    arr = events
    if not isinstance(arr, np.ndarray) or arr.dtype != MARKET_EVENT_DTYPE:
        raise TypeError("events must be an array of pulselob.MARKET_EVENT_DTYPE or a path to a .plev file")
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr


class Engine:
    """The PulseLOB engine.

    Parameters
    ----------
    config:
        An EngineConfig (default: EngineConfig.default()).
    estimator:
        "cholesky" (information form, default) or "potter" (square-root form); math_spec 5.4.1.
    mode:
        "inline" (deterministic, single-threaded, default) or "split" (estimator on a second thread;
        signals use parameters published by that thread, so they are not reproducible run to run).
    """

    def __init__(self, config: EngineConfig | None = None, *, estimator: str = "cholesky", mode: str = "inline"):
        if estimator not in ESTIMATORS:
            raise ValueError(f"estimator must be one of {ESTIMATORS}")
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.config = config if config is not None else EngineConfig.default()
        self.estimator = estimator
        self.mode = mode
        self._e = _core.Engine(self.config.to_entries(), self.config.shape, estimator, mode)

    def __repr__(self) -> str:
        return f"Engine(shape={self.config.shape!r}, estimator={self.estimator!r}, mode={self.mode!r})"

    def run(self, events: EventsLike, *, state_every: int = 0) -> RunResult:
        """Process a batch of events in order.

        Returns one Signal record per event (SIGNAL_DTYPE). With state_every = k > 0, the model state
        after every k-th model step is also returned: served parameters theta_hat (D x P), the
        unconstrained estimate theta_tilde, branching matrix G, stationary intensities, criticality,
        effective sample size, CUSUM statistics and flags (bit 0 fast forgetting, bits 8-11 projected
        rows), with the event index and time of each sample.
        """
        ev = _as_events(events)
        n = int(ev.shape[0])
        signals = np.empty(n, dtype=SIGNAL_DTYPE)
        states = None
        arrays: dict[str, np.ndarray] = {}
        if state_every > 0:
            cap = n // state_every + 1
            p = n_params(self.config.shape)
            arrays = {
                "t_ns": np.empty(cap, np.int64),
                "index": np.empty(cap, np.int64),
                "theta_hat": np.empty((cap, D, p)),
                "theta_tilde": np.empty((cap, D, p)),
                "G": np.empty((cap, D, D)),
                "lambda_inf": np.empty((cap, D)),
                "criticality": np.empty(cap),
                "n_eff": np.empty(cap),
                "cusum": np.empty((cap, 2, D)),
                "flags": np.empty(cap, np.uint32),
            }
        rows = self._e.run_into(ev, signals, state_every, arrays, True)
        if state_every > 0:
            states = {k: v[:rows] for k, v in arrays.items()}
        return RunResult(signals, states)

    def step(self, event: np.ndarray | np.void) -> np.void:
        """Process one event (a MARKET_EVENT_DTYPE record or one-element array); returns its Signal."""
        ev = np.ascontiguousarray(np.asarray(event, dtype=MARKET_EVENT_DTYPE).reshape(1))
        out = np.empty(1, dtype=SIGNAL_DTYPE)
        self._e.run_into(ev, out, 0, {}, False)
        return out[0]

    def drain(self) -> None:
        """Split mode: wait until the estimator thread has consumed every event (no-op inline)."""
        self._e.run_into(np.empty(0, MARKET_EVENT_DTYPE), np.empty(0, SIGNAL_DTYPE), 0, {}, True)

    def heartbeat(self, ts_ns: int) -> np.void:
        """Advance time with no market event (math_spec Section 9)."""
        return self.step(heartbeat(ts_ns))

    def set_inventory(self, q: float) -> None:
        self._e.set_inventory(float(q))

    def counters(self) -> dict:
        return self._e.counters()

    def state(self) -> dict:
        """Current served parameters, diagnostics and (inline mode) the unconstrained estimate."""
        return self._e.state()
