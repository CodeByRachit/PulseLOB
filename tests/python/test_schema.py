import numpy as np
import pytest

import pulselob
from pulselob import MARKET_EVENT_DTYPE, EventCheck, EventKind, Side, TsResolution, make_flags
from pulselob import _core


def _trade(n: int) -> np.ndarray:
    ev = np.zeros(n, dtype=MARKET_EVENT_DTYPE)
    ev["ts_exchange_ns"] = 1_709_251_200_081_000_000 + np.arange(n)
    ev["ts_local_ns"] = ev["ts_exchange_ns"]
    ev["px0_e8"] = 14_044_000
    ev["px1_e8"] = 14_044_000
    ev["qty0_e8"] = 294_100_000_000
    ev["count"] = 1
    ev["kind"] = EventKind.TRADE
    ev["flags"] = make_flags(Side.BUY, TsResolution.MILLISECOND)
    return ev


def test_version_comes_from_version_file():
    from pathlib import Path

    version_file = Path(__file__).resolve().parents[2] / "VERSION"
    assert pulselob.__version__ == version_file.read_text().strip()


def test_dtype_matches_native_layout():
    assert MARKET_EVENT_DTYPE.itemsize == 64 == _core.MARKET_EVENT_ITEMSIZE
    native = [(n, o) for (n, o, _s, _c) in _core.market_event_layout()]
    assert native == [(n, MARKET_EVENT_DTYPE.fields[n][1]) for n in MARKET_EVENT_DTYPE.names]


def test_validate_is_zero_copy_and_correct():
    ev = _trade(1000)
    ev["px0_e8"][10] = 0                      # non-positive price
    ev["flags"][20] = make_flags(Side.NONE, TsResolution.MILLISECOND)  # trade without aggressor
    ev["px0_e8"][30] = ev["px1_e8"][30] - 1   # buy sweep walking down
    codes = pulselob.validate_events(ev)
    assert codes.shape == (1000,)
    assert codes[10] == EventCheck.NON_POSITIVE_PRICE
    assert codes[20] == EventCheck.BAD_SIDE
    assert codes[30] == EventCheck.PRICE_ORDER
    assert np.count_nonzero(codes) == 3


def test_validate_rejects_wrong_dtype_and_layout():
    with pytest.raises(ValueError, match="itemsize"):
        pulselob.validate_events(np.zeros(4, dtype=np.int64))
    ev = _trade(8)
    with pytest.raises(ValueError, match="C-contiguous"):
        pulselob.validate_events(ev[::2])
    with pytest.raises(ValueError, match="1-D"):
        pulselob.validate_events(ev.reshape(2, 4))
