"""NumPy mirrors of the C++ MarketEvent and Signal records (docs/architecture.md Section 7).

The dtypes are written out independently of the C++ headers and checked against the layouts the
compiled module reports, so a change on one side without the other fails at import time.
"""

from __future__ import annotations

from enum import IntEnum, IntFlag

import numpy as np

from . import _core

SCHEMA_VERSION = 1

MARKET_EVENT_DTYPE = np.dtype(
    {
        "names": [
            "ts_exchange_ns",
            "ts_local_ns",
            "px0_e8",
            "qty0_e8",
            "px1_e8",
            "qty1_e8",
            "seq",
            "count",
            "instrument",
            "kind",
            "flags",
        ],
        "formats": ["<i8", "<i8", "<i8", "<i8", "<i8", "<i8", "<u8", "<u4", "<u2", "u1", "u1"],
        "offsets": [0, 8, 16, 24, 32, 40, 48, 56, 60, 62, 63],
        "itemsize": 64,
    }
)

E8 = 100_000_000


class EventKind(IntEnum):
    HEARTBEAT = 0
    TRADE = 1
    BOOK_TOP = 2
    STATUS = 3


class Side(IntEnum):
    NONE = 0
    BUY = 1
    SELL = 2


class TsResolution(IntEnum):
    NANOSECOND = 0
    MICROSECOND = 1
    MILLISECOND = 2
    SECOND = 3


class StatusCode(IntEnum):
    FEED_GAP = 1
    DISCONNECTED = 2
    RECONNECTED = 3
    RING_OVERFLOW = 4


class Flags(IntFlag):
    SIDE_MASK = 0x03
    RESOLUTION_MASK = 0x0C
    TS_ADJUSTED = 0x10
    SYNTHETIC = 0x20
    SNAPSHOT = 0x40
    RESERVED = 0x80


class EventCheck(IntEnum):
    OK = 0
    BAD_KIND = 1
    RESERVED_FLAG = 2
    BAD_SIDE = 3
    NON_POSITIVE_PRICE = 4
    NON_POSITIVE_QUANTITY = 5
    BAD_COUNT = 6
    PRICE_ORDER = 7
    CROSSED_BOOK = 8
    NON_ZERO_PAYLOAD = 9
    NEGATIVE_NOTIONAL = 10


def make_flags(side: Side, resolution: TsResolution, extra: int = 0) -> int:
    """Pack the aggressor side, timestamp resolution and extra flag bits into the flags byte."""
    return int(side) | (int(resolution) << 2) | int(extra)


SIGNAL_DTYPE = np.dtype(
    {
        "names": ["t_ns", "flags", "status", "reserved", "lambda", "tau_buy", "tau_sell", "criticality",
                  "reservation_shift", "adverse_ask", "adverse_bid", "sigma2", "bid_px_e8", "ask_px_e8"],
        "formats": ["<i8", "<u4", "<u2", "<u2", ("<f8", (4,)), "<f8", "<f8", "<f8", "<f8", "<f8", "<f8", "<f8",
                    "<i8", "<i8"],
        "offsets": [0, 8, 12, 14, 16, 48, 56, 64, 72, 80, 88, 96, 104, 112],
        "itemsize": 128,
    }
)
"""One engine output record per input event (architecture Section 7.6)."""


class SignalFlags(IntFlag):
    WARMING_UP = 1 << 0
    ASK_PULLED = 1 << 1
    BID_PULLED = 1 << 2
    KILL_SWITCH = 1 << 3
    FAST_FORGETTING = 1 << 4
    PROJECTION_ACTIVE = 1 << 5
    NO_QUOTE_BOOK = 1 << 6
    STALE_PARAMETERS = 1 << 7


class EngineStatus(IntEnum):
    """Low byte of Signal.status; the high byte is the EventCheck code of rejected input."""

    ACCEPTED = 0
    CLAMPED = 1
    REJECTED_INVALID = 2
    REJECTED_OUT_OF_ORDER = 3
    IGNORED = 4
    REJECTED_NON_FINITE = 5


def decimal_to_e8(text: str) -> int:
    """Exact decimal string to e8 integer units (the Python twin of core::parse_decimal_e8).

    Raises ValueError for malformed input or a nonzero digit beyond the 8th decimal.
    """
    s = text.strip()
    neg = s.startswith("-")
    if s[:1] in "+-":
        s = s[1:]
    whole, _, frac = s.partition(".")
    if (not whole and not frac) or not (whole or "0").isdigit() or (frac and not frac.isdigit()):
        raise ValueError(f"malformed decimal {text!r}")
    if len(frac) > 8:
        if frac[8:].strip("0"):
            raise ValueError(f"{text!r} has more than 8 significant decimals")
        frac = frac[:8]
    value = int(whole or "0") * E8 + int(frac.ljust(8, "0") or "0")
    if value > 2**63 - 1:
        raise ValueError(f"{text!r} out of range")
    return -value if neg else value


def _record(**fields: int) -> np.ndarray:
    ev = np.zeros(1, dtype=MARKET_EVENT_DTYPE)
    for k, v in fields.items():
        ev[k] = v
    return ev


def trade(ts_ns: int, instrument: int, side: Side, price_e8: int, qty_e8: int, *, ts_exchange_ns: int | None = None,
          last_price_e8: int | None = None, count: int = 1, seq: int = 0,
          resolution: TsResolution = TsResolution.NANOSECOND) -> np.ndarray:
    """A one-record Trade array (for Engine.step and tests)."""
    last = price_e8 if last_price_e8 is None else last_price_e8
    return _record(ts_exchange_ns=ts_ns if ts_exchange_ns is None else ts_exchange_ns, ts_local_ns=ts_ns, px0_e8=last,
                   qty0_e8=qty_e8, px1_e8=price_e8, qty1_e8=price_e8 * qty_e8 // E8, seq=seq, count=count,
                   instrument=instrument, kind=EventKind.TRADE, flags=make_flags(side, resolution))


def book_top(ts_ns: int, instrument: int, bid_e8: int, bid_qty_e8: int, ask_e8: int, ask_qty_e8: int, *,
             ts_exchange_ns: int | None = None, seq: int = 0) -> np.ndarray:
    """A one-record BookTop array."""
    return _record(ts_exchange_ns=ts_ns if ts_exchange_ns is None else ts_exchange_ns, ts_local_ns=ts_ns,
                   px0_e8=bid_e8, qty0_e8=bid_qty_e8, px1_e8=ask_e8, qty1_e8=ask_qty_e8, seq=seq,
                   instrument=instrument, kind=EventKind.BOOK_TOP, flags=0)


def heartbeat(ts_ns: int) -> np.ndarray:
    """A one-record Heartbeat array."""
    return _record(ts_exchange_ns=ts_ns, ts_local_ns=ts_ns, kind=EventKind.HEARTBEAT)


def status(ts_ns: int, code: StatusCode, instrument: int = 0) -> np.ndarray:
    """A one-record Status array (feed gap, disconnect, reconnect, ring overflow)."""
    return _record(ts_exchange_ns=ts_ns, ts_local_ns=ts_ns, count=int(code), instrument=instrument,
                   kind=EventKind.STATUS)


def _check_against_native() -> None:
    native = _core.market_event_layout()
    if _core.MARKET_EVENT_SCHEMA_VERSION != SCHEMA_VERSION:
        raise ImportError(
            f"schema version mismatch: C++ {_core.MARKET_EVENT_SCHEMA_VERSION}, Python {SCHEMA_VERSION}"
        )
    if MARKET_EVENT_DTYPE.itemsize != _core.MARKET_EVENT_ITEMSIZE:
        raise ImportError("MarketEvent itemsize differs between C++ and Python")
    python_layout = [
        (name, MARKET_EVENT_DTYPE.fields[name][1], MARKET_EVENT_DTYPE.fields[name][0].itemsize,
         MARKET_EVENT_DTYPE.fields[name][0].str.replace("|", ""))
        for name in MARKET_EVENT_DTYPE.names
    ]
    native_layout = [(n, o, s, c.replace("|", "")) for (n, o, s, c) in native]
    if python_layout != native_layout:
        raise ImportError(f"MarketEvent layout mismatch:\n C++    {native_layout}\n Python {python_layout}")
    if SIGNAL_DTYPE.itemsize != _core.SIGNAL_ITEMSIZE:
        raise ImportError("Signal itemsize differs between C++ and Python")
    sig_python = []
    for name in SIGNAL_DTYPE.names:
        dt, off = SIGNAL_DTYPE.fields[name][:2]
        base = dt.base if dt.subdtype else dt
        sig_python.append((name, off, dt.itemsize, base.str.replace("|", "")))
    sig_native = [(n, o, s, c.replace("|", "")) for (n, o, s, c) in _core.signal_layout()]
    if sig_python != sig_native:
        raise ImportError(f"Signal layout mismatch:\n C++    {sig_native}\n Python {sig_python}")


_check_against_native()
