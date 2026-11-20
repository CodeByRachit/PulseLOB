"""NumPy mirror of the C++ MarketEvent schema (docs/architecture.md Section 7).

The dtype is written out independently of the C++ header and checked against the layout the
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


_check_against_native()
