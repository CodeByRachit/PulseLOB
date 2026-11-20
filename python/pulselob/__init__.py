"""PulseLOB: online cross-venue order-flow toxicity with marked Hawkes processes."""

from pathlib import Path

from ._core import MARKET_EVENT_SCHEMA_VERSION, __version__, validate_events
from .config import EngineConfig
from .engine import Engine, RunResult
from .plev import PlevWriter, open_plev, read_header, recover_plev, write_plev
from .schema import (
    MARKET_EVENT_DTYPE,
    SIGNAL_DTYPE,
    EngineStatus,
    EventCheck,
    EventKind,
    Flags,
    Side,
    SignalFlags,
    StatusCode,
    TsResolution,
    book_top,
    decimal_to_e8,
    heartbeat,
    make_flags,
    status,
    trade,
)


def native_replay_path() -> Path:
    """Path of the native `pulselob-replay` executable shipped in the wheel."""
    return Path(__file__).resolve().parent / "bin" / "pulselob-replay"


__all__ = [
    "__version__",
    "MARKET_EVENT_SCHEMA_VERSION",
    "MARKET_EVENT_DTYPE",
    "SIGNAL_DTYPE",
    "Engine",
    "EngineConfig",
    "EngineStatus",
    "EventCheck",
    "EventKind",
    "Flags",
    "PlevWriter",
    "RunResult",
    "Side",
    "SignalFlags",
    "StatusCode",
    "TsResolution",
    "book_top",
    "decimal_to_e8",
    "heartbeat",
    "make_flags",
    "native_replay_path",
    "open_plev",
    "read_header",
    "recover_plev",
    "status",
    "trade",
    "validate_events",
    "write_plev",
]
