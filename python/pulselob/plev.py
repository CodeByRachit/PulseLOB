"""The .plev binary event-file format (docs/architecture.md Section 7.7).

A .plev file is a 64-byte header followed by a contiguous array of 64-byte MarketEvent records,
so the records can be memory-mapped and handed to the engine without copying. Instrument
metadata and provenance live in a JSON sidecar next to it (``<file>.plev.json``).

Header layout (little-endian):

    offset  size  field
         0     4  magic            b"PLEV"
         4     2  schema_version   MarketEvent schema version
         6     2  record_size      64
         8     4  header_size      64 (records start here)
        12     4  reserved         0
        16     8  n_records
        24     8  created_unix_ns
        32    32  reserved         0
"""

from __future__ import annotations

import json
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np

from .schema import MARKET_EVENT_DTYPE, SCHEMA_VERSION

MAGIC = b"PLEV"
HEADER_SIZE = 64
_HEADER = struct.Struct("<4sHHII QQ32s")
assert _HEADER.size == HEADER_SIZE


class PlevFormatError(ValueError):
    """The file is not a valid .plev file for this schema version."""


def sidecar_path(path: str | Path) -> Path:
    path = Path(path)
    return path.with_name(path.name + ".json")


def write_plev(path: str | Path, events: np.ndarray, metadata: dict[str, Any] | None = None) -> Path:
    """Write events (MARKET_EVENT_DTYPE) and an optional JSON sidecar; returns the .plev path."""
    path = Path(path)
    if events.dtype != MARKET_EVENT_DTYPE:
        raise TypeError("events must have dtype pulselob.MARKET_EVENT_DTYPE")
    events = np.ascontiguousarray(events)
    header = _HEADER.pack(MAGIC, SCHEMA_VERSION, MARKET_EVENT_DTYPE.itemsize, HEADER_SIZE, 0,
                          events.shape[0], time.time_ns(), bytes(32))
    with open(path, "wb") as f:
        f.write(header)
        f.write(events.tobytes(order="C"))
    if metadata is not None:
        sidecar_path(path).write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return path


def read_header(path: str | Path) -> dict[str, int]:
    with open(path, "rb") as f:
        raw = f.read(HEADER_SIZE)
    if len(raw) != HEADER_SIZE:
        raise PlevFormatError(f"{path}: file shorter than the {HEADER_SIZE}-byte header")
    magic, schema, record_size, header_size, _r0, n, created, _r1 = _HEADER.unpack(raw)
    if magic != MAGIC:
        raise PlevFormatError(f"{path}: bad magic {magic!r}")
    if schema != SCHEMA_VERSION:
        raise PlevFormatError(f"{path}: schema version {schema}, this build reads {SCHEMA_VERSION}")
    if record_size != MARKET_EVENT_DTYPE.itemsize or header_size != HEADER_SIZE:
        raise PlevFormatError(f"{path}: record size {record_size} / header size {header_size} unsupported")
    expected = HEADER_SIZE + n * record_size
    actual = Path(path).stat().st_size
    if actual != expected:
        raise PlevFormatError(f"{path}: size {actual} bytes, header implies {expected}")
    return {"schema_version": schema, "n_records": n, "created_unix_ns": created}


def open_plev(path: str | Path) -> tuple[np.memmap, dict[str, Any]]:
    """Memory-map a .plev file read-only. Returns (events, metadata); metadata includes the sidecar."""
    header = read_header(path)
    events = np.memmap(path, dtype=MARKET_EVENT_DTYPE, mode="r", offset=HEADER_SIZE,
                       shape=(header["n_records"],))
    meta: dict[str, Any] = dict(header)
    side = sidecar_path(path)
    if side.exists():
        meta["sidecar"] = json.loads(side.read_text())
    return events, meta
