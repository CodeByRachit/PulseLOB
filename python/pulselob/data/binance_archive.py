"""Historical Binance archives (data.binance.vision) to .plev (docs/data.md, option (c)).

Inputs, one file per venue, dataset and day:

* USD-M perpetual ``aggTrades`` and ``bookTicker`` (bookTicker is published only for
  2023-05-16 .. 2024-03-30, the window this loader targets);
* Spot ``aggTrades``. **Binance publishes no spot bookTicker archive at any date**, so the spot
  instrument has no top of book. Nothing is fabricated: no spot BookTop events are produced, the
  engine therefore marks spot trades with the neutral imbalance iota = 0 (and counts them in
  ``missing_book_marks``), and the sidecar records ``"spot_book": "unavailable"``. Quoting on spot
  is impossible on these data; Phase 5 quotes the perpetual instead.

Each file is downloaded once into a cache and verified against Binance's ``.CHECKSUM`` (SHA-256).
CSV rows are parsed exactly in C++ (``pulselob._core.parse_binance_csv``: decimal strings to e8
integers, no floating point; the timestamp unit is detected per file). The engine clock
``ts_local_ns`` is the exchange event time plus a deterministic per-venue latency. The venues are
merged by ``(ts_local_ns, instrument, seq)``, and heartbeats are inserted wherever the merged stream
is silent for longer than ``heartbeat_ns``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .. import _core
from ..plev import write_plev
from ..schema import MARKET_EVENT_DTYPE, EventKind, Flags

BASE_URL = "https://data.binance.vision/data"
PERP, SPOT = 0, 1
BOOK_TICKER_FIRST = dt.date(2023, 5, 16)
BOOK_TICKER_LAST = dt.date(2024, 3, 30)


def archive_url(market: str, dataset: str, symbol: str, day: dt.date | str) -> str:
    """URL of a daily archive. market: "spot" or "um" (USD-M futures); dataset: "aggTrades" or "bookTicker"."""
    if market not in ("spot", "um"):
        raise ValueError("market must be 'spot' or 'um'")
    if dataset not in ("aggTrades", "bookTicker"):
        raise ValueError("dataset must be 'aggTrades' or 'bookTicker'")
    if market == "spot" and dataset == "bookTicker":
        raise ValueError("Binance publishes no spot bookTicker archive")
    day = dt.date.fromisoformat(day) if isinstance(day, str) else day
    root = "spot" if market == "spot" else "futures/um"
    return f"{BASE_URL}/{root}/daily/{dataset}/{symbol}/{symbol}-{dataset}-{day.isoformat()}.zip"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


Fetch = Callable[[str, Path], None]


def _urlopen_fetch(url: str, dest: Path) -> None:
    with urllib.request.urlopen(url, timeout=60) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)


def download(url: str, cache_dir: Path, fetch: Fetch = _urlopen_fetch) -> Path:
    """Download `url` (and its .CHECKSUM) into cache_dir once; verify SHA-256; return the local path."""
    # Mirror the archive's directory layout: spot and futures files share file names
    # (BTCUSDT-aggTrades-<day>.zip), so a flat cache would confuse them.
    rel = url.split("/data/", 1)[1] if "/data/" in url else url.rsplit("/", 1)[1]
    dest = cache_dir / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    check = dest.with_name(dest.name + ".CHECKSUM")
    if not check.exists():
        fetch(url + ".CHECKSUM", check)
    expected = check.read_text().split()[0].strip().lower()
    if not dest.exists() or sha256(dest) != expected:
        tmp = dest.with_name(dest.name + ".part")
        fetch(url, tmp)
        got = sha256(tmp)
        if got != expected:
            tmp.unlink(missing_ok=True)
            raise IOError(f"checksum mismatch for {url}: expected {expected}, got {got}")
        tmp.replace(dest)
    return dest


def load_zip(path: Path, kind: str, instrument: int, latency_ns: int = 0) -> tuple[np.ndarray, dict]:
    """Parse the single CSV inside a Binance archive zip into MarketEvents (exact, in C++)."""
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if n.endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"{path}: expected one CSV member, found {names}")
        data = z.read(names[0])
    raw, stats = _core.parse_binance_csv(data, kind, instrument, int(latency_ns), True)
    stats["file"] = Path(path).name
    stats["sha256"] = sha256(Path(path))
    return raw.view(MARKET_EVENT_DTYPE), stats


def merge_streams(*arrays: np.ndarray) -> np.ndarray:
    """Merge event arrays by (ts_local_ns, instrument, seq): the engine's arrival-time order with the
    deterministic tie-break of architecture Section 7.4."""
    ev = np.concatenate([a for a in arrays if a.size]) if any(a.size for a in arrays) else np.empty(0, MARKET_EVENT_DTYPE)
    order = np.lexsort((ev["seq"], ev["instrument"], ev["ts_local_ns"]))
    return np.ascontiguousarray(ev[order])


def insert_heartbeats(ev: np.ndarray, heartbeat_ns: int) -> np.ndarray:
    """Insert Heartbeat events every heartbeat_ns of silence (deterministic; ts on the grid after each event)."""
    if ev.size < 2 or heartbeat_ns <= 0:
        return ev
    t = ev["ts_local_ns"]
    gaps = np.diff(t)
    counts = np.where(gaps > heartbeat_ns, (gaps - 1) // heartbeat_ns, 0)
    total = int(counts.sum())
    if total == 0:
        return ev
    hb = np.zeros(total, dtype=MARKET_EVENT_DTYPE)
    owners = np.repeat(np.arange(gaps.size), counts)
    within = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
    hb["ts_local_ns"] = t[owners] + (within + 1) * heartbeat_ns
    hb["ts_exchange_ns"] = hb["ts_local_ns"]
    hb["kind"] = EventKind.HEARTBEAT
    hb["flags"] = int(Flags.SYNTHETIC)
    out = np.concatenate([ev, hb])
    # Stable on ts_local: a heartbeat never precedes the event that opened its gap.
    order = np.argsort(out["ts_local_ns"], kind="stable")
    return np.ascontiguousarray(out[order])


@dataclass
class ArchiveSpec:
    symbol: str = "BTCUSDT"
    perp_latency_ns: int = 2_000_000  # deterministic one-way delays to a co-located receiver (configurable)
    spot_latency_ns: int = 2_000_000
    heartbeat_ns: int = 100_000_000
    perp_tick_e8: int = 10_000_000  # BTCUSDT: 0.10 on USD-M, 0.01 on spot (exchangeInfo, 2026-11-20)
    spot_tick_e8: int = 1_000_000


def build_dataset(days: list[dt.date | str], out: Path, cache_dir: Path, spec: ArchiveSpec = ArchiveSpec(),
                  fetch: Fetch = _urlopen_fetch, allow_outside_window: bool = False) -> dict:
    """Download, parse and merge perp aggTrades + perp bookTicker + spot aggTrades for each day into
    one .plev file with a provenance sidecar. Returns the sidecar metadata."""
    parts: list[np.ndarray] = []
    files: list[dict] = []
    for d in days:
        day = dt.date.fromisoformat(d) if isinstance(d, str) else d
        if not allow_outside_window and not BOOK_TICKER_FIRST <= day <= BOOK_TICKER_LAST:
            raise ValueError(f"{day}: the USD-M bookTicker archive covers {BOOK_TICKER_FIRST} .. {BOOK_TICKER_LAST} only")
        for market, dataset, inst, lat in (("um", "aggTrades", PERP, spec.perp_latency_ns),
                                           ("um", "bookTicker", PERP, spec.perp_latency_ns),
                                           ("spot", "aggTrades", SPOT, spec.spot_latency_ns)):
            url = archive_url(market, dataset, spec.symbol, day)
            path = download(url, cache_dir, fetch)
            ev, stats = load_zip(path, dataset, inst, lat)
            stats["url"] = url
            files.append(stats)
            parts.append(ev)
    merged = insert_heartbeats(merge_streams(*parts), spec.heartbeat_ns)
    meta = {
        "source": "data.binance.vision",
        "loader_version": _core.__version__,
        "symbol": spec.symbol,
        "days": [str(d) for d in days],
        "instruments": [
            {"id": PERP, "venue": "binance-usdm", "symbol": spec.symbol, "tick_size_e8": spec.perp_tick_e8,
             "role": "lead", "book": "available (bookTicker)"},
            {"id": SPOT, "venue": "binance-spot", "symbol": spec.symbol, "tick_size_e8": spec.spot_tick_e8,
             "role": "quote", "book": "unavailable"},
        ],
        "spot_book": "unavailable",
        "latency_model": {"perp_ns": spec.perp_latency_ns, "spot_ns": spec.spot_latency_ns,
                          "ts_local": "exchange event time + fixed per-venue latency"},
        "heartbeat_ns": spec.heartbeat_ns,
        "files": files,
        "records": int(merged.size),
    }
    write_plev(out, merged, meta)
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pulselob-archive",
                                 description="Convert Binance archive days (perp + spot) into one .plev file.")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--day", action="append", required=True, help="YYYY-MM-DD (repeatable)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default="data/binance-cache")
    ap.add_argument("--perp-latency-ms", type=float, default=2.0)
    ap.add_argument("--spot-latency-ms", type=float, default=2.0)
    ap.add_argument("--allow-outside-window", action="store_true")
    args = ap.parse_args(argv)
    spec = ArchiveSpec(symbol=args.symbol.upper(), perp_latency_ns=int(args.perp_latency_ms * 1e6),
                       spot_latency_ns=int(args.spot_latency_ms * 1e6))
    meta = build_dataset(args.day, Path(args.out), Path(args.cache), spec,
                         allow_outside_window=args.allow_outside_window)
    print(json.dumps({k: meta[k] for k in ("records", "days", "spot_book")}))
    return 0


__all__ = ["archive_url", "download", "load_zip", "merge_streams", "insert_heartbeats", "build_dataset", "ArchiveSpec"]
