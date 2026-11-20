"""Live cross-venue recorder for Binance Spot and USD-M perpetual futures (docs/data.md).

Three WebSocket connections run in one asyncio loop:

* spot: ``<spot_ws>/stream?streams=<s>@aggTrade/<s>@bookTicker&timeUnit=MICROSECOND``
* futures trades: ``<futures_ws>/market/stream?streams=<s>@aggTrade``
* futures book: ``<futures_ws>/public/stream?streams=<s>@bookTicker``

(verified 2026-11-20: USD-M ``aggTrade`` is delivered only on the ``/market`` path; the legacy
``/stream`` and the ``/public`` paths accept the subscription but stay silent, which is why the
recorder also has a silence watchdog).

Every message is stamped on receipt with a local arrival time: monotonic nanoseconds anchored to
the UTC wall clock at start-up. That time is the engine clock ``ts_local_ns`` (architecture
Section 7.4). The normalizer turns messages into MarketEvent records:

* aggTrade -> Trade; consecutive prints with equal instrument, side and exchange time are merged
  into one sweep (math_spec Section 2.2), stamped with the arrival of the last part;
* bookTicker -> BookTop; spot bookTicker carries no exchange time, so its ts_exchange_ns is the
  arrival time with the TS_ADJUSTED flag;
* aggregate-trade id gaps -> a Status(FEED_GAP) event; repeated ids (after a reconnect) are dropped;
* connection loss -> Status(DISCONNECTED), reconnection -> Status(RECONNECTED), with exponential
  back-off; a stream silent for longer than ``silence_timeout_s`` is treated as a lost connection.

Records are written to a .plev file through PlevWriter (valid after every flush), with a JSON
sidecar holding the instrument table, provenance and the run summary.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import signal as _signal
import sys
import time
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

import numpy as np

from .. import _core
from ..plev import PlevWriter
from ..schema import (MARKET_EVENT_DTYPE, E8, EventCheck, EventKind, Flags, Side, StatusCode, TsResolution,
                      decimal_to_e8, make_flags)

PERP, SPOT = 0, 1


class Clock:
    """Monotonic nanoseconds anchored to the UTC wall clock at construction."""

    def __init__(self) -> None:
        self._wall0 = time.time_ns()
        self._mono0 = time.monotonic_ns()

    def now_ns(self) -> int:
        return self._wall0 + (time.monotonic_ns() - self._mono0)


@dataclass
class InstrumentSpec:
    id: int
    venue: str     # "binance-usdm" or "binance-spot"
    symbol: str
    tick_size_e8: int
    role: str      # "lead" or "quote"


@dataclass
class NormalizerStats:
    trades: int = 0
    trade_prints: int = 0
    merged: int = 0
    books: int = 0
    statuses: int = 0
    gaps: int = 0
    missing_trades: int = 0
    duplicates: int = 0
    invalid: int = 0
    unparsed: int = 0
    by_instrument: Counter = field(default_factory=Counter)


class Normalizer:
    """Turns Binance stream payloads into MarketEvent records (transport-independent; unit-tested)."""

    def __init__(self) -> None:
        self.stats = NormalizerStats()
        self._pending: np.ndarray | None = None  # sweep being merged (one record)
        self._last_agg_id: dict[int, int] = {}

    # -- helpers --------------------------------------------------------------------------------
    @staticmethod
    def _record(**fields: Any) -> np.ndarray:
        r = np.zeros(1, dtype=MARKET_EVENT_DTYPE)
        for k, v in fields.items():
            r[k] = v
        return r

    def _flush_pending(self, out: list[np.ndarray]) -> None:
        if self._pending is not None:
            self._emit(self._pending, out)
            self._pending = None

    def _emit(self, rec: np.ndarray, out: list[np.ndarray]) -> None:
        code = int(_core.validate_events(rec)[0])
        if code != EventCheck.OK:
            self.stats.invalid += 1
            return
        kind = int(rec["kind"][0])
        if kind == EventKind.TRADE:
            self.stats.trades += 1
        elif kind == EventKind.BOOK_TOP:
            self.stats.books += 1
        elif kind == EventKind.STATUS:
            self.stats.statuses += 1
        self.stats.by_instrument[int(rec["instrument"][0])] += 1
        out.append(rec)

    # -- inputs ---------------------------------------------------------------------------------
    def status(self, code: StatusCode, instrument: int, arrival_ns: int) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        self._flush_pending(out)
        self._emit(self._record(ts_exchange_ns=arrival_ns, ts_local_ns=arrival_ns, count=int(code),
                                instrument=instrument, kind=EventKind.STATUS), out)
        return out

    def flush(self) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        self._flush_pending(out)
        return out

    def on_message(self, instrument: int, data: dict, arrival_ns: int, time_unit: str = "ms") -> list[np.ndarray]:
        """One stream payload (the "data" object of a combined-stream message)."""
        out: list[np.ndarray] = []
        try:
            if data.get("e") == "aggTrade":
                self._agg_trade(instrument, data, arrival_ns, time_unit, out)
            elif "b" in data and "a" in data and "u" in data:
                self._flush_pending(out)
                self._book(instrument, data, arrival_ns, time_unit, out)
            else:
                self.stats.unparsed += 1
        except (KeyError, ValueError, TypeError, OverflowError):  # malformed or out-of-range payload
            self.stats.unparsed += 1
        return out

    def _agg_trade(self, inst: int, d: dict, arrival_ns: int, unit: str, out: list[np.ndarray]) -> None:
        agg_id = int(d["a"])
        last = self._last_agg_id.get(inst)
        if last is not None and agg_id <= last:
            self.stats.duplicates += 1  # replayed after a reconnect
            return
        if last is not None and agg_id > last + 1:
            self.stats.gaps += 1
            self.stats.missing_trades += agg_id - last - 1
            self._flush_pending(out)
            self._emit(self._record(ts_exchange_ns=arrival_ns, ts_local_ns=arrival_ns, count=int(StatusCode.FEED_GAP),
                                    instrument=inst, kind=EventKind.STATUS), out)
        self._last_agg_id[inst] = agg_id
        scale = 1_000 if unit == "us" else 1_000_000
        res = TsResolution.MICROSECOND if unit == "us" else TsResolution.MILLISECOND
        ts = int(d["T"]) * scale
        side = Side.SELL if d["m"] else Side.BUY  # buyer is the maker: the aggressor sold
        px = decimal_to_e8(d["p"])
        qty = decimal_to_e8(d["q"])
        prints = int(d["l"]) - int(d["f"]) + 1
        notional = round(px * qty / E8)
        self.stats.trade_prints += prints
        p = self._pending
        if (p is not None and int(p["instrument"][0]) == inst and int(p["ts_exchange_ns"][0]) == ts
                and int(p["flags"][0]) & Flags.SIDE_MASK == side):
            worst = int(p["px0_e8"][0])
            p["px0_e8"] = max(worst, px) if side == Side.BUY else min(worst, px)
            p["qty0_e8"] = int(p["qty0_e8"][0]) + qty
            p["qty1_e8"] = int(p["qty1_e8"][0]) + notional
            p["count"] = int(p["count"][0]) + prints
            p["ts_local_ns"] = arrival_ns  # complete only once its last part has arrived
            self.stats.merged += 1
            return
        self._flush_pending(out)
        self._pending = self._record(ts_exchange_ns=ts, ts_local_ns=arrival_ns, px0_e8=px, qty0_e8=qty, px1_e8=px,
                                     qty1_e8=notional, seq=agg_id, count=prints, instrument=inst,
                                     kind=EventKind.TRADE, flags=make_flags(side, res))

    def _book(self, inst: int, d: dict, arrival_ns: int, unit: str, out: list[np.ndarray]) -> None:
        if "T" in d:  # USD-M futures: transaction time
            scale = 1_000 if unit == "us" else 1_000_000
            ts = int(d["T"]) * scale
            flags = make_flags(Side.NONE, TsResolution.MICROSECOND if unit == "us" else TsResolution.MILLISECOND)
        else:  # spot bookTicker has no exchange time
            ts = arrival_ns
            flags = make_flags(Side.NONE, TsResolution.NANOSECOND, Flags.TS_ADJUSTED)
        self._emit(self._record(ts_exchange_ns=ts, ts_local_ns=arrival_ns, px0_e8=decimal_to_e8(d["b"]),
                                qty0_e8=decimal_to_e8(d["B"]), px1_e8=decimal_to_e8(d["a"]),
                                qty1_e8=decimal_to_e8(d["A"]), seq=int(d["u"]), instrument=inst,
                                kind=EventKind.BOOK_TOP, flags=flags), out)


@dataclass
class StreamSpec:
    name: str
    url: str
    instrument_of: dict[str, int]  # stream-name suffix ("aggTrade", "bookTicker") -> instrument
    time_unit: str                 # "ms" or "us"


@dataclass
class RecorderConfig:
    symbol: str = "BTCUSDT"
    out: Path = Path("binance.plev")
    duration_s: float | None = None
    spot_ws: str = "wss://stream.binance.com:9443"
    futures_ws: str = "wss://fstream.binance.com"
    spot_rest: str = "https://api.binance.com"
    futures_rest: str = "https://fapi.binance.com"
    tick_size_e8: dict[int, int] | None = None  # None: read PRICE_FILTER from exchangeInfo
    silence_timeout_s: float = 60.0
    backoff_initial_s: float = 0.5
    backoff_max_s: float = 30.0
    flush_every_s: float = 1.0
    ping_interval_s: float = 20.0

    def streams(self) -> list[StreamSpec]:
        s = self.symbol.lower()
        return [
            StreamSpec("spot", f"{self.spot_ws}/stream?streams={s}@aggTrade/{s}@bookTicker&timeUnit=MICROSECOND",
                       {"aggTrade": SPOT, "bookTicker": SPOT}, "us"),
            StreamSpec("usdm-trades", f"{self.futures_ws}/market/stream?streams={s}@aggTrade", {"aggTrade": PERP}, "ms"),
            StreamSpec("usdm-book", f"{self.futures_ws}/public/stream?streams={s}@bookTicker", {"bookTicker": PERP},
                       "ms"),
        ]


def fetch_tick_sizes(cfg: RecorderConfig, opener: Callable[[str], bytes] | None = None) -> dict[int, int]:
    """PRICE_FILTER tick sizes from the exchangeInfo endpoints of both venues."""
    def get(url: str) -> dict:
        raw = opener(url) if opener else urllib.request.urlopen(url, timeout=10).read()
        return json.loads(raw)

    out = {}
    for inst, url in ((SPOT, f"{cfg.spot_rest}/api/v3/exchangeInfo?symbol={cfg.symbol}"),
                      (PERP, f"{cfg.futures_rest}/fapi/v1/exchangeInfo")):
        info = get(url)
        sym = next(s for s in info["symbols"] if s["symbol"] == cfg.symbol)
        flt = next(f for f in sym["filters"] if f["filterType"] == "PRICE_FILTER")
        out[inst] = decimal_to_e8(flt["tickSize"])
    return out


Connect = Callable[[str], Any]  # url -> async context manager yielding an object with `async recv()`


class Recorder:
    """Records the three streams into a .plev file until `duration_s` elapses or `stop()` is called."""

    def __init__(self, cfg: RecorderConfig, connect: Connect | None = None, clock: Clock | None = None,
                 sleep: Callable[[float], Awaitable[None]] | None = None, log: Callable[[str], None] | None = None):
        self.cfg = cfg
        self._connect = connect
        self.clock = clock or Clock()
        self._sleep = sleep or asyncio.sleep
        self._log = log or (lambda m: print(m, file=sys.stderr))
        self.norm = Normalizer()
        self.writer: PlevWriter | None = None
        self._stop: asyncio.Event | None = None  # created in run() (Python 3.9 binds events to a loop)
        self._stop_requested = False
        self.disconnects: Counter = Counter()
        self.silences: Counter = Counter()
        self.messages: Counter = Counter()
        self.last_ts_local = -1
        self.out_of_order = 0
        self._t_start = 0

    def stop(self) -> None:
        self._stop_requested = True
        if self._stop is not None:
            self._stop.set()

    def _write(self, recs: list[np.ndarray]) -> None:
        for r in recs:
            t = int(r["ts_local_ns"][0])
            if t < self.last_ts_local:  # cannot happen with one clock in one loop; checked anyway
                self.out_of_order += 1
            self.last_ts_local = max(self.last_ts_local, t)
            assert self.writer is not None
            self.writer.append(r)

    def _default_connect(self, url: str) -> Any:
        import websockets  # optional dependency: pip install "pulselob[record]"

        return websockets.connect(url, ping_interval=self.cfg.ping_interval_s, ping_timeout=self.cfg.ping_interval_s,
                                  max_size=2**20, open_timeout=10)

    async def _run_stream(self, spec: StreamSpec) -> None:
        assert self._stop is not None
        connect = self._connect or self._default_connect
        backoff = self.cfg.backoff_initial_s
        first = True
        status_inst = min(spec.instrument_of.values())
        while not self._stop.is_set():
            try:
                async with connect(spec.url) as ws:
                    if not first:
                        self._write(self.norm.status(StatusCode.RECONNECTED, status_inst, self.clock.now_ns()))
                    first = False
                    backoff = self.cfg.backoff_initial_s
                    while not self._stop.is_set():
                        recv = asyncio.ensure_future(ws.recv())
                        stop = asyncio.ensure_future(self._stop.wait())
                        done, _ = await asyncio.wait({recv, stop}, timeout=self.cfg.silence_timeout_s,
                                                     return_when=asyncio.FIRST_COMPLETED)
                        if recv not in done:
                            recv.cancel()
                            stop.cancel()
                            if self._stop.is_set():
                                return
                            self.silences[spec.name] += 1
                            self._log(f"[{spec.name}] silent for {self.cfg.silence_timeout_s:g} s, reconnecting")
                            raise ConnectionError("silent stream")
                        stop.cancel()
                        msg = recv.result()
                        arrival = self.clock.now_ns()
                        self.messages[spec.name] += 1
                        wrapper = json.loads(msg)
                        stream = wrapper.get("stream", "")
                        data = wrapper.get("data", wrapper)
                        suffix = stream.split("@", 1)[1] if "@" in stream else ("aggTrade" if data.get("e") ==
                                                                                 "aggTrade" else "bookTicker")
                        inst = spec.instrument_of.get(suffix)
                        if inst is None:
                            self.norm.stats.unparsed += 1
                            continue
                        self._write(self.norm.on_message(inst, data, arrival, spec.time_unit))
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # connection closed, network error, silent stream
                if self._stop.is_set():
                    return
                self.disconnects[spec.name] += 1
                self._write(self.norm.status(StatusCode.DISCONNECTED, status_inst, self.clock.now_ns()))
                self._log(f"[{spec.name}] disconnected ({type(exc).__name__}: {exc}); retry in {backoff:g} s")
                await self._sleep(backoff)
                backoff = min(backoff * 2.0, self.cfg.backoff_max_s)

    async def _flusher(self) -> None:
        assert self._stop is not None
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(self._stop.wait(), self.cfg.flush_every_s)
            except asyncio.TimeoutError:
                pass
            if self.writer is not None:
                self.writer.flush()

    async def run(self, tick_size_e8: dict[int, int] | None = None) -> dict:
        ticks = tick_size_e8 or self.cfg.tick_size_e8 or fetch_tick_sizes(self.cfg)
        self._stop = asyncio.Event()
        if self._stop_requested:
            self._stop.set()
        self.writer = PlevWriter(self.cfg.out)
        self._t_start = self.clock.now_ns()
        tasks = [asyncio.ensure_future(self._run_stream(s)) for s in self.cfg.streams()]
        tasks.append(asyncio.ensure_future(self._flusher()))
        try:
            if self.cfg.duration_s is not None:
                try:
                    await asyncio.wait_for(self._stop.wait(), self.cfg.duration_s)
                except asyncio.TimeoutError:
                    pass
            else:
                await self._stop.wait()
        finally:
            self._stop.set()
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self._write(self.norm.flush())
        summary = self.summary()
        meta = {
            "source": "binance websocket",
            "recorder_version": _core.__version__,
            "streams": [s.url for s in self.cfg.streams()],
            "instruments": [
                InstrumentSpec(PERP, "binance-usdm", self.cfg.symbol, ticks[PERP], "lead").__dict__,
                InstrumentSpec(SPOT, "binance-spot", self.cfg.symbol, ticks[SPOT], "quote").__dict__,
            ],
            "clock": "ts_local_ns = monotonic ns anchored to UTC at start (arrival time on receipt)",
            "summary": summary,
        }
        self.writer.close(meta)
        return summary

    def summary(self) -> dict:
        st = self.norm.stats
        now = self.clock.now_ns()
        return {
            "duration_s": (now - self._t_start) / 1e9,
            "records": self.writer.n_records if self.writer else 0,
            "trades": st.trades,
            "trade_prints": st.trade_prints,
            "sweeps_merged": st.merged,
            "books": st.books,
            "status_events": st.statuses,
            "gaps": st.gaps,
            "missing_trades": st.missing_trades,
            "duplicates_dropped": st.duplicates,
            "invalid_dropped": st.invalid,
            "unparsed_messages": st.unparsed,
            "messages": dict(self.messages),
            "disconnects": dict(self.disconnects),
            "silent_stream_restarts": dict(self.silences),
            "records_by_instrument": {str(k): v for k, v in sorted(st.by_instrument.items())},
            "out_of_order_records": self.out_of_order,
        }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pulselob-record",
                                 description="Record Binance Spot + USD-M perpetual trades and top of book to .plev.")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--out", required=True, help="output .plev path (a .plev.json sidecar is written next to it)")
    ap.add_argument("--duration", type=float, default=None, help="seconds to record (default: until Ctrl-C)")
    ap.add_argument("--silence-timeout", type=float, default=60.0)
    args = ap.parse_args(argv)
    cfg = RecorderConfig(symbol=args.symbol.upper(), out=Path(args.out), duration_s=args.duration,
                         silence_timeout_s=args.silence_timeout)

    async def go() -> dict:
        rec = Recorder(cfg)
        loop = asyncio.get_running_loop()
        for sig in (_signal.SIGINT, _signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, rec.stop)
            except (NotImplementedError, RuntimeError):
                pass
        return await rec.run()

    summary = asyncio.run(go())
    print(json.dumps(summary, indent=2))
    return 0 if math.isfinite(summary["duration_s"]) else 1
