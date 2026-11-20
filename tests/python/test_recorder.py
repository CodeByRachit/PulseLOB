"""Recorder tests. No test touches a Binance endpoint: payloads are the formats observed on the live
streams (2026-11-20), served by a scripted fake transport or by a local WebSocket server."""

import asyncio
import json
from pathlib import Path

import numpy as np
import pytest

import pulselob as pl
from pulselob.data.recorder import PERP, SPOT, Normalizer, Recorder, RecorderConfig, fetch_tick_sizes
from pulselob.schema import EventKind, Flags, Side, StatusCode, TsResolution

T0 = 1_795_194_843_644_000_000

# Payloads as received (stream wrapper stripped), from the 2026-11-20 probe.
SPOT_AGG = {"e": "aggTrade", "E": 1795194843644874, "s": "LTCUSDT", "a": 338434870, "p": "39.74000000",
            "q": "0.30400000", "f": 608596110, "l": 608596110, "T": 1795194843644594, "m": True, "M": True}
SPOT_BOOK = {"u": 13681680648, "s": "LTCUSDT", "b": "39.74000000", "B": "237.11400000", "a": "39.75000000",
             "A": "0.51100000"}
FUT_AGG = {"e": "aggTrade", "E": 1795194869284, "a": 3536241127, "s": "BTCUSDT", "p": "59250.00", "q": "0.030",
           "nq": "0.030", "f": 8306361968, "l": 8306361968, "T": 1795194869145, "m": True, "st": 1}
FUT_BOOK = {"e": "bookTicker", "u": 12182429932602, "s": "LTCUSDT", "ps": "LTCUSDT", "b": "39.71", "B": "109.848",
            "a": "39.72", "A": "460.834", "T": 1795194843172, "E": 1795194843173, "st": 1}


def rec(lst):
    return np.concatenate(lst) if lst else np.empty(0, pl.MARKET_EVENT_DTYPE)


def test_spot_and_futures_trades_and_books():
    n = Normalizer()
    out = rec(n.on_message(SPOT, SPOT_AGG, T0, "us") + n.on_message(SPOT, SPOT_BOOK, T0 + 10) +
              n.on_message(PERP, FUT_AGG, T0 + 20) + n.on_message(PERP, FUT_BOOK, T0 + 30) + n.flush())
    assert list(out["kind"]) == [EventKind.TRADE, EventKind.BOOK_TOP, EventKind.TRADE, EventKind.BOOK_TOP]
    spot_trade, spot_book, fut_trade, fut_book = out
    assert spot_trade["ts_exchange_ns"] == 1795194843644594 * 1000  # microseconds (timeUnit=MICROSECOND)
    assert spot_trade["flags"] & Flags.SIDE_MASK == Side.SELL       # m = true: the aggressor sold
    assert (spot_trade["flags"] >> 2) & 3 == TsResolution.MICROSECOND
    assert spot_trade["px0_e8"] == 3_974_000_000 and spot_trade["qty0_e8"] == 30_400_000
    assert spot_trade["ts_local_ns"] == T0
    assert spot_book["ts_exchange_ns"] == spot_book["ts_local_ns"] == T0 + 10  # no exchange time on spot
    assert spot_book["flags"] & Flags.TS_ADJUSTED
    assert fut_trade["ts_exchange_ns"] == 1795194869145 * 1_000_000
    assert fut_trade["px0_e8"] == 5_925_000_000_000
    assert fut_book["ts_exchange_ns"] == 1795194843172 * 1_000_000  # transaction time
    assert fut_book["bid_px_e8" if "bid_px_e8" in out.dtype.names else "px0_e8"] == 3_971_000_000
    assert (pl.validate_events(out) == 0).all()
    # A payload whose timestamp overflows 64-bit nanoseconds is counted, not fatal.
    assert n.on_message(SPOT, dict(SPOT_AGG, a=SPOT_AGG["a"] + 1), T0 + 40, "ms") == []
    assert n.stats.unparsed == 1


def test_sweeps_merge_and_take_the_last_arrival():
    n = Normalizer()
    base = dict(FUT_AGG, m=False)
    parts = [dict(base, a=10, p="100.00", q="1.000", f=1, l=1), dict(base, a=11, p="100.10", q="2.000", f=2, l=3),
             dict(base, a=12, p="100.05", q="0.500", f=4, l=4)]
    out = []
    for k, p in enumerate(parts):
        out += n.on_message(PERP, p, T0 + k)
    out += n.on_message(PERP, dict(base, a=13, T=base["T"] + 1, f=5, l=5), T0 + 3)
    out += n.flush()
    ev = rec(out)
    assert len(ev) == 2 and n.stats.merged == 2
    sweep = ev[0]
    assert sweep["px1_e8"] == 10_000_000_000 and sweep["px0_e8"] == 10_010_000_000  # first and worst prices
    assert sweep["qty0_e8"] == 350_000_000 and sweep["count"] == 4
    assert sweep["ts_local_ns"] == T0 + 2  # complete only when its last part arrived


def test_gaps_duplicates_and_invalid_books():
    n = Normalizer()
    out = n.on_message(PERP, dict(FUT_AGG, a=100), T0)
    out += n.on_message(PERP, dict(FUT_AGG, a=105, T=FUT_AGG["T"] + 1), T0 + 1)  # 4 trades missing
    out += n.on_message(PERP, dict(FUT_AGG, a=103, T=FUT_AGG["T"] + 2), T0 + 2)  # replay after reconnect
    out += n.on_message(PERP, dict(FUT_BOOK, b="39.72", a="39.72"), T0 + 3)       # locked book
    out += n.on_message(PERP, {"unexpected": 1}, T0 + 4)
    ev = rec(out + n.flush())
    status = ev[ev["kind"] == EventKind.STATUS]
    assert len(status) == 1 and status[0]["count"] == StatusCode.FEED_GAP and status[0]["instrument"] == PERP
    assert n.stats.gaps == 1 and n.stats.missing_trades == 4
    assert n.stats.duplicates == 1 and n.stats.invalid == 1 and n.stats.unparsed == 1
    assert np.all(np.diff(ev["ts_local_ns"]) >= 0)


class FakeClock:
    def __init__(self):
        self.t = T0

    def now_ns(self):
        self.t += 1000
        return self.t


class FakeWS:
    """Scripted connection: messages, then an exception (disconnect), "silence", or "stop"."""

    def __init__(self, script, recorder_ref):
        self.script = list(script)
        self.recorder_ref = recorder_ref

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def recv(self):
        await asyncio.sleep(0)
        if not self.script:
            await asyncio.sleep(3600)
        item = self.script.pop(0)
        if item == "silence":
            await asyncio.sleep(3600)
        if item == "stop":
            self.recorder_ref[0].stop()
            await asyncio.sleep(3600)
        if isinstance(item, Exception):
            raise item
        return item


def wrap(stream, data):
    return json.dumps({"stream": stream, "data": data})


def test_recorder_end_to_end_with_disconnects_and_silence(tmp_path):
    cfg = RecorderConfig(symbol="BTCUSDT", out=tmp_path / "rec.plev", silence_timeout_s=0.05,
                         backoff_initial_s=0.001, flush_every_s=0.01)
    ref = [None]
    scripts = {
        "spot": [[wrap("btcusdt@aggTrade", SPOT_AGG), wrap("btcusdt@bookTicker", SPOT_BOOK),
                  ConnectionResetError("peer closed")],
                 [wrap("btcusdt@aggTrade", dict(SPOT_AGG, a=SPOT_AGG["a"] + 3))]],
        # The futures-trade stream is silent first (restarted by the watchdog), then stops the run.
        "usdm-trades": [["silence"], [wrap("btcusdt@aggTrade", FUT_AGG), "stop"]],
        "usdm-book": [[wrap("btcusdt@bookTicker", FUT_BOOK)]],
    }
    counters = {k: 0 for k in scripts}

    def connect(url):
        # Futures paths first: "stream.binance.com" is also a substring of "fstream.binance.com".
        key = "usdm-trades" if "/market/" in url else ("usdm-book" if "/public/" in url else "spot")
        k = counters[key]
        counters[key] += 1
        script = scripts[key][k] if k < len(scripts[key]) else []
        return FakeWS(script, ref)

    async def no_sleep(_s):
        await asyncio.sleep(0)

    r = Recorder(cfg, connect=connect, clock=FakeClock(), sleep=no_sleep, log=lambda m: None)
    ref[0] = r
    summary = asyncio.run(r.run(tick_size_e8={PERP: 10_000_000, SPOT: 1_000_000}))
    ev, meta = pl.open_plev(cfg.out)
    assert summary["records"] == len(ev) == meta["sidecar"]["summary"]["records"]
    assert np.all(np.diff(ev["ts_local_ns"]) >= 0) and summary["out_of_order_records"] == 0
    assert (pl.validate_events(ev) == 0).all()
    codes = list(ev["count"][ev["kind"] == EventKind.STATUS])
    assert StatusCode.DISCONNECTED in codes and StatusCode.RECONNECTED in codes and StatusCode.FEED_GAP in codes
    assert summary["disconnects"]["spot"] >= 1
    assert summary["silent_stream_restarts"].get("usdm-trades", 0) >= 1
    assert summary["gaps"] == 1 and summary["missing_trades"] == 2
    assert {i["venue"] for i in meta["sidecar"]["instruments"]} == {"binance-usdm", "binance-spot"}
    assert any("/market/stream" in u for u in meta["sidecar"]["streams"])


def test_recorded_file_runs_through_the_engine(tmp_path):
    n = Normalizer()
    out = []
    t = T0
    for k in range(200):
        t += 1_000_000
        out += n.on_message(PERP, dict(FUT_BOOK, u=k, T=FUT_BOOK["T"] + k), t)
        out += n.on_message(SPOT, dict(SPOT_BOOK, u=k), t + 10)
        out += n.on_message(PERP, dict(FUT_AGG, a=1000 + k, T=FUT_AGG["T"] + k, m=k % 2 == 0, p="39.72"), t + 20)
        out += n.on_message(SPOT, dict(SPOT_AGG, a=5000 + k, T=SPOT_AGG["T"] + 1000 * k), t + 30, "us")
    ev = rec(out + n.flush())
    cfg = pl.EngineConfig.default(instruments=((PERP, 0, 1_000_000), (SPOT, 1, 1_000_000)))
    res = pl.Engine(cfg).run(ev)
    assert (res.signals["status"] & 0xFF == 0).all()
    assert np.isfinite(res.signals["tau_buy"]).all()


def test_tick_sizes_from_exchange_info():
    spot = {"symbols": [{"symbol": "BTCUSDT", "filters": [{"filterType": "PRICE_FILTER", "tickSize": "0.01000000"}]}]}
    fut = {"symbols": [{"symbol": "ETHUSDT", "filters": []},
                       {"symbol": "BTCUSDT", "filters": [{"filterType": "PRICE_FILTER", "tickSize": "0.10"}]}]}
    ticks = fetch_tick_sizes(RecorderConfig(), opener=lambda url: json.dumps(spot if "api/v3" in url else fut).encode())
    assert ticks == {SPOT: 1_000_000, PERP: 10_000_000}


def test_local_websocket_server_integration(tmp_path):
    """A real WebSocket client against a local server (no Binance): connection, framing, shutdown."""
    websockets = pytest.importorskip("websockets")

    async def scenario():
        async def handler(ws, path=None):
            p = path if path is not None else ws.request.path
            if p.startswith("/stream"):
                await ws.send(wrap("btcusdt@aggTrade", SPOT_AGG))
                await ws.send(wrap("btcusdt@bookTicker", SPOT_BOOK))
            elif p.startswith("/market/"):
                await ws.send(wrap("btcusdt@aggTrade", FUT_AGG))
            else:
                await ws.send(wrap("btcusdt@bookTicker", FUT_BOOK))
            await ws.wait_closed()

        async with websockets.serve(handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            base = f"ws://127.0.0.1:{port}"
            cfg = RecorderConfig(out=tmp_path / "local.plev", spot_ws=base, futures_ws=base, duration_s=1.0,
                                 flush_every_s=0.05)
            return await Recorder(cfg, log=lambda m: None).run(tick_size_e8={PERP: 10_000_000, SPOT: 1_000_000})

    summary = asyncio.run(scenario())
    ev, _ = pl.open_plev(tmp_path / "local.plev")
    assert summary["trades"] == 2 and summary["books"] == 2 and len(ev) == 4
    assert summary["disconnects"] == {}
