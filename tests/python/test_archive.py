"""Archive-loader tests on synthetic zip files in the exact data.binance.vision formats (no network)."""

import datetime as dt
import hashlib
import io
import zipfile

import numpy as np
import pytest

import pulselob as pl
from pulselob.data import binance_archive as ba
from pulselob.schema import EventKind, SignalFlags

DAY = "2024-03-01"
MS0 = 1709251200000  # 2024-03-01T00:00:00Z in ms


def zip_bytes(name: str, text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(name, text)
    return buf.getvalue()


def make_archive() -> dict[str, bytes]:
    """url -> bytes for one day: perp aggTrades (header, ms), perp bookTicker (header, ms), spot aggTrades
    (no header, ms in 2024). A 400 ms silence in the middle exercises heartbeat insertion."""
    perp_agg = ["agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker"]
    perp_book = ["update_id,best_bid_price,best_bid_qty,best_ask_price,best_ask_qty,transaction_time,event_time"]
    spot_agg = []
    t = MS0
    for k in range(50):
        t += 3 if k != 25 else 400
        perp_book.append(f"{1000 + k},62000.0,1.5,62000.1,2.0,{t},{t + 1}")
        perp_agg.append(f"{10 + k},62000.1,0.010,{100 + 2 * k},{101 + 2 * k},{t + 1},{'false' if k % 3 else 'true'}")
        spot_agg.append(f"{500 + k},61999.99000000,0.00100000,{900 + k},{900 + k},{t + 2},{'True' if k % 2 else 'False'},True")
    files = {
        ba.archive_url("um", "aggTrades", "BTCUSDT", DAY): zip_bytes("a.csv", "\n".join(perp_agg) + "\n"),
        ba.archive_url("um", "bookTicker", "BTCUSDT", DAY): zip_bytes("b.csv", "\n".join(perp_book) + "\n"),
        ba.archive_url("spot", "aggTrades", "BTCUSDT", DAY): zip_bytes("c.csv", "\n".join(spot_agg) + "\n"),
    }
    for url in list(files):
        digest = hashlib.sha256(files[url]).hexdigest()
        files[url + ".CHECKSUM"] = f"{digest}  {url.rsplit('/', 1)[1]}\n".encode()
    return files


class FakeFetch:
    def __init__(self, files):
        self.files = files
        self.calls = []

    def __call__(self, url, dest):
        self.calls.append(url)
        dest.write_bytes(self.files[url])


def test_urls_follow_the_archive_layout():
    assert ba.archive_url("um", "bookTicker", "BTCUSDT", "2024-03-01") == (
        "https://data.binance.vision/data/futures/um/daily/bookTicker/BTCUSDT/BTCUSDT-bookTicker-2024-03-01.zip")
    assert ba.archive_url("spot", "aggTrades", "BTCUSDT", dt.date(2025, 6, 1)) == (
        "https://data.binance.vision/data/spot/daily/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2025-06-01.zip")
    with pytest.raises(ValueError, match="no spot bookTicker"):
        ba.archive_url("spot", "bookTicker", "BTCUSDT", DAY)


def test_download_verifies_checksum_and_caches(tmp_path):
    files = make_archive()
    fetch = FakeFetch(files)
    url = ba.archive_url("um", "aggTrades", "BTCUSDT", DAY)
    p = ba.download(url, tmp_path, fetch)
    assert p.read_bytes() == files[url]
    n = len(fetch.calls)
    ba.download(url, tmp_path, fetch)
    assert len(fetch.calls) == n  # cached and verified, not fetched again
    bad = dict(files)
    bad[url] = files[url] + b"tampered"
    with pytest.raises(IOError, match="checksum mismatch"):
        ba.download(url, tmp_path / "other", FakeFetch(bad))


def test_load_zip_formats(tmp_path):
    files = make_archive()
    fetch = FakeFetch(files)
    perp = ba.download(ba.archive_url("um", "aggTrades", "BTCUSDT", DAY), tmp_path, fetch)
    ev, stats = ba.load_zip(perp, "aggTrades", 0, latency_ns=2_000_000)
    assert stats["header"] and stats["rows"] == 50 and stats["resolution"] == int(pl.TsResolution.MILLISECOND)
    assert (ev["kind"] == EventKind.TRADE).all() and (ev["count"] == 2).all()
    assert ev["px0_e8"][0] == 6_200_010_000_000 and ev["qty0_e8"][0] == 1_000_000
    assert (ev["ts_local_ns"] - ev["ts_exchange_ns"] == 2_000_000).all()
    spot = ba.download(ba.archive_url("spot", "aggTrades", "BTCUSDT", DAY), tmp_path, fetch)
    ev_s, st_s = ba.load_zip(spot, "aggTrades", 1)
    assert not st_s["header"] and (ev_s["instrument"] == 1).all()


def test_build_dataset_merges_venues_and_never_fabricates_a_spot_book(tmp_path):
    fetch = FakeFetch(make_archive())
    out = tmp_path / "btc.plev"
    meta = ba.build_dataset([DAY], out, tmp_path / "cache", fetch=fetch)
    ev, m = pl.open_plev(out)
    assert m["sidecar"]["spot_book"] == "unavailable" and meta["records"] == len(ev)
    assert (pl.validate_events(ev) == 0).all()
    order = np.lexsort((ev["seq"], ev["instrument"], ev["ts_local_ns"]))
    assert np.array_equal(order, np.arange(len(ev)))  # sorted by (ts_local, instrument, seq)
    books = ev[ev["kind"] == EventKind.BOOK_TOP]
    assert len(books) == 50 and (books["instrument"] == 0).all()  # perp only
    hb = ev[ev["kind"] == EventKind.HEARTBEAT]
    assert len(hb) == 3  # the 400 ms silence gets heartbeats at +100, +200, +300 ms
    assert {f["url"].rsplit("/", 1)[1] for f in meta["files"]} == {
        "BTCUSDT-aggTrades-2024-03-01.zip", "BTCUSDT-bookTicker-2024-03-01.zip"}
    # The engine sees no spot book: spot trades get the neutral imbalance and are counted, and quoting on
    # spot is impossible, so the signals say so; quoting the perpetual works.
    cfg = pl.EngineConfig.default()
    eng = pl.Engine(cfg)
    sig = eng.run(out).signals
    n_spot = int(((ev["kind"] == EventKind.TRADE) & (ev["instrument"] == 1)).sum())
    assert eng.counters()["missing_book_marks"] >= n_spot  # every spot trade (plus perp trades before the first book)
    assert (sig["flags"] & SignalFlags.NO_QUOTE_BOOK).all() and (sig["bid_px_e8"] == 0).all()
    perp_quotes = pl.Engine(cfg.replace(quote_instrument=0, buy_channel=0, sell_channel=1)).run(out).signals
    assert not (perp_quotes["flags"] & SignalFlags.NO_QUOTE_BOOK).all()


def test_window_is_enforced(tmp_path):
    with pytest.raises(ValueError, match="2023-05-16"):
        ba.build_dataset(["2025-06-01"], tmp_path / "x.plev", tmp_path, fetch=FakeFetch({}))


def test_heartbeats_and_merge_are_deterministic():
    a = np.zeros(3, pl.MARKET_EVENT_DTYPE)
    a["ts_local_ns"] = [0, 250_000_000, 250_000_000]
    a["instrument"] = [0, 1, 0]
    a["seq"] = [1, 1, 2]
    m = ba.merge_streams(a[[2, 1]], a[[0]])
    assert list(m["instrument"]) == [0, 0, 1]
    hb = ba.insert_heartbeats(m, 100_000_000)
    assert list(hb["ts_local_ns"]) == [0, 100_000_000, 200_000_000, 250_000_000, 250_000_000]
