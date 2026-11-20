import numpy as np
import pytest

import pulselob
from pulselob import MARKET_EVENT_DTYPE, EventKind, Side, TsResolution, make_flags
from pulselob.plev import HEADER_SIZE, PlevFormatError, open_plev, read_header, write_plev


def _events(n: int) -> np.ndarray:
    ev = np.zeros(n, dtype=MARKET_EVENT_DTYPE)
    ev["ts_exchange_ns"] = np.arange(n, dtype=np.int64) * 1_000_000
    ev["ts_local_ns"] = ev["ts_exchange_ns"] + 5_000_000
    ev["px0_e8"] = 100
    ev["px1_e8"] = 101
    ev["qty0_e8"] = 7
    ev["qty1_e8"] = 9
    ev["kind"] = EventKind.BOOK_TOP
    ev["flags"] = make_flags(Side.NONE, TsResolution.MILLISECOND)
    ev["seq"] = np.arange(n, dtype=np.uint64)
    return ev


def test_roundtrip_memmap_and_sidecar(tmp_path):
    ev = _events(1234)
    meta = {"instruments": [{"id": 0, "venue": "binance-spot", "symbol": "BTCUSDT"}]}
    path = write_plev(tmp_path / "day.plev", ev, meta)
    assert path.stat().st_size == HEADER_SIZE + 1234 * 64
    loaded, info = open_plev(path)
    assert isinstance(loaded, np.memmap)
    assert loaded.dtype == MARKET_EVENT_DTYPE
    np.testing.assert_array_equal(loaded, ev)
    assert info["n_records"] == 1234
    assert info["sidecar"] == meta
    # The mapped records go straight to the native code without a copy.
    assert np.count_nonzero(pulselob.validate_events(loaded)) == 0


def test_rejects_corrupt_files(tmp_path):
    path = write_plev(tmp_path / "x.plev", _events(4))
    raw = bytearray(path.read_bytes())
    bad_magic = tmp_path / "bad_magic.plev"
    bad_magic.write_bytes(b"NOPE" + raw[4:])
    with pytest.raises(PlevFormatError, match="magic"):
        read_header(bad_magic)
    truncated = tmp_path / "truncated.plev"
    truncated.write_bytes(raw[:-1])
    with pytest.raises(PlevFormatError, match="size"):
        read_header(truncated)


def test_write_requires_schema_dtype(tmp_path):
    with pytest.raises(TypeError):
        write_plev(tmp_path / "y.plev", np.zeros(3, dtype=np.int64))


def test_streaming_writer_is_valid_after_each_flush_and_recoverable(tmp_path):
    from pulselob import PlevWriter, open_plev, read_header, recover_plev, MARKET_EVENT_DTYPE

    path = tmp_path / "stream.plev"
    ev = np.zeros(10, dtype=MARKET_EVENT_DTYPE)
    ev["ts_local_ns"] = np.arange(10)
    w = PlevWriter(path, flush_records=4)
    w.append(ev[:3])
    w.flush()
    assert read_header(path)["n_records"] == 3  # valid mid-stream
    w.append(ev[3:])
    w.close({"note": "test"})
    arr, meta = open_plev(path)
    assert np.array_equal(arr["ts_local_ns"], np.arange(10)) and meta["sidecar"]["note"] == "test"
    with open(path, "ab") as f:  # a crash mid-record leaves a torn tail
        f.write(b"\x01" * 20)
    assert recover_plev(path) == 10
    assert read_header(path)["n_records"] == 10


def test_empty_file_opens(tmp_path):
    from pulselob import MARKET_EVENT_DTYPE, open_plev, write_plev

    path = write_plev(tmp_path / "empty.plev", np.empty(0, dtype=MARKET_EVENT_DTYPE))
    arr, _ = open_plev(path)
    assert arr.shape == (0,)
