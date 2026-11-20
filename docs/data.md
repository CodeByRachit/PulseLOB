# PulseLOB: Data Sources

| Field | Value |
|---|---|
| Phase | 4 (bindings and data infrastructure) |
| Formats verified | 2026-11-20, against live Binance streams (a short local probe) and downloaded archive files |
| Decision | Phase 4 data plan (a) + (c): a live recorder for current, gap-free cross-venue data, and the 2023–2024 archive window for immediate history; synthetic data with ground truth for controlled experiments |

All three sources produce the same thing: a `.plev` file of 64-byte `MarketEvent` records ([architecture.md](architecture.md) Section 7) sorted by the engine clock `ts_local_ns`, with a JSON sidecar `<file>.plev.json` holding the instrument table and provenance. Every source uses the same market layout:

| Instrument | Venue | Role | Channels |
|---|---|---|---|
| 0 | Binance USD-M perpetual (venue A) | leading signal | 0 aggressive buys, 1 aggressive sells |
| 1 | Binance Spot (venue B) | quoting venue | 2 aggressive buys, 3 aggressive sells |

The engine clock is always the local **arrival** time, never the exchange time, so a replay or backtest never acts on an event before it could have arrived (architecture Section 7.4).

---

## 1. Synthetic data with ground truth (`pulselob.synth`)

The generator simulates the model of [math_spec.md](math_spec.md) Section 3 exactly. It uses the default kernel bank (Erlang orders 1, 2, 2, 1; rates 1000, 200, 50 and 1 per second) with piecewise-constant parameters, by Ogata thinning in C++ (`include/pulselob/synth/`). Generation runs at several million events per second.

```bash
pulselob-synth lead_lag --out data/lead_lag.plev --seed 1            # or: python -m pulselob.synth ...
pulselob-synth regime_shift --out data/shift.plev --constant-marks
```

```python
from pulselob import synth, Engine
ds = synth.generate("flash_crash", seed=1)       # Dataset: events, t_s, channel, lambda_true, ...
synth.write(ds, "crash.plev")                    # crash.plev, crash.plev.json, crash.plev.truth.npz
signals = Engine().run(ds.events).signals
```

| Scenario | Content | Default length | Jump times recorded |
|---|---|---|---|
| `calm` | stationary flow, weak perp-to-spot lead-lag | 600 s | none |
| `two_sided` | high, symmetric volume on both sides of both venues | 300 s | none |
| `lead_lag` | exogenous perp bursts (100 ms at 1,500/s, alternating buy and sell, every 5 s) that propagate to spot through the 5 ms and 20 ms Erlang-2 humps | 300 s | burst starts |
| `regime_shift` | perp-to-spot coupling steps from 0.10 to 0.50 at mid-horizon (the V10 experiment) | 600 s | the shift |
| `flash_crash` | 2 s sell cascade on both venues (baselines 3,000/s and 1,500/s, self-excitation 0.8) | 120 s | start and end |

Each dataset has a ground-truth sidecar, `<file>.truth.npz`, holding:

- `lambda_true`: the true left-limit intensities $\lambda(t_k-)$ of all four channels at every channel event;
- `t_s` and `channel`: the channel events themselves;
- `trade_index`: the position of each channel event's `Trade` record in the event array;
- `segment_starts` and `thetas`: the parameter segments;
- `jumps`: the jump times.

**Marks.** By default, sizes are log-normal and about 15% of trades sweep 1–3 levels, so marks vary realistically. The simulated process itself is unmarked: marks are independent with mean about 1, which is the normalization of math_spec Section 2.3. With `constant_marks=True`, sizes are 1, there are no sweeps and books are balanced. The engine's mark is then exactly 1, and the data follow exactly the model being estimated. Use this setting for estimator-recovery experiments.

**Verification** (`tests/python/test_synth.py`):

- The recorded intensities equal an independent Python recursion to 1e-9.
- Time-rescaled residuals pass mean and Kolmogorov–Smirnov tests against Exp(1), across regime and burst boundaries.
- Stationary counts match $(\mathbf I-\mathbf G)^{-1}\boldsymbol\mu\,T$.
- Each scenario has the intended effect through the engine. Two-sided noise scores $\tau\approx0$ (99th percentile below 0.05). Lead-lag bursts score above 0.5 on more than 20% of trades. The crash pulls the bid on more than 90% of its trades. The regime shift is tracked from 0.10 to 0.50.

---

## 2. Live recorder, option (a) (`pulselob.data.recorder`)

```bash
pip install "pulselob[record]"                    # adds the websockets client
pulselob-record --symbol BTCUSDT --out data/btcusdt-20261120.plev --duration 3600
```

The recorder stops at `--duration` or on Ctrl-C/SIGTERM. In both cases it flushes everything and prints a summary.

### Endpoints (verified 2026-11-20)

| Connection | URL | Streams |
|---|---|---|
| spot | `wss://stream.binance.com:9443/stream?streams=<s>@aggTrade/<s>@bookTicker&timeUnit=MICROSECOND` | trades and top of book |
| USD-M trades | `wss://fstream.binance.com/market/stream?streams=<s>@aggTrade` | trades |
| USD-M book | `wss://fstream.binance.com/public/stream?streams=<s>@bookTicker` | top of book |

USD-M market streams are now split by path. The probe found that `aggTrade` data arrive **only** on the `/market` path. The legacy `/stream` path and the `/public` path accept the subscription but send nothing: no trades arrived in 20 s on BTCUSDT. `bookTicker` arrives on `/public`, and still on the legacy path. A recorder pointed at the legacy URL would run silently forever. That is why the recorder treats a stream silent for longer than `silence_timeout_s` (default 60 s) as a lost connection, reconnects, and reports it in the summary. All URLs are configurable in `RecorderConfig`.

### Payloads (observed)

| Stream | Fields used | Timestamps |
|---|---|---|
| spot `aggTrade` | `a` aggregate id, `p`, `q`, `f`, `l` first and last trade ids, `T` trade time, `m` buyer is maker | µs with `timeUnit=MICROSECOND` |
| spot `bookTicker` | `u`, `b`, `B`, `a`, `A` | **none**: the spot book ticker has no exchange time |
| USD-M `aggTrade` | as spot, plus `nq` and `st` (ignored) | ms |
| USD-M `bookTicker` | `u`, `b`, `B`, `a`, `A`, `T` transaction time, `E` event time, `ps`, `st` | ms |

Combined streams wrap each payload as `{"stream": "<s>@aggTrade", "data": {...}}`. Tick sizes are read once from `GET /api/v3/exchangeInfo?symbol=<s>` (spot) and `GET /fapi/v1/exchangeInfo` (USD-M), `PRICE_FILTER.tickSize`. For BTCUSDT these are 0.01 on spot and 0.10 on USD-M.

### Normalization

- **Arrival time.** On receipt each message is stamped with monotonic nanoseconds anchored to the UTC wall clock at start-up. One clock and one event loop make the output non-decreasing in `ts_local_ns`, and the summary reports any violation.
- **Trades.** `m = true` means the buyer was the maker, so the aggressor sold. Consecutive prints with equal instrument, side and exchange time are merged into one sweep event. Its first price is `px1`, its worst price `px0`, and quantities and trade counts are summed. A sweep is complete only when its last part has arrived, so it carries that part's arrival time.
- **Books.** USD-M books carry the transaction time `T` as `ts_exchange_ns`. Spot books have no exchange time, so `ts_exchange_ns` is the arrival time with the `TS_ADJUSTED` flag set. Crossed or locked books fail validation and are dropped and counted.
- **Gaps and duplicates.** Aggregate-trade ids are contiguous per symbol.
  - **A gap** (`a > last + 1`) produces a `Status(FEED_GAP)` event, and the missing trades are counted.
  - **A repeated id** after a reconnect is dropped.
- **Connections.** Server pings are answered automatically, and the client pings every 20 s. A lost connection writes `Status(DISCONNECTED)` and reconnects with exponential back-off (0.5 s doubling to 30 s), then writes `Status(RECONNECTED)`. Binance closes connections after 24 hours, and this is the path that handles it. In the engine, any status event trips the kill switch (math_spec Section 8.6).
- **Output.** A `.plev` file is written through `PlevWriter`, which rewrites the header count on every flush (default every second), so the file is valid after each flush. `recover_plev` truncates a torn tail after a crash. The sidecar records the stream URLs, the instrument table and the summary.

**Testing.** No test or CI job connects to Binance. `tests/python/test_recorder.py` covers:

- payloads copied from the live probe;
- a scripted transport with a disconnect, a gap, a duplicate and a silent stream;
- an end-to-end run through a real WebSocket client against a local server on `127.0.0.1`.

---

## 3. Historical archive, option (c) (`pulselob.data.binance_archive`)

```bash
pulselob-archive --symbol BTCUSDT --day 2024-03-01 --day 2024-03-02 \
    --out data/btcusdt-2024-03-01.plev --cache data/binance-cache
```

### Files

Daily files from `https://data.binance.vision/data/`:

| Dataset | URL pattern | Available |
|---|---|---|
| USD-M `aggTrades` | `futures/um/daily/aggTrades/<S>/<S>-aggTrades-<YYYY-MM-DD>.zip` | yes |
| USD-M `bookTicker` | `futures/um/daily/bookTicker/<S>/<S>-bookTicker-<YYYY-MM-DD>.zip` | **2023-05-16 to 2024-03-30 only** (BTCUSDT) |
| Spot `aggTrades` | `spot/daily/aggTrades/<S>/<S>-aggTrades-<YYYY-MM-DD>.zip` | yes |
| Spot `bookTicker` | none | **never published** |

Every zip has a `<file>.zip.CHECKSUM` holding its SHA-256. The loader downloads each file once into the cache. The cache mirrors the archive's directory layout, because spot and futures files share file names. Every file is verified before use.

### Formats (verified against downloaded files)

| File | Header row | Columns | Booleans | Timestamps |
|---|---|---|---|---|
| USD-M aggTrades | yes | `agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker` | `true`/`false` | ms |
| USD-M bookTicker | yes | `update_id,best_bid_price,best_bid_qty,best_ask_price,best_ask_qty,transaction_time,event_time` | – | ms |
| Spot aggTrades | no | the seven above plus `is_best_match` | `True`/`False` | ms in 2024 files, **µs** from 2025 |

Parsing is exact and done in C++ (`include/pulselob/feed/binance_csv.hpp`):

- **Prices and quantities** go from decimal strings to e8 integers with no floating point. A ninth nonzero decimal is an error, not a rounding.
- **The timestamp unit** is detected per file from magnitude: below 1e14 is ms, below 1e17 is µs.
- **Sweeps** are merged as in the recorder.
- **Rows that fail validation** are counted and dropped.
- **A malformed row** stops the load with its line number.

### Clocks, merge and heartbeats

- **Clock.** `ts_local_ns` is the exchange event time plus a deterministic per-venue latency (default 2 ms each; `--perp-latency-ms` and `--spot-latency-ms`). For book tickers the event time is `event_time`, and `ts_exchange_ns` is `transaction_time`.
- **Merge order.** The venues are merged by `(ts_local_ns, instrument, seq)`, which is deterministic.
- **Heartbeats.** These are inserted on the 100 ms grid after any longer silence, as a live feed would deliver them.

### The missing spot book

Binance publishes no spot top of book, so the loader creates **no spot `BookTop` events**, and the sidecar says `"spot_book": "unavailable"`. Nothing is fabricated. The engine then handles the gap explicitly:

- spot trades get the neutral imbalance $\iota=0$ in their marks, and are counted in `missing_book_marks`;
- quoting on spot is impossible, so signals carry the `NO_QUOTE_BOOK` flag and no prices.

Configure `quote_instrument=0` (with channels 0 and 1) to quote the perpetual on these data, as Phase 5 does for option (c). `tests/python/test_archive.py` checks all of this end to end.

### Known gaps and caveats

1. **No spot book at any date**, as above. The recorder (option a) is the only source of spot top of book.
2. **USD-M bookTicker only from 2023-05-16 to 2024-03-30.** The loader refuses other days unless `--allow-outside-window` is given; outside the window it would produce trades without a perpetual book.
3. **Millisecond timestamps.** Ties across venues are common, and the engine treats them as tie batches (math_spec Section 4.4). The 1 ms fast kernel is at the resolution limit of these data.
4. **Exchange clocks.** The two venues' exchange clocks are not synchronized, and the archive has no arrival times. The latency model is an assumption, recorded in the sidecar. Skew between venues biases the lead-lag weights (math_spec Section 11.1, assumption 6).
5. **Tick sizes may have changed over time.** The defaults are the current BTCUSDT values. Pass the historical ones for other periods.
6. **Daily files are UTC days.** A multi-day dataset is several files merged; there is no gap handling at day boundaries beyond heartbeats.
