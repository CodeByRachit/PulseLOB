#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

#include "pulselob/core/fixed_point.hpp"
#include "pulselob/feed/market_event.hpp"

namespace pulselob::feed::binance {

// Parser for the data.binance.vision CSV archives (docs/data.md). Formats, verified against
// downloaded files:
//
//   aggTrades, spot (no header; booleans True/False; ms until 2024, us from 2025):
//     agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker,is_best_match
//   aggTrades, USD-M futures (header; booleans true/false; ms):
//     agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker
//   bookTicker, USD-M futures (header; ms):
//     update_id,best_bid_price,best_bid_qty,best_ask_price,best_ask_qty,transaction_time,event_time
//
// Prices and quantities go through core::parse_decimal_e8 (exact, no floating point). The
// timestamp unit is detected per file from the first row's magnitude. The engine clock
// ts_local_ns is the exchange event time (bookTicker event_time, aggTrade transact_time) plus a
// deterministic latency offset.
enum class CsvKind : std::uint8_t { AggTrades = 0, BookTicker = 1 };

struct CsvOptions {
  CsvKind kind = CsvKind::AggTrades;
  std::uint16_t instrument = 0;
  std::int64_t latency_ns = 0;  // added to the exchange event time to form ts_local_ns
  bool merge_sweeps = true;     // merge consecutive aggTrades with equal side and timestamp
};

struct CsvResult {
  bool ok = true;
  const char* error = nullptr;  // set when ok is false
  std::size_t error_line = 0;   // 1-based
  std::size_t rows = 0;         // data rows read
  std::size_t events = 0;       // events appended
  std::size_t merged = 0;       // rows folded into a preceding sweep
  std::size_t invalid = 0;      // rows dropped by feed::validate (for example a crossed book)
  bool header = false;
  TsResolution resolution = TsResolution::Millisecond;
};

namespace detail {

inline bool parse_int(std::string_view s, std::int64_t& out) noexcept {
  if (s.empty()) {
    return false;
  }
  std::int64_t v = 0;
  for (char c : s) {
    if (c < '0' || c > '9' || v > (INT64_MAX - 9) / 10) {
      return false;
    }
    v = v * 10 + (c - '0');
  }
  out = v;
  return true;
}

inline bool parse_bool(std::string_view s, bool& out) noexcept {
  if (s == "true" || s == "True" || s == "TRUE") {
    out = true;
    return true;
  }
  if (s == "false" || s == "False" || s == "FALSE") {
    out = false;
    return true;
  }
  return false;
}

// Splits one line into at most N fields; returns the number of fields.
template <std::size_t N>
inline std::size_t split(std::string_view line, std::string_view (&f)[N]) noexcept {
  std::size_t n = 0;
  std::size_t start = 0;
  for (std::size_t i = 0; i <= line.size(); ++i) {
    if (i == line.size() || line[i] == ',') {
      if (n == N) {
        return N + 1;
      }
      f[n++] = line.substr(start, i - start);
      start = i + 1;
    }
  }
  return n;
}

inline TsResolution detect_unit(std::int64_t t) noexcept {
  if (t < 100'000'000'000'000) {  // < 1e14
    return TsResolution::Millisecond;
  }
  if (t < 100'000'000'000'000'000) {  // < 1e17
    return TsResolution::Microsecond;
  }
  return TsResolution::Nanosecond;
}

inline std::int64_t to_ns(std::int64_t t, TsResolution r) noexcept {
  switch (r) {
    case TsResolution::Millisecond:
      return t * 1'000'000;
    case TsResolution::Microsecond:
      return t * 1'000;
    case TsResolution::Nanosecond:
      return t;
    case TsResolution::Second:
      return t * 1'000'000'000;
  }
  return t;
}

}  // namespace detail

// Parses a whole CSV file (text) and appends MarketEvents to out. Never throws; a malformed row
// stops the parse with ok = false and the line number.
inline CsvResult parse_csv(std::string_view text, const CsvOptions& opt, std::vector<MarketEvent>& out) {
  CsvResult res{};
  std::size_t line_no = 0;
  std::size_t pos = 0;
  bool unit_known = false;
  bool have_pending = false;
  MarketEvent pending{};
  auto fail = [&](const char* why) {
    res.ok = false;
    res.error = why;
    res.error_line = line_no;
    return res;
  };
  auto emit = [&](const MarketEvent& e) {
    if (validate(e) != EventCheck::Ok) {
      ++res.invalid;
      return;
    }
    out.push_back(e);
    ++res.events;
  };
  while (pos < text.size()) {
    std::size_t end = text.find('\n', pos);
    if (end == std::string_view::npos) {
      end = text.size();
    }
    std::string_view line = text.substr(pos, end - pos);
    pos = end + 1;
    ++line_no;
    if (!line.empty() && line.back() == '\r') {
      line.remove_suffix(1);
    }
    if (line.empty()) {
      continue;
    }
    if (line_no == 1 && (line[0] < '0' || line[0] > '9')) {
      res.header = true;  // column names
      continue;
    }
    std::string_view f[9];
    const std::size_t nf = detail::split(line, f);
    ++res.rows;
    if (opt.kind == CsvKind::AggTrades) {
      if (nf != 7 && nf != 8) {
        return fail("aggTrades row must have 7 or 8 fields");
      }
      std::int64_t id = 0;
      std::int64_t first = 0;
      std::int64_t last = 0;
      std::int64_t t = 0;
      bool buyer_maker = false;
      const auto px = core::parse_decimal_e8(f[1]);
      const auto qty = core::parse_decimal_e8(f[2]);
      if (!detail::parse_int(f[0], id) || !detail::parse_int(f[3], first) || !detail::parse_int(f[4], last) ||
          !detail::parse_int(f[5], t) || !detail::parse_bool(f[6], buyer_maker) || !px || !qty ||
          last < first) {
        return fail("malformed aggTrades row");
      }
      if (!unit_known) {
        res.resolution = detail::detect_unit(t);
        unit_known = true;
      }
      const std::int64_t ts = detail::to_ns(t, res.resolution);
      const Side side = buyer_maker ? Side::Sell : Side::Buy;  // buyer is maker: the aggressor sold
      const auto notional = static_cast<std::int64_t>(
          std::llround(static_cast<double>(*px) * (static_cast<double>(*qty) / 1e8)));
      const auto prints = static_cast<std::uint32_t>(last - first + 1);
      if (opt.merge_sweeps && have_pending && pending.ts_exchange_ns == ts && pending.side() == side) {
        // Same aggressor side and timestamp: one sweep. px1 keeps the first price, px0 the worst.
        pending.px0_e8 = side == Side::Buy ? (*px > pending.px0_e8 ? *px : pending.px0_e8)
                                           : (*px < pending.px0_e8 ? *px : pending.px0_e8);
        pending.qty0_e8 += *qty;
        pending.qty1_e8 += notional;
        pending.count += prints;
        ++res.merged;
        continue;
      }
      if (have_pending) {
        emit(pending);
      }
      pending = MarketEvent{};
      pending.ts_exchange_ns = ts;
      pending.ts_local_ns = ts + opt.latency_ns;
      pending.px0_e8 = *px;
      pending.px1_e8 = *px;
      pending.qty0_e8 = *qty;
      pending.qty1_e8 = notional;
      pending.seq = static_cast<std::uint64_t>(id);
      pending.count = prints;
      pending.instrument = opt.instrument;
      pending.kind = EventKind::Trade;
      pending.flags = event_flags::make(side, res.resolution);
      have_pending = true;
    } else {
      if (nf != 7) {
        return fail("bookTicker row must have 7 fields");
      }
      std::int64_t id = 0;
      std::int64_t t_tx = 0;
      std::int64_t t_ev = 0;
      const auto bp = core::parse_decimal_e8(f[1]);
      const auto bq = core::parse_decimal_e8(f[2]);
      const auto ap = core::parse_decimal_e8(f[3]);
      const auto aq = core::parse_decimal_e8(f[4]);
      if (!detail::parse_int(f[0], id) || !detail::parse_int(f[5], t_tx) || !detail::parse_int(f[6], t_ev) ||
          !bp || !bq || !ap || !aq) {
        return fail("malformed bookTicker row");
      }
      if (!unit_known) {
        res.resolution = detail::detect_unit(t_tx);
        unit_known = true;
      }
      MarketEvent b{};
      b.ts_exchange_ns = detail::to_ns(t_tx, res.resolution);
      b.ts_local_ns = detail::to_ns(t_ev, res.resolution) + opt.latency_ns;
      b.px0_e8 = *bp;
      b.qty0_e8 = *bq;
      b.px1_e8 = *ap;
      b.qty1_e8 = *aq;
      b.seq = static_cast<std::uint64_t>(id);
      b.instrument = opt.instrument;
      b.kind = EventKind::BookTop;
      b.flags = event_flags::make(Side::None, res.resolution);
      emit(b);
    }
  }
  if (have_pending) {
    emit(pending);
  }
  return res;
}

}  // namespace pulselob::feed::binance
