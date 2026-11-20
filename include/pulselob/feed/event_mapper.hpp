#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include "pulselob/core/fixed_point.hpp"
#include "pulselob/core/time.hpp"
#include "pulselob/feed/market_event.hpp"

namespace pulselob::feed {

inline constexpr std::size_t kMaxInstruments = 8;
inline constexpr std::size_t kMaxVenues = 4;

struct InstrumentSpec {
  bool present = false;
  std::uint8_t venue = 0;    // index < kMaxVenues
  std::int64_t tick_e8 = 1;  // price tick in e8 units
};

// Instrument table and the (instrument, aggressor side) -> channel map (docs/architecture.md 7.3, 7.5).
template <std::size_t D>
struct ChannelMap {
  std::array<InstrumentSpec, kMaxInstruments> instruments{};
  std::array<std::array<int, 2>, kMaxInstruments> channel = [] {
    std::array<std::array<int, 2>, kMaxInstruments> c{};
    for (auto& row : c) {
      row = {-1, -1};
    }
    return c;
  }();
  std::array<int, D> direction{};  // s_i, used to align the book imbalance with the trade

  void set_channel(std::size_t instrument, Side side, int ch) noexcept {
    channel[instrument][side == Side::Buy ? 0 : 1] = ch;
  }
  int channel_of(std::size_t instrument, Side side) const noexcept {
    if (instrument >= kMaxInstruments || (side != Side::Buy && side != Side::Sell)) {
      return -1;
    }
    return channel[instrument][side == Side::Buy ? 0 : 1];
  }
};

template <std::size_t D>
const char* validate(const ChannelMap<D>& m) noexcept {
  std::array<int, D> uses{};
  for (std::size_t k = 0; k < kMaxInstruments; ++k) {
    const InstrumentSpec& s = m.instruments[k];
    if (s.present && (s.venue >= kMaxVenues || s.tick_e8 <= 0)) {
      return "instrument needs a venue < kMaxVenues and a positive tick";
    }
    for (int ch : m.channel[k]) {
      if (ch < -1 || ch >= static_cast<int>(D)) {
        return "channel map entry out of range";
      }
      if (ch >= 0) {
        if (!s.present) {
          return "channel mapped from an instrument missing from the instrument table";
        }
        ++uses[static_cast<std::size_t>(ch)];
      }
    }
  }
  for (std::size_t i = 0; i < D; ++i) {
    if (uses[i] != 1) {
      return "every channel must be mapped from exactly one (instrument, side)";
    }
    if (m.direction[i] < -1 || m.direction[i] > 1) {
      return "channel direction must be -1, 0 or +1";
    }
  }
  return nullptr;
}

struct BookState {
  bool valid = false;
  core::TimestampNs t = 0;
  std::int64_t bid_px_e8 = 0;
  std::int64_t bid_qty_e8 = 0;
  std::int64_t ask_px_e8 = 0;
  std::int64_t ask_qty_e8 = 0;

  double mid() const noexcept {
    return 0.5 * (core::e8_to_double(bid_px_e8) + core::e8_to_double(ask_px_e8));
  }
  // OBI = (q_bid - q_ask) / (q_bid + q_ask) in [-1, 1].
  double imbalance() const noexcept {
    const double qb = static_cast<double>(bid_qty_e8);
    const double qa = static_cast<double>(ask_qty_e8);
    return (qb - qa) / (qb + qa);
  }
};

// Raw features of one channel event (math_spec Section 2.3).
struct TradeFeatures {
  int channel = -1;
  double size = 0.0;   // traded quantity in units
  double depth = 0.0;  // sweep depth L in ticks
  double iota = 0.0;   // s_d * OBI just before the trade (0 if the book is missing)
  bool book_missing = false;
};

// Keeps the last top of book per instrument and maps validated trades to channel features.
template <std::size_t D>
class EventMapper {
 public:
  void init(const ChannelMap<D>& map) noexcept {
    map_ = map;
    books_ = {};
  }

  void on_book(const MarketEvent& e, core::TimestampNs t) noexcept {
    if (e.instrument >= kMaxInstruments) {
      return;
    }
    BookState& b = books_[e.instrument];
    b.valid = true;
    b.t = t;
    b.bid_px_e8 = e.bid_px_e8();
    b.bid_qty_e8 = e.bid_qty_e8();
    b.ask_px_e8 = e.ask_px_e8();
    b.ask_qty_e8 = e.ask_qty_e8();
  }

  // Returns false if the (instrument, side) pair is not mapped to a channel.
  bool map_trade(const MarketEvent& e, TradeFeatures& out) const noexcept {
    const int ch = map_.channel_of(e.instrument, e.side());
    if (ch < 0) {
      return false;
    }
    const InstrumentSpec& spec = map_.instruments[e.instrument];
    out.channel = ch;
    out.size = core::e8_to_double(e.trade_qty_e8());
    out.depth = static_cast<double>(std::llabs(e.px0_e8 - e.px1_e8)) / static_cast<double>(spec.tick_e8);
    const BookState& b = books_[e.instrument];
    out.book_missing = !b.valid;
    out.iota =
        b.valid ? static_cast<double>(map_.direction[static_cast<std::size_t>(ch)]) * b.imbalance() : 0.0;
    return true;
  }

  const BookState& book(std::size_t instrument) const noexcept { return books_[instrument]; }
  const ChannelMap<D>& map() const noexcept { return map_; }

 private:
  ChannelMap<D> map_{};
  std::array<BookState, kMaxInstruments> books_{};
};

}  // namespace pulselob::feed
