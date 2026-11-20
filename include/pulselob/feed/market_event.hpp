#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace pulselob::feed {

// Normalized market-data record: the one byte layout used everywhere (SPSC ring slots, NumPy
// structured arrays, memory-mapped .plev files). Specified in docs/architecture.md Section 7.
// Any change to this layout must bump kMarketEventSchemaVersion and python/pulselob/schema.py.
inline constexpr std::uint16_t kMarketEventSchemaVersion = 1;

enum class EventKind : std::uint8_t {
  Heartbeat = 0,  // no payload; advances time in quiet markets
  Trade = 1,      // one aggressive order (prints with equal venue, side and timestamp merged)
  BookTop = 2,    // best bid / best ask update
  Status = 3,     // feed status; code in `count` (see StatusCode)
};

enum class Side : std::uint8_t { None = 0, Buy = 1, Sell = 2 };

// Resolution of the source's exchange timestamp; ties are only meaningful at this resolution.
enum class TsResolution : std::uint8_t { Nanosecond = 0, Microsecond = 1, Millisecond = 2, Second = 3 };

enum class StatusCode : std::uint32_t {
  FeedGap = 1,  // sequence gap detected upstream
  Disconnected = 2,
  Reconnected = 3,
  RingOverflow = 4,  // producer found the ring full and dropped events
};

namespace event_flags {
inline constexpr std::uint8_t kSideMask = 0x03;      // bits 0-1: Side (aggressor, trades only)
inline constexpr std::uint8_t kResolutionShift = 2;  // bits 2-3: TsResolution
inline constexpr std::uint8_t kResolutionMask = 0x0C;
inline constexpr std::uint8_t kTsAdjusted = 0x10;  // bit 4: timestamp clamped by the normalizer
inline constexpr std::uint8_t kSynthetic = 0x20;   // bit 5: produced by the synthetic generator
inline constexpr std::uint8_t kSnapshot = 0x40;    // bit 6: BookTop taken from a snapshot
inline constexpr std::uint8_t kReserved = 0x80;    // bit 7: must be zero

constexpr std::uint8_t make(Side side, TsResolution res, std::uint8_t extra = 0) noexcept {
  return static_cast<std::uint8_t>(static_cast<unsigned>(side) |
                                   (static_cast<unsigned>(res) << kResolutionShift) | extra);
}
}  // namespace event_flags

struct MarketEvent {
  std::int64_t ts_exchange_ns;  //  0  matching-engine time (ns since epoch)
  std::int64_t ts_local_ns;     //  8  time the event became available to us (ns since epoch)
  std::int64_t px0_e8;          // 16  Trade: last (worst) price     | BookTop: best bid price
  std::int64_t qty0_e8;         // 24  Trade: total quantity          | BookTop: best bid quantity
  std::int64_t px1_e8;          // 32  Trade: first (best) price      | BookTop: best ask price
  std::int64_t qty1_e8;         // 40  Trade: notional (quote units)  | BookTop: best ask quantity
  std::uint64_t seq;            // 48  Trade: first trade id          | BookTop: book update id
  std::uint32_t count;          // 56  Trade: prints merged (>= 1)    | Status: StatusCode | else 0
  std::uint16_t instrument;     // 60  index into the instrument table (venue + symbol)
  EventKind kind;               // 62
  std::uint8_t flags;           // 63  see event_flags

  constexpr Side side() const noexcept { return static_cast<Side>(flags & event_flags::kSideMask); }
  constexpr TsResolution ts_resolution() const noexcept {
    return static_cast<TsResolution>((flags & event_flags::kResolutionMask) >> event_flags::kResolutionShift);
  }

  // Named views of the kind-dependent payload.
  constexpr std::int64_t trade_last_px_e8() const noexcept { return px0_e8; }
  constexpr std::int64_t trade_first_px_e8() const noexcept { return px1_e8; }
  constexpr std::int64_t trade_qty_e8() const noexcept { return qty0_e8; }
  constexpr std::int64_t trade_notional_e8() const noexcept { return qty1_e8; }
  constexpr std::int64_t bid_px_e8() const noexcept { return px0_e8; }
  constexpr std::int64_t bid_qty_e8() const noexcept { return qty0_e8; }
  constexpr std::int64_t ask_px_e8() const noexcept { return px1_e8; }
  constexpr std::int64_t ask_qty_e8() const noexcept { return qty1_e8; }
};

static_assert(sizeof(MarketEvent) == 64, "MarketEvent must be exactly 64 bytes");
static_assert(alignof(MarketEvent) == 8, "natural alignment so NumPy buffers can be reinterpreted");
static_assert(std::is_trivially_copyable_v<MarketEvent>);
static_assert(std::is_standard_layout_v<MarketEvent>);
static_assert(std::is_aggregate_v<MarketEvent>);
static_assert(offsetof(MarketEvent, ts_exchange_ns) == 0);
static_assert(offsetof(MarketEvent, ts_local_ns) == 8);
static_assert(offsetof(MarketEvent, px0_e8) == 16);
static_assert(offsetof(MarketEvent, qty0_e8) == 24);
static_assert(offsetof(MarketEvent, px1_e8) == 32);
static_assert(offsetof(MarketEvent, qty1_e8) == 40);
static_assert(offsetof(MarketEvent, seq) == 48);
static_assert(offsetof(MarketEvent, count) == 56);
static_assert(offsetof(MarketEvent, instrument) == 60);
static_assert(offsetof(MarketEvent, kind) == 62);
static_assert(offsetof(MarketEvent, flags) == 63);

enum class EventCheck : std::uint8_t {
  Ok = 0,
  BadKind = 1,
  ReservedFlag = 2,
  BadSide = 3,  // trade without a Buy/Sell aggressor, or a side on a non-trade event
  NonPositivePrice = 4,
  NonPositiveQuantity = 5,
  BadCount = 6,        // trade with count == 0, or nonzero count on a BookTop/Heartbeat
  PriceOrder = 7,      // trade sweep walking the wrong way (buy: last < first; sell: last > first)
  CrossedBook = 8,     // BookTop with bid >= ask
  NonZeroPayload = 9,  // Heartbeat/Status with price or quantity fields set
  NegativeNotional = 10,
};

// Structural validation of one record. Cheap (no branches on data size); the Phase 4 loaders and
// the Python binding run it before replay, and the engine relies on it (math_spec Section 9 step 1).
constexpr EventCheck validate(const MarketEvent& e) noexcept {
  if ((e.flags & event_flags::kReserved) != 0) {
    return EventCheck::ReservedFlag;
  }
  const Side side = e.side();
  switch (e.kind) {
    case EventKind::Trade:
      if (side != Side::Buy && side != Side::Sell) {
        return EventCheck::BadSide;
      }
      if (e.px0_e8 <= 0 || e.px1_e8 <= 0) {
        return EventCheck::NonPositivePrice;
      }
      if (e.qty0_e8 <= 0) {
        return EventCheck::NonPositiveQuantity;
      }
      if (e.qty1_e8 < 0) {
        return EventCheck::NegativeNotional;
      }
      if (e.count == 0) {
        return EventCheck::BadCount;
      }
      if ((side == Side::Buy && e.px0_e8 < e.px1_e8) || (side == Side::Sell && e.px0_e8 > e.px1_e8)) {
        return EventCheck::PriceOrder;
      }
      return EventCheck::Ok;
    case EventKind::BookTop:
      if (side != Side::None) {
        return EventCheck::BadSide;
      }
      if (e.px0_e8 <= 0 || e.px1_e8 <= 0) {
        return EventCheck::NonPositivePrice;
      }
      if (e.qty0_e8 <= 0 || e.qty1_e8 <= 0) {
        return EventCheck::NonPositiveQuantity;
      }
      if (e.count != 0) {
        return EventCheck::BadCount;
      }
      if (e.px0_e8 >= e.px1_e8) {
        return EventCheck::CrossedBook;
      }
      return EventCheck::Ok;
    case EventKind::Heartbeat:
    case EventKind::Status:
      if (side != Side::None) {
        return EventCheck::BadSide;
      }
      if (e.px0_e8 != 0 || e.qty0_e8 != 0 || e.px1_e8 != 0 || e.qty1_e8 != 0) {
        return EventCheck::NonZeroPayload;
      }
      if (e.kind == EventKind::Heartbeat && e.count != 0) {
        return EventCheck::BadCount;
      }
      return EventCheck::Ok;
  }
  return EventCheck::BadKind;
}

}  // namespace pulselob::feed
