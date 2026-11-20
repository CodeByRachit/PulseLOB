#include "pulselob/feed/market_event.hpp"

#include <gtest/gtest.h>

#include <cstdint>

namespace {

using namespace pulselob::feed;

constexpr MarketEvent make_trade(Side side, std::int64_t first, std::int64_t last) {
  MarketEvent e{};
  e.ts_exchange_ns = 1'709'251'200'081'000'000;
  e.ts_local_ns = e.ts_exchange_ns;
  e.px0_e8 = last;
  e.px1_e8 = first;
  e.qty0_e8 = 294'100'000'000;
  e.qty1_e8 = 41'303'404'000;
  e.seq = 419'003'491;
  e.count = 2;
  e.instrument = 1;
  e.kind = EventKind::Trade;
  e.flags = event_flags::make(side, TsResolution::Millisecond);
  return e;
}

constexpr MarketEvent make_book(std::int64_t bid, std::int64_t ask) {
  MarketEvent e{};
  e.ts_exchange_ns = 1'709'251'200'081'000'000;
  e.ts_local_ns = 1'709'251'200'095'000'000;
  e.px0_e8 = bid;
  e.qty0_e8 = 1'517'300'000'000;
  e.px1_e8 = ask;
  e.qty1_e8 = 7'277'200'000'000;
  e.seq = 4'070'922'169'940;
  e.kind = EventKind::BookTop;
  e.flags = event_flags::make(Side::None, TsResolution::Millisecond);
  return e;
}

static_assert(validate(make_trade(Side::Buy, 14'044'000, 14'045'000)) == EventCheck::Ok);
static_assert(validate(make_book(14'043'000, 14'045'000)) == EventCheck::Ok);

TEST(MarketEvent, FlagsRoundTrip) {
  const auto e = make_trade(Side::Sell, 14'045'000, 14'044'000);
  EXPECT_EQ(e.side(), Side::Sell);
  EXPECT_EQ(e.ts_resolution(), TsResolution::Millisecond);
  EXPECT_EQ(e.trade_last_px_e8(), 14'044'000);
  EXPECT_EQ(e.trade_first_px_e8(), 14'045'000);
  const auto flags = event_flags::make(Side::Buy, TsResolution::Microsecond, event_flags::kSynthetic);
  EXPECT_EQ(flags, 0x01 | (1 << 2) | 0x20);
}

TEST(MarketEvent, TradeValidation) {
  EXPECT_EQ(validate(make_trade(Side::Buy, 100, 100)), EventCheck::Ok);
  EXPECT_EQ(validate(make_trade(Side::Buy, 101, 100)), EventCheck::PriceOrder);
  EXPECT_EQ(validate(make_trade(Side::Sell, 100, 101)), EventCheck::PriceOrder);
  EXPECT_EQ(validate(make_trade(Side::None, 100, 100)), EventCheck::BadSide);
  EXPECT_EQ(validate(make_trade(Side::Buy, 0, 100)), EventCheck::NonPositivePrice);
  auto e = make_trade(Side::Buy, 100, 100);
  e.qty0_e8 = 0;
  EXPECT_EQ(validate(e), EventCheck::NonPositiveQuantity);
  e = make_trade(Side::Buy, 100, 100);
  e.count = 0;
  EXPECT_EQ(validate(e), EventCheck::BadCount);
  e = make_trade(Side::Buy, 100, 100);
  e.qty1_e8 = -1;
  EXPECT_EQ(validate(e), EventCheck::NegativeNotional);
  e = make_trade(Side::Buy, 100, 100);
  e.flags |= event_flags::kReserved;
  EXPECT_EQ(validate(e), EventCheck::ReservedFlag);
}

TEST(MarketEvent, BookValidation) {
  EXPECT_EQ(validate(make_book(100, 100)), EventCheck::CrossedBook);
  EXPECT_EQ(validate(make_book(101, 100)), EventCheck::CrossedBook);
  auto e = make_book(100, 101);
  e.flags = event_flags::make(Side::Buy, TsResolution::Millisecond);
  EXPECT_EQ(validate(e), EventCheck::BadSide);
  e = make_book(100, 101);
  e.qty1_e8 = 0;
  EXPECT_EQ(validate(e), EventCheck::NonPositiveQuantity);
  e = make_book(100, 101);
  e.count = 3;
  EXPECT_EQ(validate(e), EventCheck::BadCount);
}

TEST(MarketEvent, HeartbeatStatusAndUnknownKind) {
  MarketEvent hb{};
  hb.kind = EventKind::Heartbeat;
  EXPECT_EQ(validate(hb), EventCheck::Ok);
  hb.count = 1;
  EXPECT_EQ(validate(hb), EventCheck::BadCount);

  MarketEvent st{};
  st.kind = EventKind::Status;
  st.count = static_cast<std::uint32_t>(StatusCode::FeedGap);
  EXPECT_EQ(validate(st), EventCheck::Ok);
  st.px0_e8 = 1;
  EXPECT_EQ(validate(st), EventCheck::NonZeroPayload);

  MarketEvent bad{};
  bad.kind = static_cast<EventKind>(9);
  EXPECT_EQ(validate(bad), EventCheck::BadKind);
}

}  // namespace
