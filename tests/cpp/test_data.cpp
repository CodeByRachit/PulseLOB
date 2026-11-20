#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <string_view>
#include <vector>

#include "hawkes_sim.hpp"
#include "pulselob/engine/config_io.hpp"
#include "pulselob/feed/binance_csv.hpp"
#include "test_config.hpp"

namespace {

using namespace pulselob;
using feed::binance::CsvKind;
using feed::binance::CsvOptions;
using feed::binance::parse_csv;

// Rows copied from data.binance.vision files (LTCUSDT, downloaded 2026-11-20).
constexpr std::string_view kFuturesAgg =
    "agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker\n"
    "310372335,80.0,16.078,861872724,861872730,1709251200028,false\n"
    "310372336,80.01,95.431,861872731,861872742,1709251200071,false\n";
constexpr std::string_view kSpotAgg2025 =
    "291960155,87.03000000,14.14900000,463867078,463867078,1748736000358530,False,True\n"
    "291960156,87.02000000,1.11800000,463867079,463867082,1748736000881648,True,True\n";
constexpr std::string_view kFuturesBook =
    "update_id,best_bid_price,best_bid_qty,best_ask_price,best_ask_qty,transaction_time,event_time\r\n"
    "4070922147230,79.99000000,133.42100000,80.00000000,56.77300000,1709251200014,1709251200020\r\n";

TEST(BinanceCsv, FuturesAggTradesWithHeaderInMilliseconds) {
  std::vector<feed::MarketEvent> out;
  CsvOptions opt{};
  opt.instrument = 0;
  opt.latency_ns = 2'000'000;
  const auto r = parse_csv(kFuturesAgg, opt, out);
  ASSERT_TRUE(r.ok);
  EXPECT_TRUE(r.header);
  EXPECT_EQ(r.resolution, feed::TsResolution::Millisecond);
  ASSERT_EQ(out.size(), 2U);
  const auto& e = out[0];
  EXPECT_EQ(e.kind, feed::EventKind::Trade);
  EXPECT_EQ(e.side(), feed::Side::Buy);  // is_buyer_maker = false: the buyer was the aggressor
  EXPECT_EQ(e.px0_e8, 8'000'000'000);
  EXPECT_EQ(e.qty0_e8, 1'607'800'000);
  EXPECT_EQ(e.ts_exchange_ns, 1'709'251'200'028'000'000);
  EXPECT_EQ(e.ts_local_ns, e.ts_exchange_ns + 2'000'000);
  EXPECT_EQ(e.count, 7U);  // trades 861872724..861872730
  EXPECT_EQ(e.seq, 310372335U);
  EXPECT_EQ(e.ts_resolution(), feed::TsResolution::Millisecond);
}

TEST(BinanceCsv, SpotAggTradesWithoutHeaderInMicroseconds) {
  std::vector<feed::MarketEvent> out;
  const auto r = parse_csv(kSpotAgg2025, CsvOptions{CsvKind::AggTrades, 1, 0, true}, out);
  ASSERT_TRUE(r.ok);
  EXPECT_FALSE(r.header);
  EXPECT_EQ(r.resolution, feed::TsResolution::Microsecond);
  ASSERT_EQ(out.size(), 2U);
  EXPECT_EQ(out[0].ts_exchange_ns, 1'748'736'000'358'530'000);
  EXPECT_EQ(out[0].side(), feed::Side::Buy);   // False
  EXPECT_EQ(out[1].side(), feed::Side::Sell);  // True: the buyer was the maker
  EXPECT_EQ(out[1].instrument, 1);
}

TEST(BinanceCsv, SweepsAreMergedIntoOneValidEvent) {
  // Three sells at one millisecond (a sweep down the book), then a buy at the same time.
  const std::string csv =
      "1,100.00,1.0,10,10,1709251200000,true\n"
      "2,99.99,2.0,11,12,1709251200000,true\n"
      "3,99.98,0.5,13,13,1709251200000,true\n"
      "4,100.01,1.0,14,14,1709251200000,false\n";
  std::vector<feed::MarketEvent> out;
  const auto r = parse_csv(csv, CsvOptions{}, out);
  ASSERT_TRUE(r.ok);
  EXPECT_EQ(r.merged, 2U);
  ASSERT_EQ(out.size(), 2U);
  EXPECT_EQ(out[0].side(), feed::Side::Sell);
  EXPECT_EQ(out[0].px1_e8, 10'000'000'000);  // first price
  EXPECT_EQ(out[0].px0_e8, 9'998'000'000);   // worst price reached
  EXPECT_EQ(out[0].qty0_e8, 350'000'000);
  EXPECT_EQ(out[0].count, 4U);
  EXPECT_EQ(feed::validate(out[0]), feed::EventCheck::Ok);
  EXPECT_EQ(out[1].side(), feed::Side::Buy);
  std::vector<feed::MarketEvent> unmerged;
  CsvOptions no_merge{};
  no_merge.merge_sweeps = false;
  EXPECT_EQ(parse_csv(csv, no_merge, unmerged).events, 4U);
}

TEST(BinanceCsv, BookTickerUsesTransactionAndEventTimes) {
  std::vector<feed::MarketEvent> out;
  const auto r = parse_csv(kFuturesBook, CsvOptions{CsvKind::BookTicker, 0, 1'000'000, true}, out);
  ASSERT_TRUE(r.ok);
  ASSERT_EQ(out.size(), 1U);
  EXPECT_EQ(out[0].kind, feed::EventKind::BookTop);
  EXPECT_EQ(out[0].ts_exchange_ns, 1'709'251'200'014'000'000);
  EXPECT_EQ(out[0].ts_local_ns, 1'709'251'200'020'000'000 + 1'000'000);
  EXPECT_EQ(out[0].bid_px_e8(), 7'999'000'000);
  EXPECT_EQ(out[0].ask_qty_e8(), 5'677'300'000);
  EXPECT_EQ(out[0].seq, 4070922147230U);
}

TEST(BinanceCsv, InvalidRowsAreCountedAndMalformedRowsFail) {
  std::vector<feed::MarketEvent> out;
  const auto crossed = parse_csv("1,100.0,1.0,100.0,1.0,1709251200014,1709251200020\n",
                                 CsvOptions{CsvKind::BookTicker, 0, 0, true}, out);
  ASSERT_TRUE(crossed.ok);
  EXPECT_EQ(crossed.invalid, 1U);
  EXPECT_TRUE(out.empty());
  const auto bad = parse_csv("1,80.0,1.0,1,1,1709251200028,false\n2,80.0.1,1.0,2,2,1709251200029,false\n",
                             CsvOptions{}, out);
  EXPECT_FALSE(bad.ok);
  EXPECT_EQ(bad.error_line, 2U);
  const auto ninth = parse_csv("1,80.000000001,1.0,1,1,1709251200028,false\n", CsvOptions{}, out);
  EXPECT_FALSE(ninth.ok);  // a nonzero 9th decimal cannot be represented exactly in e8
}

TEST(ConfigIo, TextRoundTripAndErrors) {
  const auto base = testkit::default_engine_config();
  engine::EngineConfig<testkit::Shape> c = base;
  const char* err = nullptr;
  std::size_t line = 0;
  ASSERT_TRUE(
      engine::parse_config_text("# comment\nrho_max 0.9  # trailing\n\ncusum_threshold 30\n"
                                "instruments 2 0 0 10000000 1 1 1000000\n",
                                c, err, line));
  EXPECT_EQ(c.model.rho_max, 0.9);
  EXPECT_EQ(c.model.cusum_threshold, 30.0);
  EXPECT_EQ(engine::validate(c), nullptr);
  EXPECT_FALSE(engine::parse_config_text("rho_max 0.9\nno_such_key 1\n", c, err, line));
  EXPECT_EQ(line, 2U);
  EXPECT_FALSE(engine::parse_config_text("omega 1 2 3\n", c, err, line));
  EXPECT_FALSE(engine::parse_config_text("rho_max abc\n", c, err, line));
  EXPECT_FALSE(engine::parse_config_text("channel_map 1 0 3 0\n", c, err, line));  // side 3
}

// Piecewise-constant parameters: with no excitation the process is Poisson in each segment, so the
// segment counts must match the segment rates.
TEST(Synth, SegmentsSwitchParametersAtTheirStartTimes) {
  core::FixedMatrix<4, testkit::kP> calm{};
  core::FixedMatrix<4, testkit::kP> burst{};
  for (std::size_t i = 0; i < 4; ++i) {
    calm(i, 0) = 50.0;
    burst(i, 0) = i == 0 ? 2000.0 : 50.0;
  }
  synth::HawkesSimulator<testkit::Shape> sim(testkit::default_beta(), calm);
  sim.schedule_switch(10.0, burst);
  sim.schedule_switch(12.0, calm);
  synth::Rng rng(3);
  std::vector<double> lam;
  const auto ev = sim.simulate(20.0, rng, 10'000'000, &lam);
  ASSERT_EQ(lam.size(), ev.size() * 4);
  double in_burst = 0;
  double outside = 0;
  for (std::size_t k = 0; k < ev.size(); ++k) {
    const bool b = ev[k].t_s >= 10.0 && ev[k].t_s < 12.0;
    if (ev[k].channel == 0) {
      (b ? in_burst : outside) += 1.0;
    }
    EXPECT_EQ(lam[4 * k], b ? 2000.0 : 50.0);  // recorded true intensity
  }
  EXPECT_NEAR(in_burst, 4000.0, 5.0 * std::sqrt(4000.0));
  EXPECT_NEAR(outside, 900.0, 5.0 * std::sqrt(900.0));
}

}  // namespace
