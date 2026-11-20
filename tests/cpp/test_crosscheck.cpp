// Cross-check against the definition-level Python reference (research/phase3/crosscheck_reference.py).
// The fixture holds the configuration, the input events and the reference outputs for every event:
// signals (intensities, toxicity, reservation shift, adverse-selection premia, variance, quotes,
// flags, status) and, every third model step, Theta~ (the batch discounted least-squares estimate
// rebuilt from its definition), Theta^, lambda_inf, v, the CUSUM statistics and n_eff.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "pulselob/engine/config_io.hpp"
#include "pulselob/engine/engine.hpp"
#include "report.hpp"
#include "test_config.hpp"

namespace {

using namespace pulselob;
using testkit::kD;
using testkit::kP;
using testkit::Shape;

struct Expect {
  int status = 0;
  std::int64_t t = 0;
  std::uint32_t flags = 0;
  std::vector<double> sig;  // tau_b tau_s crit shift adv_a adv_b sigma2
  std::int64_t bid = 0;
  std::int64_t ask = 0;
  std::vector<double> lam;
  std::vector<double> model;  // empty if no model record
};

struct Fixture {
  std::map<std::string, std::vector<std::string>> cfg;
  std::vector<feed::MarketEvent> events;
  std::vector<Expect> expect;
};

Fixture load() {
  std::ifstream in(std::string(PULSELOB_FIXTURE_DIR) + "/phase3_crosscheck.txt");
  Fixture f;
  if (!in) {
    return f;
  }
  std::string line;
  std::string section;
  std::size_t n_events = 0;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    if (line[0] == '[') {
      section = line;
      continue;
    }
    std::istringstream ss(line);
    if (section == "[config]") {
      std::string key;
      ss >> key;
      std::string v;
      while (ss >> v) {
        f.cfg[key].push_back(v);
      }
    } else if (section == "[events]") {
      if (n_events == 0) {
        ss >> n_events;
        continue;
      }
      feed::MarketEvent e{};
      long long a[11];
      for (auto& x : a) {
        ss >> x;
      }
      e.ts_exchange_ns = a[0];
      e.ts_local_ns = a[1];
      e.px0_e8 = a[2];
      e.qty0_e8 = a[3];
      e.px1_e8 = a[4];
      e.qty1_e8 = a[5];
      e.seq = static_cast<std::uint64_t>(a[6]);
      e.count = static_cast<std::uint32_t>(a[7]);
      e.instrument = static_cast<std::uint16_t>(a[8]);
      e.kind = static_cast<feed::EventKind>(a[9]);
      e.flags = static_cast<std::uint8_t>(a[10]);
      f.events.push_back(e);
    } else if (section == "[expect]") {
      Expect x;
      long long st = 0;
      long long t = 0;
      long long fl = 0;
      ss >> st >> t >> fl;
      x.status = static_cast<int>(st);
      x.t = t;
      x.flags = static_cast<std::uint32_t>(fl);
      for (int k = 0; k < 7; ++k) {
        double v = 0;
        ss >> v;
        x.sig.push_back(v);
      }
      long long b = 0;
      long long a = 0;
      ss >> b >> a;
      x.bid = b;
      x.ask = a;
      for (std::size_t i = 0; i < kD; ++i) {
        double v = 0;
        ss >> v;
        x.lam.push_back(v);
      }
      std::string bar;
      if (ss >> bar && bar == "|") {
        double v = 0;
        while (ss >> v) {
          x.model.push_back(v);
        }
      }
      f.expect.push_back(x);
    }
  }
  return f;
}

double num(const Fixture& f, const std::string& k, std::size_t i = 0) {
  return std::stod(f.cfg.at(k).at(i));
}

// The fixture's [config] section is the text form of engine/config_io.hpp (plus the run-time
// inventory), so the shared parser builds the configuration.
engine::EngineConfig<Shape> make_config(const Fixture& f) {
  std::string text;
  for (const auto& [key, values] : f.cfg) {
    text += key;
    for (const auto& v : values) {
      text += " " + v;
    }
    text += "\n";
  }
  engine::EngineConfig<Shape> c{};
  const char* err = nullptr;
  std::size_t line = 0;
  const std::string_view ignore[] = {"inventory"};
  EXPECT_TRUE(engine::parse_config_text(text, c, err, line, ignore))
      << (err ? err : "") << " at line " << line;
  return c;
}

struct Report {
  double sig = 0.0;
  double lam = 0.0;
  double theta_tilde = 0.0;
  double theta_hat = 0.0;
  double other = 0.0;
  std::size_t quote_mismatch = 0;
  std::size_t flag_mismatch = 0;
  std::size_t status_mismatch = 0;
  std::size_t model_records = 0;
};

double unit_rel(double a, double b) {
  return std::fabs(a - b) / std::fmax(std::fabs(b), 1.0);
}

template <class Est>
Report run(const Fixture& f) {
  const auto cfg = make_config(f);
  EXPECT_EQ(engine::validate(cfg), nullptr);
  auto eng = std::make_unique<engine::Engine<Shape, Est>>(cfg);
  eng->set_inventory(num(f, "inventory"));
  Report r;
  for (std::size_t k = 0; k < f.events.size(); ++k) {
    const auto st = eng->on_event(f.events[k]);
    const Expect& x = f.expect[k];
    const auto& s = eng->signal();
    r.status_mismatch += (static_cast<int>(s.status) != x.status) ? 1U : 0U;
    (void)st;
    r.flag_mismatch += s.flags != x.flags ? 1U : 0U;
    r.quote_mismatch += (s.bid_px_e8 != x.bid || s.ask_px_e8 != x.ask) ? 1U : 0U;
    const double got[7] = {s.tau_buy,     s.tau_sell,    s.criticality, s.reservation_shift,
                           s.adverse_ask, s.adverse_bid, s.sigma2};
    for (int i = 0; i < 7; ++i) {
      const double e = unit_rel(got[i], x.sig[static_cast<std::size_t>(i)]);
      if (e > r.sig && std::getenv("PULSELOB_CROSSCHECK_VERBOSE") != nullptr) {
        std::printf("  new worst signal %d at event %zu: got %.17g ref %.17g\n", i, k, got[i],
                    x.sig[static_cast<std::size_t>(i)]);
      }
      r.sig = std::fmax(r.sig, e);
    }
    for (std::size_t i = 0; i < kD; ++i) {
      r.lam = std::fmax(r.lam, std::fabs(s.lambda[i] - x.lam[i]) / std::fabs(x.lam[i]));
    }
    if (x.model.empty()) {
      continue;
    }
    ++r.model_records;
    const auto& tt = eng->model().estimator().theta();
    const auto& sv = eng->model().served();
    double num_t = 0.0;
    double den_t = 0.0;
    double num_h = 0.0;
    double den_h = 0.0;
    for (std::size_t i = 0; i < kD; ++i) {
      for (std::size_t p = 0; p < kP; ++p) {
        const double rt = x.model[i * kP + p];
        const double rh = x.model[kD * kP + i * kP + p];
        num_t = std::fmax(num_t, std::fabs(tt(i, p) - rt));
        den_t = std::fmax(den_t, std::fabs(rt));
        num_h = std::fmax(num_h, std::fabs(sv.theta(i, p) - rh));
        den_h = std::fmax(den_h, std::fabs(rh));
      }
    }
    r.theta_tilde = std::fmax(r.theta_tilde, num_t / den_t);
    r.theta_hat = std::fmax(r.theta_hat, num_h / den_h);
    std::size_t o = 2 * kD * kP;
    for (std::size_t i = 0; i < kD; ++i) {
      r.other = std::fmax(r.other, unit_rel(sv.lambda_inf[i], x.model[o + i]));
      r.other = std::fmax(r.other, unit_rel(sv.v[i], x.model[o + kD + i]));
    }
    o += 2 * kD;
    const auto& cs = eng->model().detector().cusum();
    for (std::size_t a = 0; a < 2; ++a) {
      for (std::size_t i = 0; i < kD; ++i) {
        r.other = std::fmax(r.other, unit_rel(cs(a, i), x.model[o + a * kD + i]));
      }
    }
    o += 2 * kD;
    r.other = std::fmax(r.other, unit_rel(sv.n_eff, x.model[o]));
  }
  EXPECT_EQ(eng->model().estimator().factorization_failures(), 0U);
  EXPECT_EQ(eng->model().counters().long_gaps, 2U);
  EXPECT_GT(eng->model().counters().alarms, 0U);
  EXPECT_EQ(eng->counters().ooo_clamped, 1U);
  EXPECT_EQ(eng->counters().ooo_rejected, 1U);
  EXPECT_EQ(eng->counters().unmapped, 1U);
  return r;
}

// signal_tol: the Potter back-end meets 1e-10 on Theta~ and Theta^ (math_spec V5) but its
// covariance-form rounding (U U^T vs M^{-1}, worst at start-up when data dominate a badly scaled
// ridge) is amplified up to ~1e-9 in cancellation-sensitive derived signals (the reservation shift
// uses y^inf, a difference of nearly equal kernel masses). docs/benchmarks.md reports this; the
// default Cholesky back-end meets 1e-10 everywhere.
template <class Est>
void check(const char* name, double signal_tol) {
  const Fixture f = load();
  if (f.events.empty()) {
    GTEST_FAIL() << "fixture missing: run research/phase3/crosscheck_reference.py --write";
  }
  ASSERT_EQ(f.events.size(), f.expect.size());
  const Report r = run<Est>(f);
  std::printf(
      "[crosscheck %s] events=%zu model_records=%zu theta~=%.2e theta^=%.2e signals=%.2e lambda=%.2e "
      "lambda_inf/v/cusum/n_eff=%.2e quote/flag/status mismatches=%zu/%zu/%zu\n",
      name, f.events.size(), r.model_records, r.theta_tilde, r.theta_hat, r.sig, r.lam, r.other,
      r.quote_mismatch, r.flag_mismatch, r.status_mismatch);
  ::testing::Test::RecordProperty("theta_tilde", testkit::sci(r.theta_tilde));
  ::testing::Test::RecordProperty("theta_hat", testkit::sci(r.theta_hat));
  ::testing::Test::RecordProperty("signals", testkit::sci(r.sig));
  EXPECT_GT(r.model_records, 50U);
  EXPECT_LT(r.theta_tilde, 1e-10);
  EXPECT_LT(r.theta_hat, 1e-10);
  EXPECT_LT(r.sig, signal_tol);
  EXPECT_LT(r.lam, signal_tol);
  EXPECT_LT(r.other, signal_tol);
  EXPECT_EQ(r.quote_mismatch, 0U);
  EXPECT_EQ(r.flag_mismatch, 0U);
  EXPECT_EQ(r.status_mismatch, 0U);
}

TEST(CrossCheck, CholeskyEngineMatchesPythonReference) {
  check<hawkes::CholeskyEstimator<Shape>>("cholesky", 1e-10);
}
TEST(CrossCheck, PotterEngineMatchesPythonReference) {
  check<hawkes::PotterEstimator<Shape>>("potter", 1e-8);
}

}  // namespace
