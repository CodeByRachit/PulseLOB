// pulselob-replay: runs a .plev event file through the C++ engine and writes one Signal record
// (128 bytes) per event. It is the native reference path for the Python bindings' parity test
// and a standalone replay tool.
//
//   pulselob-replay --config CONFIG.txt --input EVENTS.plev --output SIGNALS.bin
//                   [--shape default|exp3] [--estimator cholesky|potter] [--mode inline|split]
//                   [--inventory Q] [--repeat N]
//
// CONFIG.txt is the text form of engine/config_io.hpp (pulselob.EngineConfig.to_text() writes it).
// The output is the raw array of Signal records, readable with numpy.fromfile(path, SIGNAL_DTYPE).

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "pulselob/core/fp_env.hpp"
#include "pulselob/core/model_shape.hpp"
#include "pulselob/engine/config_io.hpp"
#include "pulselob/engine/engine.hpp"
#include "pulselob/engine/split_engine.hpp"
#include "pulselob/feed/market_event.hpp"

namespace {

using pulselob::feed::MarketEvent;

int fail(const std::string& msg) {
  std::fprintf(stderr, "pulselob-replay: %s\n", msg.c_str());
  return 2;
}

bool read_plev(const std::string& path, std::vector<MarketEvent>& out, std::string& err) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    err = "cannot open " + path;
    return false;
  }
  unsigned char h[64];
  f.read(reinterpret_cast<char*>(h), 64);
  if (f.gcount() != 64 || std::memcmp(h, "PLEV", 4) != 0) {
    err = path + ": not a .plev file";
    return false;
  }
  std::uint16_t schema = 0;
  std::uint16_t record = 0;
  std::uint32_t header = 0;
  std::uint64_t n = 0;
  std::memcpy(&schema, h + 4, 2);
  std::memcpy(&record, h + 6, 2);
  std::memcpy(&header, h + 8, 4);
  std::memcpy(&n, h + 16, 8);
  if (schema != pulselob::feed::kMarketEventSchemaVersion || record != sizeof(MarketEvent) || header != 64) {
    err = path + ": unsupported schema, record or header size";
    return false;
  }
  out.resize(n);
  f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(n * sizeof(MarketEvent)));
  if (static_cast<std::uint64_t>(f.gcount()) != n * sizeof(MarketEvent) || f.peek() != EOF) {
    err = path + ": size does not match the header";
    return false;
  }
  return true;
}

template <class E>
void drive(E& eng, const std::vector<MarketEvent>& ev, std::vector<pulselob::engine::Signal<4>>& sig) {
  for (std::size_t i = 0; i < ev.size(); ++i) {
    eng.on_event(ev[i]);
    sig[i] = eng.signal();
  }
}

template <class Shape, class Est>
int run(const std::string& cfg_text, const std::vector<MarketEvent>& ev, const std::string& mode,
        double inventory, int repeat, std::vector<pulselob::engine::Signal<4>>& sig, double& seconds) {
  pulselob::engine::EngineConfig<Shape> cfg{};
  const char* err = nullptr;
  std::size_t line = 0;
  if (!pulselob::engine::parse_config_text(cfg_text, cfg, err, line)) {
    return fail("config line " + std::to_string(line) + ": " + err);
  }
  if (const char* v = pulselob::engine::validate(cfg)) {
    return fail(std::string("invalid configuration: ") + v);
  }
  const pulselob::core::FlushToZeroGuard ftz;
  seconds = 1e300;
  for (int r = 0; r < repeat; ++r) {
    const auto t0 = std::chrono::steady_clock::now();
    if (mode == "split") {
      auto e = std::make_unique<pulselob::engine::SplitEngine<Shape, Est>>(cfg);
      e->set_inventory(inventory);
      drive(*e, ev, sig);
      e->drain();
      e->stop();
    } else {
      auto e = std::make_unique<pulselob::engine::Engine<Shape, Est>>(cfg);
      e->set_inventory(inventory);
      drive(*e, ev, sig);
    }
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    seconds = s < seconds ? s : seconds;
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  std::string config;
  std::string input;
  std::string output;
  std::string shape = "default";
  std::string estimator = "cholesky";
  std::string mode = "inline";
  double inventory = 0.0;
  int repeat = 1;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
    if (a == "--config") {
      config = next();
    } else if (a == "--input") {
      input = next();
    } else if (a == "--output") {
      output = next();
    } else if (a == "--shape") {
      shape = next();
    } else if (a == "--estimator") {
      estimator = next();
    } else if (a == "--mode") {
      mode = next();
    } else if (a == "--inventory") {
      inventory = std::stod(next());
    } else if (a == "--repeat") {
      repeat = std::stoi(next());
    } else {
      return fail("unknown argument " + a);
    }
  }
  if (config.empty() || input.empty()) {
    return fail("usage: pulselob-replay --config CONFIG.txt --input EVENTS.plev [--output SIGNALS.bin] ...");
  }
  std::ifstream cf(config);
  if (!cf) {
    return fail("cannot open " + config);
  }
  const std::string cfg_text((std::istreambuf_iterator<char>(cf)), std::istreambuf_iterator<char>());
  std::vector<MarketEvent> ev;
  std::string err;
  if (!read_plev(input, ev, err)) {
    return fail(err);
  }
  std::vector<pulselob::engine::Signal<4>> sig(ev.size());
  double seconds = 0.0;
  int rc = 0;
  using pulselob::hawkes::CholeskyEstimator;
  using pulselob::hawkes::PotterEstimator;
  using Default = pulselob::core::DefaultShape;
  using Exp3 = pulselob::core::ModelShape<4, 1, 1, 1>;
  if (shape == "default" && estimator == "cholesky") {
    rc = run<Default, CholeskyEstimator<Default>>(cfg_text, ev, mode, inventory, repeat, sig, seconds);
  } else if (shape == "default" && estimator == "potter") {
    rc = run<Default, PotterEstimator<Default>>(cfg_text, ev, mode, inventory, repeat, sig, seconds);
  } else if (shape == "exp3" && estimator == "cholesky") {
    rc = run<Exp3, CholeskyEstimator<Exp3>>(cfg_text, ev, mode, inventory, repeat, sig, seconds);
  } else if (shape == "exp3" && estimator == "potter") {
    rc = run<Exp3, PotterEstimator<Exp3>>(cfg_text, ev, mode, inventory, repeat, sig, seconds);
  } else {
    return fail("unknown --shape or --estimator");
  }
  if (rc != 0) {
    return rc;
  }
  if (!output.empty()) {
    std::ofstream of(output, std::ios::binary);
    of.write(reinterpret_cast<const char*>(sig.data()),
             static_cast<std::streamsize>(sig.size() * sizeof(pulselob::engine::Signal<4>)));
    if (!of) {
      return fail("cannot write " + output);
    }
  }
  std::printf("{\"events\": %zu, \"seconds\": %.6f, \"events_per_second\": %.0f}\n", ev.size(), seconds,
              seconds > 0 ? static_cast<double>(ev.size()) / seconds : 0.0);
  return 0;
}
