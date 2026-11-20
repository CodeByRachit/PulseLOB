// Python extension module pulselob._core (docs/architecture.md Section 10).
//
// Exposes the MarketEvent and Signal layouts, the engine (both model shapes, both estimator
// back-ends, inline and split modes) with a zero-copy batch path that releases the GIL, the
// synthetic Hawkes generator, and the exact Binance CSV parser. The Python package wraps these
// in pulselob.Engine, pulselob.synth and pulselob.data.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "pulselob/core/fp_env.hpp"
#include "pulselob/core/model_shape.hpp"
#include "pulselob/engine/config.hpp"
#include "pulselob/engine/config_io.hpp"
#include "pulselob/engine/engine.hpp"
#include "pulselob/engine/signal.hpp"
#include "pulselob/engine/split_engine.hpp"
#include "pulselob/feed/binance_csv.hpp"
#include "pulselob/feed/market_event.hpp"
#include "pulselob/hawkes/estimator.hpp"
#include "pulselob/synth/hawkes_simulator.hpp"
#include "pulselob/synth/market_stream.hpp"
#include "pulselob/version.hpp"

namespace py = pybind11;
using pulselob::feed::MarketEvent;
namespace pl = pulselob;

static_assert(std::endian::native == std::endian::little,
              "the NumPy dtypes and .plev files are little-endian");

namespace {

constexpr std::size_t kD = 4;
using Sig = pl::engine::Signal<kD>;
using DefaultShape = pl::core::DefaultShape;
using Exp3Shape = pl::core::ModelShape<4, 1, 1, 1>;  // fallback bank: three exponentials, P = 13

// ---------------------------------------------------------------------------------------------
// Layouts (the Python side builds its dtypes independently and checks them against these).
// ---------------------------------------------------------------------------------------------
py::list market_event_layout() {
  py::list f;
  const auto add = [&](const char* name, std::size_t offset, std::size_t size, const char* code) {
    f.append(py::make_tuple(name, offset, size, code));
  };
  add("ts_exchange_ns", offsetof(MarketEvent, ts_exchange_ns), 8, "<i8");
  add("ts_local_ns", offsetof(MarketEvent, ts_local_ns), 8, "<i8");
  add("px0_e8", offsetof(MarketEvent, px0_e8), 8, "<i8");
  add("qty0_e8", offsetof(MarketEvent, qty0_e8), 8, "<i8");
  add("px1_e8", offsetof(MarketEvent, px1_e8), 8, "<i8");
  add("qty1_e8", offsetof(MarketEvent, qty1_e8), 8, "<i8");
  add("seq", offsetof(MarketEvent, seq), 8, "<u8");
  add("count", offsetof(MarketEvent, count), 4, "<u4");
  add("instrument", offsetof(MarketEvent, instrument), 2, "<u2");
  add("kind", offsetof(MarketEvent, kind), 1, "u1");
  add("flags", offsetof(MarketEvent, flags), 1, "u1");
  return f;
}

py::list signal_layout() {
  py::list f;
  const auto add = [&](const char* name, std::size_t offset, std::size_t size, const char* code) {
    f.append(py::make_tuple(name, offset, size, code));
  };
  add("t_ns", offsetof(Sig, t_ns), 8, "<i8");
  add("flags", offsetof(Sig, flags), 4, "<u4");
  add("status", offsetof(Sig, status), 2, "<u2");
  add("reserved", offsetof(Sig, reserved), 2, "<u2");
  add("lambda", offsetof(Sig, lambda), 8 * kD, "<f8");
  add("tau_buy", offsetof(Sig, tau_buy), 8, "<f8");
  add("tau_sell", offsetof(Sig, tau_sell), 8, "<f8");
  add("criticality", offsetof(Sig, criticality), 8, "<f8");
  add("reservation_shift", offsetof(Sig, reservation_shift), 8, "<f8");
  add("adverse_ask", offsetof(Sig, adverse_ask), 8, "<f8");
  add("adverse_bid", offsetof(Sig, adverse_bid), 8, "<f8");
  add("sigma2", offsetof(Sig, sigma2), 8, "<f8");
  add("bid_px_e8", offsetof(Sig, bid_px_e8), 8, "<i8");
  add("ask_px_e8", offsetof(Sig, ask_px_e8), 8, "<i8");
  return f;
}

// Zero-copy view of a NumPy array as MarketEvent records (architecture Section 7.9).
const MarketEvent* as_events(const py::array& arr, std::size_t& n) {
  if (arr.ndim() != 1) {
    throw std::invalid_argument("events must be a 1-D array");
  }
  if (static_cast<std::size_t>(arr.itemsize()) != sizeof(MarketEvent)) {
    throw std::invalid_argument("events must use pulselob.MARKET_EVENT_DTYPE (itemsize " +
                                std::to_string(sizeof(MarketEvent)) + ")");
  }
  if ((arr.flags() & py::array::c_style) == 0) {
    throw std::invalid_argument("events must be C-contiguous (use numpy.ascontiguousarray)");
  }
  if (reinterpret_cast<std::uintptr_t>(arr.data()) % alignof(MarketEvent) != 0) {
    throw std::invalid_argument("events buffer must be 8-byte aligned");
  }
  n = static_cast<std::size_t>(arr.shape(0));
  return static_cast<const MarketEvent*>(arr.data());
}

// Checks an output array: writable, C-contiguous, elements of elem_bytes, at least min_len rows of
// row_elems elements each.
template <class T>
T* writable(py::array& a, std::size_t elem_bytes, std::size_t row_elems, std::size_t min_len,
            const char* what) {
  const auto rows = a.ndim() == 0 ? std::size_t{0} : static_cast<std::size_t>(a.shape(0));
  const bool ok = a.writeable() && (a.flags() & py::array::c_style) != 0 &&
                  static_cast<std::size_t>(a.itemsize()) == elem_bytes && rows >= min_len &&
                  (rows == 0 || static_cast<std::size_t>(a.size()) == rows * row_elems);
  if (!ok) {
    throw std::invalid_argument(std::string(what) +
                                ": expected a writable C-contiguous array of the right dtype/shape");
  }
  return static_cast<T*>(a.mutable_data());
}

py::array_t<std::uint8_t> validate_events(const py::array& events) {
  std::size_t n = 0;
  const MarketEvent* ev = as_events(events, n);
  py::array_t<std::uint8_t> codes(static_cast<py::ssize_t>(n));
  std::uint8_t* out = codes.mutable_data();
  {
    py::gil_scoped_release release;
    for (std::size_t i = 0; i < n; ++i) {
      out[i] = static_cast<std::uint8_t>(pl::feed::validate(ev[i]));
    }
  }
  return codes;
}

py::array_t<std::uint8_t> events_to_array(const std::vector<MarketEvent>& ev) {
  py::array_t<std::uint8_t> out(static_cast<py::ssize_t>(ev.size() * sizeof(MarketEvent)));
  if (!ev.empty()) {
    std::memcpy(out.mutable_data(), ev.data(), ev.size() * sizeof(MarketEvent));
  }
  return out;
}

// ---------------------------------------------------------------------------------------------
// Configuration from {key: [values]}.
// ---------------------------------------------------------------------------------------------
template <class Shape>
pl::engine::EngineConfig<Shape> config_from_dict(const py::dict& entries) {
  pl::engine::EngineConfig<Shape> c{};
  for (const auto& item : entries) {
    const std::string key = py::cast<std::string>(item.first);
    std::vector<double> v;
    if (py::isinstance<py::float_>(item.second) || py::isinstance<py::int_>(item.second) ||
        py::isinstance<py::bool_>(item.second)) {
      v.push_back(py::cast<double>(item.second));
    } else {
      v = py::cast<std::vector<double>>(item.second);
    }
    if (const char* err = pl::engine::set_config_value(c, key, std::span<const double>(v.data(), v.size()))) {
      throw std::invalid_argument("config '" + key + "': " + err);
    }
  }
  if (const char* err = pl::engine::validate(c)) {
    throw std::invalid_argument(std::string("invalid configuration: ") + err);
  }
  return c;
}

// ---------------------------------------------------------------------------------------------
// Engine: one interface over shape x estimator x mode.
// ---------------------------------------------------------------------------------------------
struct StateBuffers {
  std::int64_t* t_ns = nullptr;
  std::int64_t* index = nullptr;
  double* theta_hat = nullptr;     // [M, D, P]
  double* theta_tilde = nullptr;   // [M, D, P] (NaN in split mode)
  double* G = nullptr;             // [M, D, D]
  double* lambda_inf = nullptr;    // [M, D]
  double* criticality = nullptr;   // [M]
  double* n_eff = nullptr;         // [M]
  double* cusum = nullptr;         // [M, 2, D] (NaN in split mode)
  std::uint32_t* flags = nullptr;  // [M]: bit 0 fast forgetting, bits 8.. projected rows
  std::size_t capacity = 0;
};

class EngineIface {
 public:
  virtual ~EngineIface() = default;
  virtual int on_event(const MarketEvent& e) = 0;
  virtual const Sig& signal() const = 0;
  virtual void set_inventory(double q) = 0;
  virtual std::uint64_t steps() const = 0;
  virtual void record_state(StateBuffers& b, std::size_t row, std::int64_t t, std::int64_t index) const = 0;
  virtual void finish() = 0;
  virtual py::dict counters() const = 0;
  virtual py::dict state() const = 0;
  virtual std::size_t P() const = 0;
};

template <class Shape>
py::array_t<double> matrix(const pl::core::FixedMatrix<kD, Shape::P>& m) {
  py::array_t<double> a({static_cast<py::ssize_t>(kD), static_cast<py::ssize_t>(Shape::P)});
  auto r = a.mutable_unchecked<2>();
  for (std::size_t i = 0; i < kD; ++i) {
    for (std::size_t p = 0; p < Shape::P; ++p) {
      r(static_cast<py::ssize_t>(i), static_cast<py::ssize_t>(p)) = m(i, p);
    }
  }
  return a;
}

template <class Shape>
void put_served(const pl::hawkes::Served<Shape>& s, StateBuffers& b, std::size_t row) {
  constexpr std::size_t P = Shape::P;
  for (std::size_t i = 0; i < kD; ++i) {
    for (std::size_t p = 0; p < P; ++p) {
      b.theta_hat[(row * kD + i) * P + p] = s.theta(i, p);
    }
    for (std::size_t j = 0; j < kD; ++j) {
      b.G[(row * kD + i) * kD + j] = s.G(i, j);
    }
    b.lambda_inf[row * kD + i] = s.lambda_inf[i];
  }
  b.criticality[row] = s.criticality;
  b.n_eff[row] = s.n_eff;
  b.flags[row] = (s.flags & 1U) | (s.projected_rows << 8);
}

py::dict engine_counters(const pl::engine::EngineCounters& c) {
  py::dict d;
  d["inputs"] = c.inputs;
  d["trades"] = c.trades;
  d["books"] = c.books;
  d["heartbeats"] = c.heartbeats;
  d["statuses"] = c.statuses;
  d["invalid"] = std::vector<std::uint64_t>(c.invalid.begin(), c.invalid.end());
  d["ooo_clamped"] = c.ooo_clamped;
  d["ooo_rejected"] = c.ooo_rejected;
  d["unmapped"] = c.unmapped;
  d["missing_book_marks"] = c.missing_book_marks;
  d["nonfinite_inputs"] = c.nonfinite_inputs;
  d["feed_faults"] = c.feed_faults;
  d["ring_overflows"] = c.ring_overflows;
  return d;
}

void add_model_counters(py::dict& d, const pl::hawkes::ModelCounters& m) {
  d["model_steps"] = m.steps;
  d["model_events"] = m.events;
  d["model_heartbeats"] = m.heartbeats;
  d["intervals"] = m.intervals;
  d["cutoffs"] = m.cutoffs;
  d["long_gaps"] = m.long_gaps;
  d["projections"] = m.projections;
  d["alarms"] = m.alarms;
  d["rejected_steps"] = m.rejected_inputs;
}

template <class Shape>
py::dict served_dict(const pl::hawkes::Served<Shape>& s) {
  py::dict d;
  d["theta_hat"] = matrix<Shape>(s.theta);
  py::array_t<double> G({static_cast<py::ssize_t>(kD), static_cast<py::ssize_t>(kD)});
  auto g = G.mutable_unchecked<2>();
  py::array_t<double> li(static_cast<py::ssize_t>(kD));
  py::array_t<double> v(static_cast<py::ssize_t>(kD));
  for (std::size_t i = 0; i < kD; ++i) {
    for (std::size_t j = 0; j < kD; ++j) {
      g(static_cast<py::ssize_t>(i), static_cast<py::ssize_t>(j)) = s.G(i, j);
    }
    li.mutable_at(static_cast<py::ssize_t>(i)) = s.lambda_inf[i];
    v.mutable_at(static_cast<py::ssize_t>(i)) = s.v[i];
  }
  d["G"] = G;
  d["lambda_inf"] = li;
  d["v"] = v;
  d["criticality"] = s.criticality;
  d["n_eff"] = s.n_eff;
  d["steps"] = s.steps;
  d["fast_forgetting"] = (s.flags & 1U) != 0;
  d["projected_rows"] = s.projected_rows;
  return d;
}

template <class Shape, class Est>
class InlineImpl final : public EngineIface {
 public:
  explicit InlineImpl(const pl::engine::EngineConfig<Shape>& c)
      : e_(std::make_unique<pl::engine::Engine<Shape, Est>>(c)) {}
  int on_event(const MarketEvent& ev) override { return static_cast<int>(e_->on_event(ev)); }
  const Sig& signal() const override { return e_->signal(); }
  void set_inventory(double q) override { e_->set_inventory(q); }
  std::uint64_t steps() const override { return e_->model().counters().steps; }
  void record_state(StateBuffers& b, std::size_t row, std::int64_t t, std::int64_t index) const override {
    constexpr std::size_t P = Shape::P;
    const auto& m = e_->model();
    b.t_ns[row] = t;
    b.index[row] = index;
    put_served(m.served(), b, row);
    const auto& tt = m.estimator().theta();
    for (std::size_t i = 0; i < kD; ++i) {
      for (std::size_t p = 0; p < P; ++p) {
        b.theta_tilde[(row * kD + i) * P + p] = tt(i, p);
      }
    }
    for (std::size_t a = 0; a < 2; ++a) {
      for (std::size_t i = 0; i < kD; ++i) {
        b.cusum[(row * 2 + a) * kD + i] = m.detector().cusum()(a, i);
      }
    }
  }
  void finish() override {}
  py::dict counters() const override {
    py::dict d = engine_counters(e_->counters());
    add_model_counters(d, e_->model().counters());
    d["factorization_failures"] = e_->model().estimator().factorization_failures();
    d["kill_switch_trips"] = e_->decision().kill_switch().trips;
    d["ask_pull_cycles"] = e_->decision().ask_trigger().cycles;
    d["bid_pull_cycles"] = e_->decision().bid_trigger().cycles;
    d["ask_requotes"] = e_->decision().ask_throttle().requotes;
    d["bid_requotes"] = e_->decision().bid_throttle().requotes;
    return d;
  }
  py::dict state() const override {
    py::dict d = served_dict<Shape>(e_->model().served());
    d["theta_tilde"] = matrix<Shape>(e_->model().estimator().theta());
    py::array_t<double> cs({py::ssize_t{2}, static_cast<py::ssize_t>(kD)});
    auto c = cs.mutable_unchecked<2>();
    for (std::size_t a = 0; a < 2; ++a) {
      for (std::size_t i = 0; i < kD; ++i) {
        c(static_cast<py::ssize_t>(a), static_cast<py::ssize_t>(i)) = e_->model().detector().cusum()(a, i);
      }
    }
    d["cusum"] = cs;
    d["clock_ns"] = e_->clock();
    d["inventory"] = e_->inventory();
    return d;
  }
  std::size_t P() const override { return Shape::P; }

 private:
  std::unique_ptr<pl::engine::Engine<Shape, Est>> e_;
};

template <class Shape, class Est>
class SplitImpl final : public EngineIface {
 public:
  explicit SplitImpl(const pl::engine::EngineConfig<Shape>& c)
      : e_(std::make_unique<pl::engine::SplitEngine<Shape, Est>>(c)) {}
  ~SplitImpl() override { e_->stop(); }
  int on_event(const MarketEvent& ev) override { return static_cast<int>(e_->on_event(ev)); }
  const Sig& signal() const override { return e_->signal(); }
  void set_inventory(double q) override { e_->set_inventory(q); }
  std::uint64_t steps() const override { return e_->steps_pushed(); }
  void record_state(StateBuffers& b, std::size_t row, std::int64_t t, std::int64_t index) const override {
    constexpr std::size_t P = Shape::P;
    b.t_ns[row] = t;
    b.index[row] = index;
    put_served(e_->served(), b, row);  // the hot thread's (possibly lagging) parameters
    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (std::size_t k = 0; k < kD * P; ++k) {
      b.theta_tilde[row * kD * P + k] = nan;
    }
    for (std::size_t k = 0; k < 2 * kD; ++k) {
      b.cusum[row * 2 * kD + k] = nan;
    }
  }
  void finish() override { e_->drain(); }
  py::dict counters() const override {
    py::dict d = engine_counters(e_->counters());
    d["steps_pushed"] = e_->steps_pushed();
    d["stale_signals"] = e_->stale_signals();
    d["publications"] = e_->publications();
    return d;
  }
  py::dict state() const override {
    py::dict d = served_dict<Shape>(e_->served());
    d["steps_pushed"] = e_->steps_pushed();
    return d;
  }
  std::size_t P() const override { return Shape::P; }

 private:
  std::unique_ptr<pl::engine::SplitEngine<Shape, Est>> e_;
};

template <class Shape>
std::unique_ptr<EngineIface> make_impl(const py::dict& cfg, const std::string& estimator,
                                       const std::string& mode) {
  const auto c = config_from_dict<Shape>(cfg);
  using Chol = pl::hawkes::CholeskyEstimator<Shape>;
  using Pot = pl::hawkes::PotterEstimator<Shape>;
  const bool split = mode == "split";
  if (!split && mode != "inline") {
    throw std::invalid_argument("mode must be 'inline' or 'split'");
  }
  if (estimator == "cholesky") {
    return split ? std::unique_ptr<EngineIface>(std::make_unique<SplitImpl<Shape, Chol>>(c))
                 : std::unique_ptr<EngineIface>(std::make_unique<InlineImpl<Shape, Chol>>(c));
  }
  if (estimator == "potter") {
    return split ? std::unique_ptr<EngineIface>(std::make_unique<SplitImpl<Shape, Pot>>(c))
                 : std::unique_ptr<EngineIface>(std::make_unique<InlineImpl<Shape, Pot>>(c));
  }
  throw std::invalid_argument("estimator must be 'cholesky' or 'potter'");
}

class PyEngine {
 public:
  PyEngine(const py::dict& config, const std::string& shape, const std::string& estimator,
           const std::string& mode)
      : shape_(shape), estimator_(estimator), mode_(mode) {
    if (shape == "default") {
      impl_ = make_impl<DefaultShape>(config, estimator, mode);
    } else if (shape == "exp3") {
      impl_ = make_impl<Exp3Shape>(config, estimator, mode);
    } else {
      throw std::invalid_argument("shape must be 'default' or 'exp3'");
    }
  }

  // Runs events through the engine, writing one Signal per event into `signals` and, if
  // state_every > 0, the model state after every state_every-th model step into the state arrays.
  // The input is read in place and the GIL is released for the whole loop. Returns the number of
  // state rows written.
  std::size_t run_into(const py::array& events, py::array& signals, std::size_t state_every,
                       py::dict state_arrays, bool finish) {
    std::size_t n = 0;
    const MarketEvent* ev = as_events(events, n);
    auto* out = writable<Sig>(signals, sizeof(Sig), 1, n, "signals");
    StateBuffers b{};
    if (state_every > 0) {
      auto get = [&](const char* k) -> py::array { return py::cast<py::array>(state_arrays[k]); };
      py::array a_t = get("t_ns");
      py::array a_i = get("index");
      py::array a_th = get("theta_hat");
      py::array a_tt = get("theta_tilde");
      py::array a_g = get("G");
      py::array a_li = get("lambda_inf");
      py::array a_c = get("criticality");
      py::array a_n = get("n_eff");
      py::array a_cs = get("cusum");
      py::array a_f = get("flags");
      b.capacity = static_cast<std::size_t>(a_t.shape(0));
      const std::size_t P = impl_->P();
      b.t_ns = writable<std::int64_t>(a_t, 8, 1, b.capacity, "t_ns");
      b.index = writable<std::int64_t>(a_i, 8, 1, b.capacity, "index");
      b.theta_hat = writable<double>(a_th, 8, kD * P, b.capacity, "theta_hat");
      b.theta_tilde = writable<double>(a_tt, 8, kD * P, b.capacity, "theta_tilde");
      b.G = writable<double>(a_g, 8, kD * kD, b.capacity, "G");
      b.lambda_inf = writable<double>(a_li, 8, kD, b.capacity, "lambda_inf");
      b.criticality = writable<double>(a_c, 8, 1, b.capacity, "criticality");
      b.n_eff = writable<double>(a_n, 8, 1, b.capacity, "n_eff");
      b.cusum = writable<double>(a_cs, 8, 2 * kD, b.capacity, "cusum");
      b.flags = writable<std::uint32_t>(a_f, 4, 1, b.capacity, "flags");
    }
    std::size_t rows = 0;
    {
      py::gil_scoped_release release;
      const pl::core::FlushToZeroGuard ftz;
      std::uint64_t last_steps = impl_->steps();
      for (std::size_t i = 0; i < n; ++i) {
        impl_->on_event(ev[i]);
        std::memcpy(static_cast<void*>(out + i), &impl_->signal(), sizeof(Sig));
        if (state_every > 0) {
          const std::uint64_t s = impl_->steps();
          if (s != last_steps && s % state_every == 0 && rows < b.capacity) {
            impl_->record_state(b, rows, impl_->signal().t_ns, static_cast<std::int64_t>(i));
            ++rows;
          }
          last_steps = s;
        }
      }
      if (finish) {
        impl_->finish();  // split mode: wait until the estimator thread has consumed the batch
      }
    }
    return rows;
  }

  void set_inventory(double q) { impl_->set_inventory(q); }
  py::dict counters() const { return impl_->counters(); }
  py::dict state() const { return impl_->state(); }
  std::size_t P() const { return impl_->P(); }
  std::uint64_t steps() const { return impl_->steps(); }
  std::string repr() const {
    return "<pulselob._core.Engine shape=" + shape_ + " estimator=" + estimator_ + " mode=" + mode_ + ">";
  }

 private:
  std::unique_ptr<EngineIface> impl_;
  std::string shape_;
  std::string estimator_;
  std::string mode_;
};

// ---------------------------------------------------------------------------------------------
// Synthetic generator.
// ---------------------------------------------------------------------------------------------
py::tuple simulate_hawkes(const std::vector<double>& beta, const std::vector<double>& segment_starts,
                          const py::array_t<double, py::array::c_style | py::array::forcecast>& thetas,
                          double horizon_s, std::uint64_t seed, std::size_t max_events, bool record_lambda) {
  using Shape = DefaultShape;
  constexpr std::size_t P = Shape::P;
  if (beta.size() != Shape::Q) {
    throw std::invalid_argument("beta must have 4 rates (default kernel bank)");
  }
  if (thetas.ndim() != 3 || static_cast<std::size_t>(thetas.shape(0)) != segment_starts.size() ||
      thetas.shape(1) != static_cast<py::ssize_t>(kD) || thetas.shape(2) != static_cast<py::ssize_t>(P)) {
    throw std::invalid_argument("thetas must have shape (segments, 4, 17)");
  }
  if (segment_starts.empty() || segment_starts[0] != 0.0) {
    throw std::invalid_argument("the first segment must start at 0");
  }
  auto th = thetas.unchecked<3>();
  auto make = [&](std::size_t s) {
    pl::core::FixedMatrix<kD, P> m{};
    for (std::size_t i = 0; i < kD; ++i) {
      for (std::size_t p = 0; p < P; ++p) {
        const double x =
            th(static_cast<py::ssize_t>(s), static_cast<py::ssize_t>(i), static_cast<py::ssize_t>(p));
        if (!(x >= 0.0)) {
          throw std::invalid_argument("simulation parameters must be finite and non-negative");
        }
        m(i, p) = x;
      }
    }
    return m;
  };
  std::array<double, Shape::Q> b{};
  for (std::size_t q = 0; q < Shape::Q; ++q) {
    b[q] = beta[q];
  }
  pl::synth::HawkesSimulator<Shape> sim(b, make(0));
  for (std::size_t s = 1; s < segment_starts.size(); ++s) {
    if (!(segment_starts[s] > segment_starts[s - 1])) {
      throw std::invalid_argument("segment start times must increase");
    }
    sim.schedule_switch(segment_starts[s], make(s));
  }
  std::vector<pl::synth::ChannelEvent> ev;
  std::vector<double> lam;
  {
    py::gil_scoped_release release;
    pl::synth::Rng rng(seed);
    ev = sim.simulate(horizon_s, rng, max_events, record_lambda ? &lam : nullptr);
  }
  py::array_t<double> t(static_cast<py::ssize_t>(ev.size()));
  py::array_t<std::int32_t> ch(static_cast<py::ssize_t>(ev.size()));
  for (std::size_t k = 0; k < ev.size(); ++k) {
    t.mutable_at(static_cast<py::ssize_t>(k)) = ev[k].t_s;
    ch.mutable_at(static_cast<py::ssize_t>(k)) = ev[k].channel;
  }
  py::object lam_out = py::none();
  if (record_lambda) {
    py::array_t<double> l({static_cast<py::ssize_t>(ev.size()), static_cast<py::ssize_t>(kD)});
    if (!lam.empty()) {
      std::memcpy(l.mutable_data(), lam.data(), lam.size() * sizeof(double));
    }
    lam_out = l;
  }
  return py::make_tuple(t, ch, lam_out);
}

py::array_t<std::uint8_t> build_market_events(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& t_s,
    const py::array_t<std::int32_t, py::array::c_style | py::array::forcecast>& ch, std::uint64_t seed,
    const py::dict& o) {
  if (t_s.ndim() != 1 || ch.ndim() != 1 || t_s.shape(0) != ch.shape(0)) {
    throw std::invalid_argument("t_s and channel must be 1-D arrays of equal length");
  }
  std::vector<pl::synth::ChannelEvent> ev(static_cast<std::size_t>(t_s.shape(0)));
  for (std::size_t k = 0; k < ev.size(); ++k) {
    const int c = ch.at(static_cast<py::ssize_t>(k));
    if (c < 0 || c >= 4) {
      throw std::invalid_argument("channels must be in [0, 4)");
    }
    ev[k] = {t_s.at(static_cast<py::ssize_t>(k)), c, 1.0};
  }
  pl::synth::StreamOptions opt{};
  if (o.contains("t0_ns"))
    opt.t0_ns = py::cast<std::int64_t>(o["t0_ns"]);
  if (o.contains("tick_e8"))
    opt.tick_e8 = py::cast<std::array<std::int64_t, 2>>(o["tick_e8"]);
  if (o.contains("mid_e8"))
    opt.mid_e8 = py::cast<std::array<std::int64_t, 2>>(o["mid_e8"]);
  if (o.contains("heartbeat_ns"))
    opt.heartbeat_ns = py::cast<std::int64_t>(o["heartbeat_ns"]);
  if (o.contains("books"))
    opt.books = py::cast<bool>(o["books"]);
  if (o.contains("heartbeats"))
    opt.heartbeats = py::cast<bool>(o["heartbeats"]);
  if (o.contains("constant_marks"))
    opt.constant_marks = py::cast<bool>(o["constant_marks"]);
  if (o.contains("latency_ns"))
    opt.latency_ns = py::cast<std::array<std::int64_t, 2>>(o["latency_ns"]);
  std::vector<MarketEvent> out;
  {
    py::gil_scoped_release release;
    pl::synth::Rng rng(seed);
    out = pl::synth::to_market_events(ev, rng, opt);
  }
  return events_to_array(out);
}

// ---------------------------------------------------------------------------------------------
// Binance CSV.
// ---------------------------------------------------------------------------------------------
py::tuple parse_binance_csv(const py::bytes& data, const std::string& kind, std::uint16_t instrument,
                            std::int64_t latency_ns, bool merge_sweeps) {
  pl::feed::binance::CsvOptions opt{};
  if (kind == "aggTrades") {
    opt.kind = pl::feed::binance::CsvKind::AggTrades;
  } else if (kind == "bookTicker") {
    opt.kind = pl::feed::binance::CsvKind::BookTicker;
  } else {
    throw std::invalid_argument("kind must be 'aggTrades' or 'bookTicker'");
  }
  opt.instrument = instrument;
  opt.latency_ns = latency_ns;
  opt.merge_sweeps = merge_sweeps;
  char* buf = nullptr;
  py::ssize_t len = 0;
  if (PyBytes_AsStringAndSize(data.ptr(), &buf, &len) != 0) {
    throw py::error_already_set();
  }
  std::vector<MarketEvent> out;
  pl::feed::binance::CsvResult res{};
  {
    py::gil_scoped_release release;
    res = pl::feed::binance::parse_csv(std::string_view(buf, static_cast<std::size_t>(len)), opt, out);
  }
  if (!res.ok) {
    throw std::invalid_argument(std::string("CSV line ") + std::to_string(res.error_line) + ": " + res.error);
  }
  py::dict stats;
  stats["rows"] = res.rows;
  stats["events"] = res.events;
  stats["merged"] = res.merged;
  stats["invalid"] = res.invalid;
  stats["header"] = res.header;
  stats["resolution"] = static_cast<int>(res.resolution);
  return py::make_tuple(events_to_array(out), stats);
}

}  // namespace

PYBIND11_MODULE(_core, m) {
  m.doc() = "PulseLOB native core";
  m.attr("__version__") = pl::kVersionString;
  m.attr("MARKET_EVENT_SCHEMA_VERSION") = pl::feed::kMarketEventSchemaVersion;
  m.attr("MARKET_EVENT_ITEMSIZE") = sizeof(MarketEvent);
  m.attr("SIGNAL_ITEMSIZE") = sizeof(Sig);
  m.def("market_event_layout", &market_event_layout,
        "Field layout of the C++ MarketEvent: list of (name, offset, size, numpy type code).");
  m.def("signal_layout", &signal_layout, "Field layout of the C++ Signal<4>.");
  m.def("validate_events", &validate_events, py::arg("events"),
        "Validate a MarketEvent array without copying it; returns one EventCheck code per event.");

  py::class_<PyEngine>(m, "Engine")
      .def(py::init<const py::dict&, const std::string&, const std::string&, const std::string&>(),
           py::arg("config"), py::arg("shape") = "default", py::arg("estimator") = "cholesky",
           py::arg("mode") = "inline")
      .def("run_into", &PyEngine::run_into, py::arg("events"), py::arg("signals"), py::arg("state_every") = 0,
           py::arg("state_arrays") = py::dict(), py::arg("finish") = true)
      .def("set_inventory", &PyEngine::set_inventory, py::arg("q"))
      .def("counters", &PyEngine::counters)
      .def("state", &PyEngine::state)
      .def_property_readonly("P", &PyEngine::P)
      .def_property_readonly("steps", &PyEngine::steps)
      .def("__repr__", &PyEngine::repr);

  m.def("simulate_hawkes", &simulate_hawkes, py::arg("beta"), py::arg("segment_starts"), py::arg("thetas"),
        py::arg("horizon_s"), py::arg("seed"), py::arg("max_events") = 50'000'000,
        py::arg("record_lambda") = false,
        "Ogata thinning of the D = 4 default-bank Hawkes model with piecewise-constant parameters.");
  m.def("build_market_events", &build_market_events, py::arg("t_s"), py::arg("channel"), py::arg("seed"),
        py::arg("options") = py::dict(),
        "Convert channel events to MarketEvent bytes (two-instrument market).");
  m.def("parse_binance_csv", &parse_binance_csv, py::arg("data"), py::arg("kind"), py::arg("instrument"),
        py::arg("latency_ns") = 0, py::arg("merge_sweeps") = true,
        "Parse a data.binance.vision aggTrades or bookTicker CSV into MarketEvent bytes.");
}
