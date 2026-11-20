#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>

#include "pulselob/core/fixed_matrix.hpp"
#include "pulselob/core/time.hpp"
#include "pulselob/engine/config.hpp"
#include "pulselob/engine/decision.hpp"
#include "pulselob/engine/signal.hpp"
#include "pulselob/feed/event_mapper.hpp"
#include "pulselob/feed/market_event.hpp"
#include "pulselob/hawkes/estimator.hpp"
#include "pulselob/hawkes/marks.hpp"
#include "pulselob/hawkes/model.hpp"

namespace pulselob::engine {

enum class Status : std::uint8_t {
  Accepted = 0,
  Clamped = 1,             // out of order within epsilon_ooo: evaluated at the engine clock
  RejectedInvalid = 2,     // failed feed::validate; no state change
  RejectedOutOfOrder = 3,  // earlier than the clock by more than epsilon_ooo; no state change
  Ignored = 4,             // valid trade on an unmapped (instrument, side)
  RejectedNonFinite = 5,   // non-finite derived input (model step rejected); no model state change
};

// The deterministic, single-threaded engine (docs/architecture.md Sections 6.1 and 8): one call per
// event runs the full per-event algorithm of math_spec Section 9 and leaves one Signal. The object
// is fixed-size and trivially copyable, so a snapshot is a copy and restore is an assignment.
//
// Estimator selects the estimator back-end (hawkes::PotterEstimator or hawkes::CholeskyEstimator).
template <class Shape, class Estimator = hawkes::CholeskyEstimator<Shape>>
class Engine {
 public:
  static constexpr std::size_t D = Shape::D;
  using ModelType = hawkes::Model<Shape, Estimator>;
  using Layout = hawkes::BankLayout<Shape>;

  // Precondition: validate(cfg) == nullptr (checked by the caller; construction never throws).
  explicit Engine(const EngineConfig<Shape>& cfg) noexcept { init(cfg); }

  void init(const EngineConfig<Shape>& cfg) noexcept {
    cfg_ = cfg;
    model_.init(cfg.model);
    marks_.init(cfg.marks);
    mapper_.init(cfg.channels);
    decision_.init(cfg, model_.bank());
    counters_ = {};
    signal_ = {};
    clock_ = 0;
    started_ = false;
    inventory_ = 0.0;
    inventory_fault_ = false;
  }

  Status on_event(const feed::MarketEvent& e) noexcept {
    ++counters_.inputs;
    const feed::EventCheck check = feed::validate(e);
    if (check != feed::EventCheck::Ok) {
      ++counters_.invalid[static_cast<std::size_t>(check)];
      return reject(Status::RejectedInvalid, check);
    }
    core::TimestampNs t = e.ts_local_ns;
    Status status = Status::Accepted;
    if (started_ && t < clock_) {
      if (clock_ - t > cfg_.ooo_tolerance_ns) {
        ++counters_.ooo_rejected;
        return reject(Status::RejectedOutOfOrder, check);
      }
      t = clock_;
      status = Status::Clamped;
      ++counters_.ooo_clamped;
    }
    started_ = true;
    clock_ = t;
    decision_.on_input(false);
    if (e.instrument < feed::kMaxInstruments && cfg_.channels.instruments[e.instrument].present) {
      decision_.on_venue(cfg_.channels.instruments[e.instrument].venue, t);
    }
    bool fault = take_inventory_fault();
    bool stepped = false;
    switch (e.kind) {
      case feed::EventKind::Trade: {
        ++counters_.trades;
        feed::TradeFeatures f{};
        if (!mapper_.map_trade(e, f)) {
          ++counters_.unmapped;
          status = Status::Ignored;
          break;
        }
        counters_.missing_book_marks += f.book_missing ? 1U : 0U;
        // Mark statistics are updated only if the model accepts the step, so a rejection
        // leaves every model-side state unchanged.
        typename hawkes::MarkNormalizer<D>::Update up{};
        const double g = marks_.peek(static_cast<std::size_t>(f.channel), f.size, f.depth, f.iota, up);
        if (model_.step(t, f.channel, g) != ModelType::StepResult::Ok) {
          ++counters_.nonfinite_inputs;
          status = Status::RejectedNonFinite;
          fault = true;
          break;
        }
        marks_.commit(up);
        stepped = true;
        break;
      }
      case feed::EventKind::BookTop:
        ++counters_.books;
        mapper_.on_book(e, t);
        break;
      case feed::EventKind::Heartbeat:
        ++counters_.heartbeats;
        model_.step(t, -1, 0.0);
        stepped = true;
        break;
      case feed::EventKind::Status:
        ++counters_.statuses;
        ++counters_.feed_faults;
        fault = true;
        break;
    }
    emit(t, stepped, fault, status, check);
    return status;
  }

  // A heartbeat at time t (spec Section 9: steps 2-3, ridge replenishment, step 8).
  Status on_heartbeat(core::TimestampNs t) noexcept {
    feed::MarketEvent hb{};
    hb.ts_exchange_ns = t;
    hb.ts_local_ns = t;
    hb.kind = feed::EventKind::Heartbeat;
    return on_event(hb);
  }

  // Inventory used by the control law (set by the simulator or the order gateway). A non-finite
  // value is ignored, counted, and trips the kill switch at the next event.
  void set_inventory(double q) noexcept {
    if (!std::isfinite(q)) {
      ++counters_.nonfinite_inputs;
      inventory_fault_ = true;
      return;
    }
    inventory_ = q;
  }

  const Signal<D>& signal() const noexcept { return signal_; }
  const EngineCounters& counters() const noexcept { return counters_; }
  const ModelType& model() const noexcept { return model_; }
  const DecisionLayer<Shape>& decision() const noexcept { return decision_; }
  const hawkes::MarkNormalizer<D>& marks() const noexcept { return marks_; }
  const feed::EventMapper<D>& mapper() const noexcept { return mapper_; }
  const EngineConfig<Shape>& config() const noexcept { return cfg_; }
  core::TimestampNs clock() const noexcept { return clock_; }
  double inventory() const noexcept { return inventory_; }

 private:
  bool take_inventory_fault() noexcept {
    const bool f = inventory_fault_;
    inventory_fault_ = false;
    return f;
  }

  Status reject(Status s, feed::EventCheck check) noexcept {
    decision_.on_input(true);
    signal_.status =
        static_cast<std::uint16_t>(static_cast<unsigned>(s) | (static_cast<unsigned>(check) << 8));
    return s;
  }

  void emit(core::TimestampNs t, bool stepped, bool fault, Status status, feed::EventCheck check) noexcept {
    // Post-event state at t: unchanged R + pi after a step; decayed from t_prev otherwise.
    const double dt = stepped || !model_.started() ? 0.0 : core::ns_to_seconds(t - model_.t_prev());
    model_.kernel().post_event_at(model_.bank(), dt, rpost_);
    decision_.decide(t, model_.bank(), rpost_, model_.served(), mapper_, inventory_, fault, signal_);
    signal_.status =
        static_cast<std::uint16_t>(static_cast<unsigned>(status) | (static_cast<unsigned>(check) << 8));
  }

  EngineConfig<Shape> cfg_{};
  ModelType model_{};
  hawkes::MarkNormalizer<D> marks_{};
  feed::EventMapper<D> mapper_{};
  DecisionLayer<Shape> decision_{};
  core::FixedMatrix<Layout::S, D> rpost_{};
  Signal<D> signal_{};
  EngineCounters counters_{};
  core::TimestampNs clock_ = 0;
  double inventory_ = 0.0;
  bool started_ = false;
  bool inventory_fault_ = false;
};

// Replays a span of events through an engine, handing each Signal to the sink
// (docs/architecture.md Section 8). The sink's on_signal must be noexcept.
template <class Sink, std::size_t D>
concept SignalSink = requires(Sink& s, const Signal<D>& sig) {
  { s.on_signal(sig) } noexcept;
};

template <class EngineT, class Sink>
  requires SignalSink<Sink, EngineT::D>
std::size_t replay(EngineT& engine, std::span<const feed::MarketEvent> events, Sink& sink) noexcept {
  for (const feed::MarketEvent& e : events) {
    engine.on_event(e);
    sink.on_signal(engine.signal());
  }
  return events.size();
}

}  // namespace pulselob::engine
