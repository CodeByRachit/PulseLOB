#pragma once

#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <thread>

#if defined(__APPLE__)
#include <pthread.h>
#include <sys/qos.h>
#endif

#include "pulselob/core/fixed_matrix.hpp"
#include "pulselob/core/fp_env.hpp"
#include "pulselob/core/seqlock.hpp"
#include "pulselob/core/time.hpp"
#include "pulselob/engine/config.hpp"
#include "pulselob/engine/decision.hpp"
#include "pulselob/engine/engine.hpp"
#include "pulselob/engine/signal.hpp"
#include "pulselob/feed/event_mapper.hpp"
#include "pulselob/feed/market_event.hpp"
#include "pulselob/feed/spsc_ring.hpp"
#include "pulselob/hawkes/estimator.hpp"
#include "pulselob/hawkes/kernel.hpp"
#include "pulselob/hawkes/marks.hpp"
#include "pulselob/hawkes/model.hpp"

namespace pulselob::engine {

// Split mode (docs/architecture.md Section 6.1): the hot thread keeps its own copy of the kernel
// state and computes forecasts, toxicity and quotes with the most recently published parameters;
// the estimator thread runs the full model on the same channel events, received through an SPSC
// ring, and publishes the served parameters through a seqlock.
//
// The estimator thread sees exactly the inline engine's input sequence, so after drain() its
// parameters equal the inline engine's bit for bit. The hot thread's signals use parameters that
// lag the stream by the estimator's backlog, so they are not reproducible run to run; the
// kStaleParameters flag marks signals computed with parameters more than kStaleSteps steps old.
template <class Shape, class Estimator = hawkes::CholeskyEstimator<Shape>>
class SplitEngine {
 public:
  static constexpr std::size_t D = Shape::D;
  static constexpr std::size_t kRingCapacity = std::size_t{1} << 16;
  static constexpr std::uint64_t kStaleSteps = 64;
  static constexpr std::size_t kPublishEvery = 32;  // publish at least this often during a backlog
  using ModelType = hawkes::Model<Shape, Estimator>;
  using Layout = hawkes::BankLayout<Shape>;

  struct Step {
    core::TimestampNs t;
    std::int64_t channel;  // < 0: heartbeat
    double g;
  };

  // Precondition: validate(cfg) == nullptr. Allocates the ring, the seqlock and the estimator-side
  // model (cold path), then starts the estimator thread.
  explicit SplitEngine(const EngineConfig<Shape>& cfg)
      : cfg_(cfg),
        ring_(std::make_unique<feed::SpscRing<Step, kRingCapacity>>()),
        model_(std::make_unique<ModelType>(cfg.model)),
        published_(std::make_unique<core::Seqlock<hawkes::Served<Shape>>>(model_->served())) {
    bank_ = model_->bank();
    marks_.init(cfg.marks);
    mapper_.init(cfg.channels);
    decision_.init(cfg, bank_);
    served_ = model_->served();
    served_version_ = published_->version();
    thread_ = std::thread([this] { estimator_loop(); });
  }

  ~SplitEngine() { stop(); }
  SplitEngine(const SplitEngine&) = delete;
  SplitEngine& operator=(const SplitEngine&) = delete;

  void stop() noexcept {
    if (thread_.joinable()) {
      stop_.store(true, std::memory_order_release);
      thread_.join();
    }
  }

  // Hot path. Same validation, out-of-order policy and mapping as Engine::on_event.
  Status on_event(const feed::MarketEvent& e) noexcept {
    ++counters_.inputs;
    const feed::EventCheck check = feed::validate(e);
    if (check != feed::EventCheck::Ok) {
      ++counters_.invalid[static_cast<std::size_t>(check)];
      decision_.on_input(true);
      return finish_reject(Status::RejectedInvalid, check);
    }
    core::TimestampNs t = e.ts_local_ns;
    Status status = Status::Accepted;
    if (started_ && t < clock_) {
      if (clock_ - t > cfg_.ooo_tolerance_ns) {
        ++counters_.ooo_rejected;
        decision_.on_input(true);
        return finish_reject(Status::RejectedOutOfOrder, check);
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
    bool fault = false;
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
        const double g = marks_.mark(static_cast<std::size_t>(f.channel), f.size, f.depth, f.iota);
        push_step(t, f.channel, g, fault);
        stepped = true;
        break;
      }
      case feed::EventKind::BookTop:
        ++counters_.books;
        mapper_.on_book(e, t);
        break;
      case feed::EventKind::Heartbeat:
        ++counters_.heartbeats;
        push_step(t, -1, 0.0, fault);
        stepped = true;
        break;
      case feed::EventKind::Status:
        ++counters_.statuses;
        ++counters_.feed_faults;
        fault = true;
        break;
    }
    // Newest published parameters: one acquire load, a copy only when they changed.
    const std::uint64_t v = published_->version();
    if (v != served_version_) {
      served_version_ = published_->load(served_);
    }
    const double dt = stepped || !kernel_started_ ? 0.0 : core::ns_to_seconds(t - t_prev_);
    kernel_.post_event_at(bank_, dt, rpost_);
    decision_.decide(t, bank_, rpost_, served_, mapper_, inventory_, fault, signal_);
    if (steps_pushed_ > served_.steps + kStaleSteps) {
      signal_.flags |= signal_flags::kStaleParameters;
      ++stale_signals_;
    }
    signal_.status =
        static_cast<std::uint16_t>(static_cast<unsigned>(status) | (static_cast<unsigned>(check) << 8));
    return status;
  }

  void set_inventory(double q) noexcept {
    if (std::isfinite(q)) {
      inventory_ = q;
    } else {
      ++counters_.nonfinite_inputs;
    }
  }

  // Blocks (spinning) until the estimator has consumed every pushed step and published the result.
  void drain() noexcept {
    while (consumed_.load(std::memory_order_acquire) < steps_pushed_) {
      std::this_thread::yield();
    }
    for (;;) {
      served_version_ = published_->load(served_);
      if (served_.steps >= steps_pushed_) {
        return;
      }
      std::this_thread::yield();
    }
  }

  const Signal<D>& signal() const noexcept { return signal_; }
  const EngineCounters& counters() const noexcept { return counters_; }
  const hawkes::Served<Shape>& served() const noexcept { return served_; }
  // The estimator-side model; only safe to read after drain() with no concurrent on_event, or after stop().
  const ModelType& model() const noexcept { return *model_; }
  std::uint64_t steps_pushed() const noexcept { return steps_pushed_; }
  std::uint64_t stale_signals() const noexcept { return stale_signals_; }
  std::uint64_t publications() const noexcept { return publications_.load(std::memory_order_relaxed); }

 private:
  void push_step(core::TimestampNs t, int channel, double g, bool& fault) noexcept {
    // Hot copy of the kernel recursion (math_spec Section 4.1).
    if (!kernel_started_) {
      kernel_started_ = true;
      t_prev_ = t;
    }
    if (t > t_prev_) {
      kernel_.flush();
      hawkes::IntervalTerms<Shape> it{};
      it.dt = core::ns_to_seconds(t - t_prev_);
      for (std::size_t q = 0; q < Shape::Q; ++q) {
        const double bdt = bank_.beta[q] * it.dt;
        it.e[q] = std::exp(-bdt);
        it.cut[q] = bdt > bank_.x_cut;
      }
      kernel_.decay(it);
      t_prev_ = t;
    }
    if (channel >= 0) {
      kernel_.add_pending(static_cast<std::size_t>(channel), g);
    }
    if (!ring_->try_push(Step{t, channel, g})) {
      // The estimator fell 2^16 steps behind: its state is no longer the model of this stream.
      ++counters_.ring_overflows;
      fault = true;
      return;
    }
    ++steps_pushed_;
  }

  Status finish_reject(Status s, feed::EventCheck check) noexcept {
    signal_.status =
        static_cast<std::uint16_t>(static_cast<unsigned>(s) | (static_cast<unsigned>(check) << 8));
    return s;
  }

  void estimator_loop() noexcept {
#if defined(__APPLE__)
    // macOS has no hard affinity; the user-interactive class keeps the thread on performance cores.
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
    const core::FlushToZeroGuard ftz;
    std::array<Step, 256> buf{};
    std::size_t since_publish = 0;
    std::uint64_t consumed = 0;
    while (true) {
      const std::size_t n = ring_->pop_batch(std::span<Step>(buf));
      if (n == 0) {
        if (since_publish > 0) {
          published_->store(model_->served());
          publications_.fetch_add(1, std::memory_order_relaxed);
          since_publish = 0;
        }
        if (stop_.load(std::memory_order_acquire)) {
          return;
        }
        std::this_thread::yield();
        continue;
      }
      for (std::size_t k = 0; k < n; ++k) {
        model_->step(buf[k].t, static_cast<int>(buf[k].channel), buf[k].g);
        if (++since_publish >= kPublishEvery) {
          published_->store(model_->served());
          publications_.fetch_add(1, std::memory_order_relaxed);
          since_publish = 0;
        }
      }
      consumed += n;
      consumed_.store(consumed, std::memory_order_release);
    }
  }

  // ---- hot-thread state ----
  EngineConfig<Shape> cfg_;
  hawkes::KernelBank<Shape> bank_{};
  hawkes::KernelState<Shape> kernel_{};
  hawkes::MarkNormalizer<D> marks_{};
  feed::EventMapper<D> mapper_{};
  DecisionLayer<Shape> decision_{};
  hawkes::Served<Shape> served_{};
  core::FixedMatrix<Layout::S, D> rpost_{};
  Signal<D> signal_{};
  EngineCounters counters_{};
  core::TimestampNs clock_ = 0;
  core::TimestampNs t_prev_ = 0;
  std::uint64_t served_version_ = 0;
  std::uint64_t steps_pushed_ = 0;
  std::uint64_t stale_signals_ = 0;
  double inventory_ = 0.0;
  bool started_ = false;
  bool kernel_started_ = false;

  // ---- shared ----
  std::unique_ptr<feed::SpscRing<Step, kRingCapacity>> ring_;
  std::unique_ptr<ModelType> model_;  // estimator thread only (after construction)
  std::unique_ptr<core::Seqlock<hawkes::Served<Shape>>> published_;
  alignas(core::kFalseSharingBytes) std::atomic<std::uint64_t> consumed_{0};
  alignas(core::kFalseSharingBytes) std::atomic<std::uint64_t> publications_{0};
  alignas(core::kFalseSharingBytes) std::atomic<bool> stop_{false};
  std::thread thread_;
};

}  // namespace pulselob::engine
