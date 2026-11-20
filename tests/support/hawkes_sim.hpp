#pragma once

// Test and benchmark support: aliases of the public synthetic-data component (pulselob/synth), so
// tests, benchmarks and the Python generator share one simulator.

#include "pulselob/synth/hawkes_simulator.hpp"
#include "pulselob/synth/market_stream.hpp"

namespace pulselob::testkit {

using synth::ChannelEvent;
using synth::Rng;
using synth::StreamOptions;
using synth::to_market_events;
template <class Shape>
using HawkesSimulator = synth::HawkesSimulator<Shape>;

}  // namespace pulselob::testkit
