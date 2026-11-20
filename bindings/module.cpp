// Python extension module pulselob._core.
// Phase 2 exposes the version, the MarketEvent schema, and zero-copy validation of NumPy event
// arrays; the engine itself is bound in Phase 3/4.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "pulselob/feed/market_event.hpp"
#include "pulselob/version.hpp"

namespace py = pybind11;
using pulselob::feed::MarketEvent;

static_assert(std::endian::native == std::endian::little,
              "the NumPy dtype and .plev files are little-endian");

namespace {

// (name, byte offset, byte size, NumPy type code) for every field, in memory order. The Python
// side builds its dtype independently and asserts equality with this at import time.
py::list market_event_layout() {
  py::list fields;
  const auto add = [&](const char* name, std::size_t offset, std::size_t size, const char* code) {
    fields.append(py::make_tuple(name, offset, size, code));
  };
  add("ts_exchange_ns", offsetof(MarketEvent, ts_exchange_ns), sizeof(MarketEvent::ts_exchange_ns), "<i8");
  add("ts_local_ns", offsetof(MarketEvent, ts_local_ns), sizeof(MarketEvent::ts_local_ns), "<i8");
  add("px0_e8", offsetof(MarketEvent, px0_e8), sizeof(MarketEvent::px0_e8), "<i8");
  add("qty0_e8", offsetof(MarketEvent, qty0_e8), sizeof(MarketEvent::qty0_e8), "<i8");
  add("px1_e8", offsetof(MarketEvent, px1_e8), sizeof(MarketEvent::px1_e8), "<i8");
  add("qty1_e8", offsetof(MarketEvent, qty1_e8), sizeof(MarketEvent::qty1_e8), "<i8");
  add("seq", offsetof(MarketEvent, seq), sizeof(MarketEvent::seq), "<u8");
  add("count", offsetof(MarketEvent, count), sizeof(MarketEvent::count), "<u4");
  add("instrument", offsetof(MarketEvent, instrument), sizeof(MarketEvent::instrument), "<u2");
  add("kind", offsetof(MarketEvent, kind), sizeof(MarketEvent::kind), "u1");
  add("flags", offsetof(MarketEvent, flags), sizeof(MarketEvent::flags), "u1");
  return fields;
}

// Interprets a NumPy array as MarketEvent records without copying. Requirements (the zero-copy
// contract of docs/architecture.md Section 7.9): 64-byte records, C-contiguous, 8-byte aligned,
// little-endian host.
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
  const auto addr = reinterpret_cast<std::uintptr_t>(arr.data());
  if (addr % alignof(MarketEvent) != 0) {
    throw std::invalid_argument("events buffer must be 8-byte aligned");
  }
  n = static_cast<std::size_t>(arr.shape(0));
  return static_cast<const MarketEvent*>(arr.data());
}

py::array_t<std::uint8_t> validate_events(const py::array& events) {
  std::size_t n = 0;
  const MarketEvent* ev = as_events(events, n);
  py::array_t<std::uint8_t> codes(static_cast<py::ssize_t>(n));
  std::uint8_t* out = codes.mutable_data();
  {
    py::gil_scoped_release release;
    for (std::size_t i = 0; i < n; ++i) {
      out[i] = static_cast<std::uint8_t>(pulselob::feed::validate(ev[i]));
    }
  }
  return codes;
}

}  // namespace

PYBIND11_MODULE(_core, m) {
  m.doc() = "PulseLOB native core";
  m.attr("__version__") = pulselob::kVersionString;
  m.attr("MARKET_EVENT_SCHEMA_VERSION") = pulselob::feed::kMarketEventSchemaVersion;
  m.attr("MARKET_EVENT_ITEMSIZE") = sizeof(MarketEvent);
  m.def("market_event_layout", &market_event_layout,
        "Field layout of the C++ MarketEvent: list of (name, offset, size, numpy type code).");
  m.def("validate_events", &validate_events, py::arg("events"),
        "Validate a MarketEvent array without copying it; returns one EventCheck code per event.");
}
