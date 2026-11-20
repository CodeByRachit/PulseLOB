#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <span>
#include <string_view>

#include "pulselob/engine/config.hpp"
#include "pulselob/feed/market_event.hpp"

namespace pulselob::engine {

// Flat key/value form of EngineConfig: one key per field, values as numbers (booleans 0/1,
// arrays in row-major order). The Python package (pulselob.config), the native replay tool and the
// cross-check fixture all use these names, so there is a single mapping to keep correct.
//
//   text form: one "key v1 v2 ..." per line; '#' starts a comment; blank lines are ignored.
//   instruments  n  (id venue tick_e8) * n
//   channel_map  n  (instrument side(1 buy, 2 sell) channel) * n

namespace detail {

inline bool as_index(double v, std::size_t limit, std::size_t& out) noexcept {
  if (!(v >= 0.0) || v != std::floor(v) || v >= static_cast<double>(limit)) {
    return false;
  }
  out = static_cast<std::size_t>(v);
  return true;
}

inline bool as_int64(double v, std::int64_t& out) noexcept {
  if (!(std::fabs(v) < 9.0e18) || v != std::floor(v)) {
    return false;
  }
  out = static_cast<std::int64_t>(v);
  return true;
}

}  // namespace detail

// Sets one field. Returns nullptr on success or a static string describing the problem.
template <class Shape>
const char* set_config_value(EngineConfig<Shape>& c, std::string_view key,
                             std::span<const double> v) noexcept {
  constexpr std::size_t D = Shape::D;
  constexpr std::size_t P = Shape::P;
  constexpr std::size_t Q = Shape::Q;
  auto scalar = [&](double& dst) -> const char* {
    if (v.size() != 1) {
      return "expected one value";
    }
    dst = v[0];
    return nullptr;
  };
  auto boolean = [&](bool& dst) -> const char* {
    if (v.size() != 1 || (v[0] != 0.0 && v[0] != 1.0)) {
      return "expected 0 or 1";
    }
    dst = v[0] != 0.0;
    return nullptr;
  };
  auto ns = [&](std::int64_t& dst) -> const char* {
    if (v.size() != 1 || !detail::as_int64(v[0], dst)) {
      return "expected one integer";
    }
    return nullptr;
  };
  auto index = [&](std::size_t& dst, std::size_t limit) -> const char* {
    if (v.size() != 1 || !detail::as_index(v[0], limit, dst)) {
      return "expected an index in range";
    }
    return nullptr;
  };
  auto array = [&](auto& dst, std::size_t n) -> const char* {
    if (v.size() != n) {
      return "wrong number of values";
    }
    for (std::size_t i = 0; i < n; ++i) {
      dst[i] = v[i];
    }
    return nullptr;
  };
  auto& m = c.model;
  if (key == "beta") {
    return array(m.beta, Q);
  }
  if (key == "half_life_slow_s") {
    return scalar(m.half_life_slow_s);
  }
  if (key == "half_life_fast_s") {
    return scalar(m.half_life_fast_s);
  }
  if (key == "dwell_s") {
    return scalar(m.dwell_s);
  }
  if (key == "cusum_enabled") {
    return boolean(m.cusum_enabled);
  }
  if (key == "cusum_rho") {
    return scalar(m.cusum_rho);
  }
  if (key == "cusum_threshold") {
    return scalar(m.cusum_threshold);
  }
  if (key == "heartbeat_ns") {
    return ns(m.heartbeat_ns);
  }
  if (key == "rho_max") {
    return scalar(m.rho_max);
  }
  if (key == "chi") {
    return scalar(m.chi);
  }
  if (key == "eps_v") {
    return scalar(m.eps_v);
  }
  if (key == "rho_crit") {
    return scalar(m.rho_crit);
  }
  if (key == "mu_min") {
    return scalar(m.mu_min);
  }
  if (key == "mu_max") {
    return scalar(m.mu_max);
  }
  if (key == "v_ref_from_prior") {
    return boolean(m.v_ref_from_prior);
  }
  if (key == "x_cut") {
    return scalar(m.x_cut);
  }
  if (key == "pivot_eta") {
    return scalar(m.pivot_eta);
  }
  if (key == "warmup_events") {
    return scalar(m.warmup_events);
  }
  if (key == "omega") {
    return array(m.omega, P);
  }
  if (key == "theta0") {
    if (v.size() != D * P) {
      return "theta0 needs D * P values (row-major)";
    }
    for (std::size_t i = 0; i < D; ++i) {
      for (std::size_t p = 0; p < P; ++p) {
        m.theta0(i, p) = v[i * P + p];
      }
    }
    return nullptr;
  }
  if (key == "c_u") {
    return scalar(c.marks.c_u);
  }
  if (key == "c_L") {
    return scalar(c.marks.c_L);
  }
  if (key == "c_iota") {
    return scalar(c.marks.c_iota);
  }
  if (key == "u_max") {
    return scalar(c.marks.u_max);
  }
  if (key == "L_max") {
    return scalar(c.marks.L_max);
  }
  if (key == "eta_g") {
    return scalar(c.marks.eta_g);
  }
  if (key == "eta_nu") {
    return scalar(c.marks.eta_nu);
  }
  if (key == "horizon_s") {
    return scalar(c.forecast.horizon_s);
  }
  if (key == "neumann_terms") {
    return index(c.forecast.neumann_terms, 65);
  }
  if (key == "direction") {
    if (v.size() != D) {
      return "wrong number of values";
    }
    for (std::size_t i = 0; i < D; ++i) {
      if (v[i] != -1.0 && v[i] != 0.0 && v[i] != 1.0) {
        return "direction values must be -1, 0 or 1";
      }
      c.forecast.direction[i] = static_cast<int>(v[i]);
      c.channels.direction[i] = static_cast<int>(v[i]);
    }
    return nullptr;
  }
  if (key == "impact") {
    return array(c.forecast.impact, D);
  }
  auto& k = c.control;
  if (key == "gamma") {
    return scalar(k.gamma);
  }
  if (key == "k") {
    return scalar(k.k);
  }
  if (key == "horizon_inv_s") {
    return scalar(k.horizon_inv_s);
  }
  if (key == "sigma_w2") {
    return scalar(k.sigma_w2);
  }
  if (key == "delta_max") {
    return scalar(k.delta_max);
  }
  if (key == "q_max") {
    return scalar(k.q_max);
  }
  if (key == "jump_benign") {
    return array(k.jump_benign, D);
  }
  if (key == "jump_informed") {
    return array(k.jump_informed, D);
  }
  if (key == "buy_channel") {
    return index(k.buy_channel, D);
  }
  if (key == "sell_channel") {
    return index(k.sell_channel, D);
  }
  if (key == "hysteresis") {
    return scalar(k.hysteresis);
  }
  if (key == "cooldown_s") {
    return scalar(k.cooldown_s);
  }
  if (key == "requote_s") {
    return scalar(k.requote_s);
  }
  if (key == "latency_rt_s") {
    return scalar(k.latency_rt_s);
  }
  if (key == "instruments") {
    std::size_t n = 0;
    if (v.empty() || !detail::as_index(v[0], feed::kMaxInstruments + 1, n) || v.size() != 1 + 3 * n) {
      return "instruments: n followed by n (id venue tick_e8) triples";
    }
    c.channels.instruments = {};
    for (std::size_t r = 0; r < n; ++r) {
      std::size_t id = 0;
      std::size_t venue = 0;
      std::int64_t tick = 0;
      if (!detail::as_index(v[1 + 3 * r], feed::kMaxInstruments, id) ||
          !detail::as_index(v[2 + 3 * r], feed::kMaxVenues, venue) || !detail::as_int64(v[3 + 3 * r], tick)) {
        return "instruments: bad id, venue or tick";
      }
      c.channels.instruments[id] = {true, static_cast<std::uint8_t>(venue), tick};
    }
    return nullptr;
  }
  if (key == "channel_map") {
    std::size_t n = 0;
    if (v.empty() || !detail::as_index(v[0], 2 * feed::kMaxInstruments + 1, n) || v.size() != 1 + 3 * n) {
      return "channel_map: n followed by n (instrument side channel) triples";
    }
    for (auto& row : c.channels.channel) {
      row = {-1, -1};
    }
    for (std::size_t r = 0; r < n; ++r) {
      std::size_t inst = 0;
      std::size_t side = 0;
      std::size_t ch = 0;
      if (!detail::as_index(v[1 + 3 * r], feed::kMaxInstruments, inst) ||
          !detail::as_index(v[2 + 3 * r], 3, side) || side == 0 || !detail::as_index(v[3 + 3 * r], D, ch)) {
        return "channel_map: bad instrument, side (1 buy, 2 sell) or channel";
      }
      c.channels.set_channel(inst, static_cast<feed::Side>(side), static_cast<int>(ch));
    }
    return nullptr;
  }
  if (key == "quote_instrument") {
    std::size_t q = 0;
    if (const char* e = index(q, feed::kMaxInstruments)) {
      return e;
    }
    c.quote_instrument = static_cast<std::uint16_t>(q);
    return nullptr;
  }
  if (key == "ooo_tolerance_ns") {
    return ns(c.ooo_tolerance_ns);
  }
  if (key == "stale_ns") {
    return ns(c.stale_ns);
  }
  if (key == "reject_rate_eta") {
    return scalar(c.reject_rate_eta);
  }
  if (key == "reject_rate_max") {
    return scalar(c.reject_rate_max);
  }
  return "unknown configuration key";
}

// Parses the text form. On failure returns false with error and error_line set. Keys listed in
// `ignore` (for example run-time settings stored alongside a configuration) are skipped.
template <class Shape>
bool parse_config_text(std::string_view text, EngineConfig<Shape>& c, const char*& error,
                       std::size_t& error_line, std::span<const std::string_view> ignore = {}) noexcept {
  constexpr std::size_t kMaxValues = 1 + Shape::D * Shape::P + 3 * 2 * feed::kMaxInstruments;
  double values[kMaxValues];
  std::size_t pos = 0;
  std::size_t line_no = 0;
  while (pos < text.size()) {
    std::size_t end = text.find('\n', pos);
    if (end == std::string_view::npos) {
      end = text.size();
    }
    std::string_view line = text.substr(pos, end - pos);
    pos = end + 1;
    ++line_no;
    if (const std::size_t hash = line.find('#'); hash != std::string_view::npos) {
      line = line.substr(0, hash);
    }
    std::size_t i = 0;
    auto skip_ws = [&] {
      while (i < line.size() && (line[i] == ' ' || line[i] == '\t' || line[i] == '\r')) {
        ++i;
      }
    };
    skip_ws();
    if (i == line.size()) {
      continue;
    }
    const std::size_t k0 = i;
    while (i < line.size() && line[i] != ' ' && line[i] != '\t' && line[i] != '\r') {
      ++i;
    }
    const std::string_view key = line.substr(k0, i - k0);
    bool skip = false;
    for (const std::string_view ig : ignore) {
      skip = skip || ig == key;
    }
    if (skip) {
      continue;
    }
    std::size_t n = 0;
    for (;;) {
      skip_ws();
      if (i == line.size()) {
        break;
      }
      if (n == kMaxValues) {
        error = "too many values";
        error_line = line_no;
        return false;
      }
      // strtod on a bounded copy of the token (std::from_chars for double needs macOS 13.3+).
      std::size_t j = i;
      while (j < line.size() && line[j] != ' ' && line[j] != '\t' && line[j] != '\r') {
        ++j;
      }
      char token[64];
      if (j - i >= sizeof(token)) {
        error = "number too long";
        error_line = line_no;
        return false;
      }
      for (std::size_t t = i; t < j; ++t) {
        token[t - i] = line[t];
      }
      token[j - i] = '\0';
      char* parse_end = nullptr;
      values[n] = std::strtod(token, &parse_end);
      if (parse_end != token + (j - i) || j == i) {
        error = "malformed number";
        error_line = line_no;
        return false;
      }
      i = j;
      ++n;
    }
    if (const char* e = set_config_value(c, key, std::span<const double>(values, n))) {
      error = e;
      error_line = line_no;
      return false;
    }
  }
  return true;
}

}  // namespace pulselob::engine
