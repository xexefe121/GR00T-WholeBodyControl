#pragma once

// Offline substitutes for the RPC transport, LowState monitor and clock only.
// The restore algorithm, constants and evidence gate are extracted from the
// current runtime source by audit_g1_true23_restore_rpc_no_robot.py.
// No Unitree SDK, DDS, socket, publisher or real-time sleep is used here.
#include "true23_active_gantry_core.hpp"

#include <chrono>
#include <iostream>
#include <memory>

namespace active = gear_sonic::true23::active;
namespace true23 = gear_sonic::true23;
using json = nlohmann::json;

namespace fixture {
struct World {
  json config;
  std::int64_t now_ns = 1'000'000'000;
  std::string service;
  int fsm_id = 0;
  int fsm_mode = 0;
  int checks = 0;
  json calls = json::array();

  explicit World(json settings) : config(std::move(settings)) {
    service = config.value("service", std::string("ai"));
    fsm_id = config.value("fsm_id", 0);
    fsm_mode = config.value("fsm_mode", 0);
  }

  std::int64_t elapsed_ns() const { return now_ns - 1'000'000'000; }
  bool disabled() const {
    const auto start = config.value("disabled_at_ns", std::int64_t(-1));
    const auto end = config.value("disabled_until_ns", INT64_MAX);
    return config.value("disabled", false) ||
           (start >= 0 && elapsed_ns() >= start && elapsed_ns() < end);
  }
  bool bit30() const { return config.value("bit30", false) || disabled(); }
  void record(const char* method, json args, int code = 0) {
    calls.push_back({{"elapsed_ns", elapsed_ns()}, {"method", method},
                     {"args", std::move(args)}, {"result", code},
                     {"service", service}, {"fsm_id", fsm_id},
                     {"fsm_mode", fsm_mode}, {"mock_disabled", disabled()},
                     {"mock_bit30", bit30()}});
  }
};
inline std::unique_ptr<World> world;

template <class Rep, class Period>
void SleepFor(std::chrono::duration<Rep, Period> duration) {
  world->now_ns +=
      std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
}
}  // namespace fixture

std::int64_t NowNs() { return fixture::world->now_ns; }

namespace unitree::robot::b2 {
class MotionSwitcherClient {
 public:
  void SetTimeout(float) {}
  void Init() {}
  int CheckMode(std::string& form, std::string& name) {
    auto& w = *fixture::world;
    const bool fail = ++w.checks <= w.config.value("check_failures", 0);
    if (!fail) {
      form = "0";
      name = w.service;
    }
    w.record("CheckMode", json::array(), fail ? 7002 : 0);
    return fail ? 7002 : 0;
  }
  int SelectMode(const std::string& name) {
    auto& w = *fixture::world;
    const int code = w.config.value("select_result", 0);
    if (code == 0 || w.config.value("select_applies_despite_error", false))
      w.service = name;
    w.record("SelectMode", {name}, code);
    return code;
  }
};
}  // namespace unitree::robot::b2

namespace unitree::robot::g1 {
enum class InternalFsmMode { WALKRUN };
class LocoClient {
 public:
  void SetTimeout(float) {}
  void Init() {}
  int SwitchToInternalCtrl(InternalFsmMode) {
    auto& w = *fixture::world;
    const int code = w.config.value("internal_result", 0);
    // Accepted RPC is not assumed to change firmware state.
    w.record("SwitchToInternalCtrl", {"WALKRUN"}, code);
    return code;
  }
  int GetFsmId(int& value) {
    auto& w = *fixture::world;
    const int code = w.config.value("fsm_read_result", 0);
    if (code == 0) value = w.fsm_id;
    w.record("GetFsmId", json::array(), code);
    return code;
  }
  int GetFsmMode(int& value) {
    auto& w = *fixture::world;
    const int code = w.config.value("fsm_read_result", 0);
    if (code == 0) value = w.fsm_mode;
    w.record("GetFsmMode", json::array(), code);
    return code;
  }
  int SetFsmId(int value) {
    auto& w = *fixture::world;
    const bool fail = value == w.config.value("set_fsm_fail_id", -1);
    const bool ignored = value == w.config.value("set_fsm_ignored_id", -1);
    if (!fail && !ignored) {
      w.fsm_id = value;
      w.fsm_mode = value == 4 && w.config.value("stand_never_completes", false)
                       ? 1 : 0;
    }
    w.record("SetFsmId", {value}, fail ? 3104 : 0);
    return fail ? 3104 : 0;
  }
};
}  // namespace unitree::robot::g1

class StateMonitor {
 public:
  std::optional<active::StateSample> LatestPhysicalState() const {
    auto& w = *fixture::world;
    w.record("LatestPhysicalState", json::array());
    if (w.config.value("missing_state", false)) return std::nullopt;
    active::StateSample state;
    state.tick = w.config.value("frozen_tick", false)
                     ? 1 : static_cast<std::uint32_t>(w.now_ns / 2'000'000);
    state.mode_machine = true23::kRequiredModeMachine;
    state.crc_valid = !w.config.value("bad_crc", false);
    state.received_monotonic_ns = w.now_ns -
        (w.config.value("stale_state", false) ? 100'000'000 : 0);
    for (const int slot : true23::kHardwareJointIds) {
      state.motor_mode[slot] = w.disabled() ? 0 : 1;
      state.motor_status[slot] = w.bit30() ? (1U << 30U) : 0;
      state.tau_est[slot] = w.config.value("zero_torque", false) ? 0.0 : 1.0;
    }
    state.q[3] = state.q[9] = w.config.value("knee_q_rad", 0.3);
    if (w.config.value("nonfinite_state", false))
      state.imu_rpy[0] = std::numeric_limits<double>::quiet_NaN();
    return state;
  }
};
