// Standalone offline executable: generated_runtime.inc contains the actual
// current restore functions, with only sleep_for replaced by a virtual clock.
#include "true23_restore_rpc_fixture.hpp"
#include "generated_runtime.inc"

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  try {
    fixture::world = std::make_unique<fixture::World>(json::parse(argv[1]));
    ReleasedMotionMode released{.form = "0", .name = "ai",
                               .locomotion_fsm_id = 801,
                               .locomotion_fsm_mode = 0,
                               .physical_state = json::object()};
    MotionRestoreResult result;
    StateMonitor monitor;
    bool returned = false;
    std::string error;
    try {
      RestoreMotionModeAfterNormalHold(released, result, monitor);
      returned = true;
    } catch (const std::exception& failure) {
      error = failure.what();
    }
    const auto& w = *fixture::world;
    const json report = {
        {"restore_returned_success", returned}, {"error", error},
        {"elapsed_virtual_ns", w.elapsed_ns()}, {"calls", w.calls},
        {"final_mock_service", w.service}, {"final_mock_fsm_id", w.fsm_id},
        {"final_mock_fsm_mode", w.fsm_mode},
        {"physical_state", result.physical_state},
        {"check_mode_retries", result.check_mode_retries},
        {"internal_control_attempts", result.internal_control_attempts},
        {"fsm_kick_attempts", result.fsm_kick_attempts},
        {"stable_samples", result.stable_samples},
        {"select_mode_attempts", result.select_mode_attempts},
        {"dds_opened", false}, {"robot_commands_published", false},
        {"lowcmd_publisher_created", false}};
    std::cout << report.dump() << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 2;
  }
}
