// Explicit G1 locomotion FSM / balance-mode command utility. Uses only the
// high-level locomotion RPC; never creates a LowCmd publisher.
//
// g1_restore_walkrun issues a fixed damp -> lock-stand -> walkrun sequence.
// Lock stand (FSM 4) holds whatever posture it is given, so it cannot lift a
// robot that is already crouched; the squat-to-stand transition is FSM 706.
// This utility exists to issue those transitions explicitly and to report the
// resulting FSM, so recovery does not depend on the operator remote.

#include <nlohmann/json.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/g1/loco/g1_loco_client.hpp>

#include <chrono>
#include <iostream>
#include <string>
#include <thread>

namespace {

constexpr char kConfirmation[] = "I_CONFIRM_G1_FSM_COMMAND";

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4 || std::string(argv[argc - 1]) != kConfirmation) {
    std::cerr << "Usage: " << argv[0]
              << " <network-interface> fsm <id> I_CONFIRM_G1_FSM_COMMAND\n"
              << "       " << argv[0]
              << " <network-interface> balance <mode> I_CONFIRM_G1_FSM_COMMAND\n";
    return 2;
  }

  const std::string interface = argv[1];
  const std::string verb = argv[2];
  const int value = std::stoi(argv[3]);

  unitree::robot::ChannelFactory::Instance()->Init(0, interface.c_str());
  unitree::robot::g1::LocoClient locomotion;
  locomotion.SetTimeout(5.0F);
  locomotion.Init();

  int before_fsm_id = -1;
  int before_fsm_mode = -1;
  locomotion.GetFsmId(before_fsm_id);
  locomotion.GetFsmMode(before_fsm_mode);

  int ret = -1;
  if (verb == "fsm") {
    ret = locomotion.SetFsmId(value);
  } else if (verb == "balance") {
    ret = locomotion.SetBalanceMode(value);
  } else {
    std::cerr << "unknown verb: " << verb << '\n';
    return 2;
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(1500));

  int after_fsm_id = -1;
  int after_fsm_mode = -1;
  locomotion.GetFsmId(after_fsm_id);
  locomotion.GetFsmMode(after_fsm_mode);

  const nlohmann::json result = {
      {"lowcmd_opened", false},
      {"verb", verb},
      {"value", value},
      {"ret", ret},
      {"before_fsm_id", before_fsm_id},
      {"before_fsm_mode", before_fsm_mode},
      {"after_fsm_id", after_fsm_id},
      {"after_fsm_mode", after_fsm_mode},
  };
  std::cout << result.dump() << '\n';
  return ret == 0 ? 0 : 1;
}
