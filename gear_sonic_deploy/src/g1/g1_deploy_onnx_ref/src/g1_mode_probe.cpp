// Read-only Unitree G1 motion-service/FSM probe. Never creates LowCmd.

#include <nlohmann/json.hpp>
#include <unitree/robot/b2/motion_switcher/motion_switcher_client.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/g1/loco/g1_loco_client.hpp>

#include <iostream>
#include <string>

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <network-interface>\n";
    return 2;
  }

  unitree::robot::ChannelFactory::Instance()->Init(0, argv[1]);
  unitree::robot::b2::MotionSwitcherClient motion;
  motion.SetTimeout(3.0F);
  motion.Init();
  unitree::robot::g1::LocoClient locomotion;
  locomotion.SetTimeout(3.0F);
  locomotion.Init();

  std::string form;
  std::string name;
  int fsm_id = -1;
  int fsm_mode = -1;
  int balance_mode = -1;
  const int motion_ret = motion.CheckMode(form, name);
  const int fsm_id_ret = locomotion.GetFsmId(fsm_id);
  const int fsm_mode_ret = locomotion.GetFsmMode(fsm_mode);
  const int balance_mode_ret = locomotion.GetBalanceMode(balance_mode);

  const nlohmann::json result = {
      {"read_only", true},
      {"lowcmd_opened", false},
      {"network", argv[1]},
      {"motion_mode", {{"ret", motion_ret}, {"form", form}, {"name", name}}},
      {"locomotion",
       {{"fsm_id_ret", fsm_id_ret},
        {"fsm_id", fsm_id},
        {"fsm_mode_ret", fsm_mode_ret},
        {"fsm_mode", fsm_mode},
        {"balance_mode_ret", balance_mode_ret},
        {"balance_mode", balance_mode}}},
  };
  std::cout << result.dump() << '\n';
  return motion_ret == 0 && fsm_id_ret == 0 && fsm_mode_ret == 0
             ? 0
             : 1;
}
