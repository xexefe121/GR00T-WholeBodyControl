// Explicit G1 recovery utility. Uses only the high-level locomotion RPC;
// never creates a LowCmd publisher.

#include <nlohmann/json.hpp>
#include <unitree/robot/b2/motion_switcher/motion_switcher_client.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/g1/loco/g1_loco_client.hpp>

#include <chrono>
#include <iostream>
#include <string>
#include <thread>

namespace {

constexpr char kConfirmation[] = "I_CONFIRM_G1_WALKRUN_RESTORE";
constexpr int kRequiredStableSamples = 50;
constexpr int kMaximumPollSamples = 200;

bool IsStandingFsm(int fsm_id, int fsm_mode) {
  return (fsm_id == 500 || fsm_id == 801) && fsm_mode >= 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3 || std::string(argv[2]) != kConfirmation) {
    std::cerr << "Usage: " << argv[0]
              << " <network-interface> I_CONFIRM_G1_WALKRUN_RESTORE\n";
    return 2;
  }

  unitree::robot::ChannelFactory::Instance()->Init(0, argv[1]);
  unitree::robot::b2::MotionSwitcherClient motion;
  motion.SetTimeout(3.0F);
  motion.Init();
  unitree::robot::g1::LocoClient locomotion;
  locomotion.SetTimeout(1.0F);
  locomotion.Init();

  std::string form;
  std::string name;
  int before_fsm_id = -1;
  int before_fsm_mode = -1;
  if (motion.CheckMode(form, name) != 0 ||
      (!name.empty() && name != "ai")) {
    std::cerr << "[BLOCKED] expected empty or ai motion service\n";
    return 3;
  }
  if (name == "ai") {
    locomotion.GetFsmId(before_fsm_id);
    locomotion.GetFsmMode(before_fsm_mode);
  }

  std::cout << "[RESTORE] before service=" << name
            << " fsm_id=" << before_fsm_id
            << " fsm_mode=" << before_fsm_mode << '\n';

  int select_attempts = 0;
  int select_ret = 0;
  if (name.empty()) {
    bool service_selected = false;
    for (int attempt = 0; attempt < 60; ++attempt) {
      form.clear();
      name.clear();
      if (motion.CheckMode(form, name) == 0 && name == "ai") {
        service_selected = true;
        break;
      }
      ++select_attempts;
      select_ret = motion.SelectMode("ai");
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    if (!service_selected) {
      std::cerr << "[FAILED] ai service did not select; attempts="
                << select_attempts << " last_ret=" << select_ret << '\n';
      return 4;
    }
  }
  std::cout << "[RESTORE] ai selected after attempts=" << select_attempts
            << " last_ret=" << select_ret << '\n';

  // Match the locally proven gantry recovery path used by
  // unitree_g1_android_console: damp -> locked stand -> normal balance.
  const int damp_ret = locomotion.SetFsmId(1);
  std::cout << "[RESTORE] damp SetFsmId(1) ret=" << damp_ret << '\n';
  if (damp_ret != 0) {
    return 5;
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(400));

  const int stand_ret = locomotion.SetFsmId(4);
  std::cout << "[RESTORE] stand SetFsmId(4) ret=" << stand_ret << '\n';
  if (stand_ret != 0) {
    return 6;
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(3000));

  const int start_ret = locomotion.SetFsmId(500);
  std::cout << "[RESTORE] normal SetFsmId(500) ret=" << start_ret << '\n';
  if (start_ret != 0) {
    return 7;
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(350));

  const int balance_ret = locomotion.SetBalanceMode(0);
  std::cout << "[RESTORE] SetBalanceMode(0) ret=" << balance_ret << '\n';
  if (balance_ret != 0) {
    return 8;
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  const int reassert_ret = locomotion.SetFsmId(500);
  std::cout << "[RESTORE] reassert SetFsmId(500) ret=" << reassert_ret
            << '\n';
  if (reassert_ret != 0) {
    return 9;
  }

  int stable_samples = 0;
  int after_fsm_id = -1;
  int after_fsm_mode = -1;
  for (int sample = 0; sample < kMaximumPollSamples; ++sample) {
    const bool read_ok = locomotion.GetFsmId(after_fsm_id) == 0 &&
                         locomotion.GetFsmMode(after_fsm_mode) == 0;
    stable_samples = read_ok && IsStandingFsm(after_fsm_id, after_fsm_mode)
                         ? stable_samples + 1
                         : 0;
    if (stable_samples >= kRequiredStableSamples) {
      const nlohmann::json result = {
          {"passed", true},
          {"lowcmd_opened", false},
          {"service", name},
          {"before_fsm_id", before_fsm_id},
          {"before_fsm_mode", before_fsm_mode},
          {"after_fsm_id", after_fsm_id},
          {"after_fsm_mode", after_fsm_mode},
          {"stable_samples", stable_samples},
          {"required_stable_samples", kRequiredStableSamples},
          {"recovery_sequence", "1,4,500,balance0,500"},
          {"select_attempts", select_attempts},
          {"select_ret", select_ret},
          {"damp_ret", damp_ret},
          {"stand_ret", stand_ret},
          {"start_ret", start_ret},
          {"balance_ret", balance_ret},
          {"reassert_ret", reassert_ret},
      };
      std::cout << result.dump() << '\n';
      return 0;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  const nlohmann::json result = {
      {"passed", false},
      {"lowcmd_opened", false},
      {"service", name},
      {"before_fsm_id", before_fsm_id},
      {"before_fsm_mode", before_fsm_mode},
      {"after_fsm_id", after_fsm_id},
      {"after_fsm_mode", after_fsm_mode},
      {"stable_samples", stable_samples},
      {"required_stable_samples", kRequiredStableSamples},
      {"recovery_sequence", "1,4,500,balance0,500"},
      {"select_attempts", select_attempts},
      {"select_ret", select_ret},
      {"damp_ret", damp_ret},
      {"stand_ret", stand_ret},
      {"start_ret", start_ret},
      {"balance_ret", balance_ret},
      {"reassert_ret", reassert_ret},
  };
  std::cerr << result.dump() << '\n';
  return 10;
}
