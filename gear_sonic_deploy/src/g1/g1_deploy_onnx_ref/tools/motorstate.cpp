// Per-joint mode, position, velocity and measured torque from rt/lowstate.  Read-only.
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <chrono>
#include <cstdio>
#include <thread>
using LowState = unitree_hg::msg::dds_::LowState_;
int shown = 0;
void Handler(const void* message) {
  if (shown >= 2) return;
  const auto& s = *static_cast<const LowState*>(message);
  printf("mode_machine=%u\n", (unsigned)s.mode_machine());
  for (int slot : {0, 3, 4, 10, 12, 15}) {
    const auto& m = s.motor_state().at(slot);
    printf("  slot %2d: mode=%u q=%+.4f dq=%+.4f tau_est=%+.3f\n", slot, (unsigned)m.mode(), m.q(), m.dq(), m.tau_est());
  }
  ++shown;
}
int main(int argc, char** argv) {
  unitree::robot::ChannelFactory::Instance()->Init(0, argv[1]);
  unitree::robot::ChannelSubscriber<LowState> sub("rt/lowstate");
  sub.InitChannel(Handler, 1);
  std::this_thread::sleep_for(std::chrono::seconds(2));
  return 0;
}
