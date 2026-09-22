// Live waist-yaw readout (hardware slot 12) while an operator straightens the
// torso by hand.  On 2026-09-20 the torso ended up twisted about 155 degrees in
// the harness after an abort, which the strict pre-flight correctly refused.
// Read-only.
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <thread>
using LowState = unitree_hg::msg::dds_::LowState_;
double latest = 0.0; bool have = false;
void Handler(const void* message) {
  const auto& s = *static_cast<const LowState*>(message);
  latest = s.motor_state().at(12).q(); have = true;
}
int main(int argc, char** argv) {
  unitree::robot::ChannelFactory::Instance()->Init(0, argv[1]);
  unitree::robot::ChannelSubscriber<LowState> sub("rt/lowstate");
  sub.InitChannel(Handler, 1);
  for (int i = 0; i < 20; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1500));
    if (!have) continue;
    printf("waist yaw %+.3f rad (%+.0f deg)%s\n", latest, latest * 180.0 / M_PI,
           std::fabs(latest) < 0.15 ? "   <-- SQUARE" : "");
    fflush(stdout);
  }
  return 0;
}
