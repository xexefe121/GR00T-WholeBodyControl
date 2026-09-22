// Pelvis tilt from upright, compared with the ladder's 20 degree abort limit.  Read-only.
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <thread>
using LowState = unitree_hg::msg::dds_::LowState_;
int shown = 0;
void Handler(const void* message) {
  if (shown >= 3) return;
  const auto& s = *static_cast<const LowState*>(message);
  const auto& q = s.imu_state().quaternion();
  const double tilt = std::acos(std::fmax(-1.0, std::fmin(1.0, 1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2]))));
  const auto& rpy = s.imu_state().rpy();
  printf("tilt=%.4f rad (%.1f deg)   limit=0.35 rad (20.0 deg)   rpy=%.3f %.3f %.3f\n",
         tilt, tilt * 180.0 / M_PI, rpy[0], rpy[1], rpy[2]);
  ++shown;
}
int main(int argc, char** argv) {
  unitree::robot::ChannelFactory::Instance()->Init(0, argv[1]);
  unitree::robot::ChannelSubscriber<LowState> sub("rt/lowstate");
  sub.InitChannel(Handler, 1);
  std::this_thread::sleep_for(std::chrono::seconds(2));
  return 0;
}
