// Actuation probe: commands the measured pose plus a small offset on one joint
// and reports how far that joint moved.  THIS PUBLISHES LowCmd ON rt/lowcmd.
//
// Position hold cannot distinguish "the control board is acting on our
// commands" from "it is ignoring them", because hold commands the pose the
// robot already has and so produces almost no torque either way.  Without
// debug mode the robot accepts every message and applies nothing; this probe
// answers that in about two seconds.  The message construction - packed wire,
// per-motor mode 1 on the 23 mapped slots, mode_machine from the observed
// state, CRC identical to the SDK's crc32_core - matches the native loop.
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
using LowCmd = unitree_hg::msg::dds_::LowCmd_;
using LowState = unitree_hg::msg::dds_::LowState_;
constexpr std::array<int, 23> kSlots = {0,1,2,3,4,5,6,7,8,9,10,11,12,15,16,17,18,19,22,23,24,25,26};
std::atomic<bool> have{false};
std::atomic<uint8_t> machine{0};
std::array<float, 35> measured{};
uint32_t Crc32(const void* bytes, uint32_t words) {
  const auto* in = static_cast<const uint8_t*>(bytes);
  uint32_t crc = 0xffffffffU;
  const uint32_t poly = 0x04c11db7U;
  for (uint32_t i = 0; i < words; ++i) {
    uint32_t data = 0;
    std::memcpy(&data, in + i * 4, 4);
    uint32_t bit = 1U << 31;
    for (uint32_t b = 0; b < 32; ++b) {
      crc = (crc & 0x80000000U) ? (crc << 1) ^ poly : crc << 1;
      if (data & bit) crc ^= poly;
      bit >>= 1;
    }
  }
  return crc;
}
int main(int argc, char** argv) {
  const int slot = argc > 2 ? atoi(argv[2]) : 15;
  const float offset = argc > 3 ? atof(argv[3]) : 0.12F;
  const float kp = argc > 4 ? atof(argv[4]) : 60.0F;
  unitree::robot::ChannelFactory::Instance()->Init(0, argv[1]);
  unitree::robot::ChannelSubscriber<LowState> sub("rt/lowstate");
  sub.InitChannel([](const void* m) {
    const auto& s = *static_cast<const LowState*>(m);
    machine = s.mode_machine();
    for (int i = 0; i < 35; ++i) measured[i] = s.motor_state().at(i).q();
    have = true;
  }, 1);
  for (int i = 0; i < 300 && !have; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(10));
  if (!have) { printf("no lowstate\n"); return 1; }
  const std::array<float, 35> hold = measured;
  unitree::robot::ChannelPublisher<LowCmd> pub("rt/lowcmd");
  pub.InitChannel();
  for (int tick = 0; tick < 1200; ++tick) {
    LowCmd cmd;
    cmd.mode_pr() = 0;
    cmd.mode_machine() = machine.load();
    for (auto& m : cmd.motor_cmd()) { m.mode() = 0; m.q() = 0; m.dq() = 0; m.tau() = 0; m.kp() = 0; m.kd() = 0; }
    for (int s : kSlots) {
      auto& m = cmd.motor_cmd().at(s);
      m.mode() = 1; m.dq() = 0; m.tau() = 0; m.kp() = kp; m.kd() = 1.5F;
      m.q() = hold[s] + (s == slot ? offset : 0.0F);
    }
    cmd.crc() = Crc32(&cmd, (sizeof(LowCmd) >> 2) - 1);
    pub.Write(cmd);
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  printf("joint %d moved %+.4f rad\n", slot, measured[slot] - hold[slot]);
  return 0;
}
