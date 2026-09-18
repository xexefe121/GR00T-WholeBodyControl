// Fix 5 native 500 Hz lowcmd loop, prepared for the G1 onboard computer.
//
// This executable is intentionally incapable of robot contact: DDS is fixed
// to domain 232, interface lo, and a test-only topic.  Replay remains the
// default source.  The optional LowState subscriber is read-only and shares
// the same loopback-only DDS factory, so it cannot create a robot-facing
// command path.

#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <zmq.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/mman.h>
#include <sched.h>
#include <time.h>
#include <vector>

namespace {

using LowCmd = unitree_hg::msg::dds_::LowCmd_;
using LowState = unitree_hg::msg::dds_::LowState_;
inline constexpr std::string_view kSafeTopic = "rt/fix5_timing_no_robot_lowcmd";
inline constexpr std::string_view kLowStateTopic = "rt/lowstate";
inline constexpr int kSafeDomain = 232;
inline constexpr std::uint32_t kReplayMagic = 0x344D4642U;  // BFM4
inline constexpr std::uint32_t kStateMagic = 0x34535442U;   // BTS4
inline constexpr std::uint32_t kTargetMagic = 0x34544742U;  // BGT4
inline constexpr std::size_t kJoints = 23;
inline constexpr std::int64_t kPeriodNs = 2'000'000;

#pragma pack(push, 1)
struct ReplayHeader {
  std::uint32_t magic;
  std::uint32_t version;
  std::uint64_t sample_count;
  double sample_period_s;
};

struct ReplaySample {
  double timestamp_s;
  std::array<float, kJoints> q;
  std::array<float, kJoints> dq;
  std::array<float, 4> quat;
  std::array<float, 3> gyro;
  std::array<float, 3> accel;
};

struct InitialCommand {
  std::uint32_t magic;
  std::uint32_t version;
  std::array<float, kJoints> q;
  std::array<float, kJoints> kp;
  std::array<float, kJoints> kd;
};

struct TargetWire {
  std::uint32_t magic;
  std::uint32_t version;
  std::uint64_t state_sequence;
  std::uint64_t policy_state_receive_mono_ns;
  std::uint64_t policy_output_mono_ns;
  std::array<float, kJoints> q;
  std::array<float, kJoints> kp;
  std::array<float, kJoints> kd;
};
#pragma pack(pop)

static_assert(std::is_trivially_copyable_v<ReplaySample>);
static_assert(std::is_trivially_copyable_v<TargetWire>);
static_assert(sizeof(ReplaySample) == 232);
static_assert(sizeof(TargetWire) == 308);

std::uint32_t Crc32(const void* bytes, std::uint32_t words) {
  const auto* input = static_cast<const std::uint8_t*>(bytes);
  std::uint32_t crc = 0xffffffffU;
  constexpr std::uint32_t polynomial = 0x04c11db7U;
  for (std::uint32_t index = 0; index < words; ++index) {
    std::uint32_t data = 0;
    std::memcpy(&data, input + index * sizeof(data), sizeof(data));
    std::uint32_t bit = 1U << 31U;
    for (std::uint32_t count = 0; count < 32U; ++count) {
      crc = (crc & 0x80000000U) ? (crc << 1U) ^ polynomial : crc << 1U;
      if (data & bit) crc ^= polynomial;
      bit >>= 1U;
    }
  }
  return crc;
}

std::int64_t MonotonicNs() {
  timespec value{};
  if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) throw std::runtime_error("clock_gettime failed");
  return static_cast<std::int64_t>(value.tv_sec) * 1'000'000'000LL + value.tv_nsec;
}

std::int64_t RealtimeNs() {
  timespec value{};
  if (clock_gettime(CLOCK_REALTIME, &value) != 0) throw std::runtime_error("clock_gettime failed");
  return static_cast<std::int64_t>(value.tv_sec) * 1'000'000'000LL + value.tv_nsec;
}

void SleepUntil(std::int64_t deadline_ns) {
  timespec deadline{deadline_ns / 1'000'000'000LL, deadline_ns % 1'000'000'000LL};
  while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &deadline, nullptr) == EINTR) {}
}

class SampleSource {
 public:
  virtual ~SampleSource() = default;
  virtual bool Next(ReplaySample& sample) = 0;
};

class ReplaySampleSource final : public SampleSource {
 public:
  explicit ReplaySampleSource(const std::string& path) : stream_(path, std::ios::binary) {
    if (!stream_.read(reinterpret_cast<char*>(&header_), sizeof(header_)) ||
        header_.magic != kReplayMagic || header_.version != 1 ||
        header_.sample_period_s != 0.002 || header_.sample_count == 0) {
      throw std::runtime_error("invalid Fix 4 flat replay");
    }
  }
  bool Next(ReplaySample& sample) override {
    return static_cast<bool>(stream_.read(reinterpret_cast<char*>(&sample), sizeof(sample)));
  }
 private:
  std::ifstream stream_;
  ReplayHeader header_{};
};

// This source is intentionally read-only.  The ChannelFactory remains fixed
// to loopback before it is created, so selecting it cannot move LowCmd off
// loopback.  It is useful for a locally bridged LowState feed and keeps the
// live-state conversion compiled and reviewable without authorising actuation.
class DdsSampleSource final : public SampleSource {
 public:
  DdsSampleSource() {
    subscriber_ = std::make_shared<unitree::robot::ChannelSubscriber<LowState>>(
        std::string(kLowStateTopic));
    subscriber_->InitChannel([this](const void* message) { OnMessage(message); }, 1);
  }
  ~DdsSampleSource() override {
    if (subscriber_) subscriber_->CloseChannel();
  }
  bool Next(ReplaySample& sample) override {
    std::lock_guard lock(mutex_);
    if (!has_sample_) return false;
    sample = latest_;
    has_sample_ = false;
    return true;
  }
 private:
  void OnMessage(const void* message) {
    if (message == nullptr) return;
    const auto& state = *static_cast<const LowState*>(message);
    ReplaySample converted{};
    converted.timestamp_s = static_cast<double>(MonotonicNs()) / 1e9;
    constexpr std::array<int, kJoints> kHardwareSlots = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19, 22, 23, 24, 25, 26};
    for (std::size_t joint = 0; joint < kJoints; ++joint) {
      converted.q[joint] = state.motor_state()[kHardwareSlots[joint]].q();
      converted.dq[joint] = state.motor_state()[kHardwareSlots[joint]].dq();
    }
    const auto& imu = state.imu_state();
    std::copy_n(imu.quaternion().begin(), converted.quat.size(), converted.quat.begin());
    std::copy_n(imu.gyroscope().begin(), converted.gyro.size(), converted.gyro.begin());
    std::copy_n(imu.accelerometer().begin(), converted.accel.size(), converted.accel.begin());
    std::lock_guard lock(mutex_);
    latest_ = converted;
    has_sample_ = true;
  }
  ReplaySample latest_{};
  bool has_sample_ = false;
  std::mutex mutex_;
  std::shared_ptr<unitree::robot::ChannelSubscriber<LowState>> subscriber_;
};

struct Arguments {
  std::string source = "replay";
  std::string replay;
  std::string initial_command;
  std::string state_endpoint;
  std::string target_endpoint;
  std::string output;
  std::uint64_t ticks = 0;
  bool lockstep = false;
};

Arguments Parse(int argc, char** argv) {
  Arguments result;
  const auto value = [&](int& index, std::string_view name) {
    if (++index == argc) throw std::runtime_error(std::string(name) + " requires a value");
    return std::string(argv[index]);
  };
  for (int index = 1; index < argc; ++index) {
    const std::string option = argv[index];
    if (option == "--help" || option == "-h") {
      std::cout << "Fix 5 native 500 Hz loop. LowCmd DDS is compiled to domain 232, lo, and a test-only topic.\n"
                   "--source <replay|dds> --replay <flat.bin> --initial-command <command.bin> "
                   "--state-endpoint <tcp://bind-address:port> --target-endpoint <tcp://bind-address:port> "
                   "--output <dir> [--ticks N] [--lockstep]\n";
      std::exit(0);
    } else if (option == "--source") result.source = value(index, option);
    else if (option == "--replay") result.replay = value(index, option);
    else if (option == "--initial-command") result.initial_command = value(index, option);
    else if (option == "--state-endpoint") result.state_endpoint = value(index, option);
    else if (option == "--target-endpoint") result.target_endpoint = value(index, option);
    else if (option == "--output") result.output = value(index, option);
    else if (option == "--ticks") result.ticks = std::stoull(value(index, option));
    else if (option == "--lockstep") result.lockstep = true;
    else throw std::runtime_error("unknown option: " + option);
  }
  if ((result.source != "replay" && result.source != "dds") ||
      (result.source == "replay" && result.replay.empty()) || result.initial_command.empty() || result.state_endpoint.empty() ||
      result.target_endpoint.empty() || result.output.empty()) throw std::runtime_error("required argument missing");
  const auto is_tcp = [](const std::string& endpoint) { return endpoint.rfind("tcp://", 0) == 0; };
  if (!is_tcp(result.state_endpoint) || !is_tcp(result.target_endpoint))
    throw std::runtime_error("Fix 5 state and target endpoints must be tcp:// bind endpoints");
  return result;
}

InitialCommand LoadInitial(const std::string& path) {
  std::ifstream stream(path, std::ios::binary);
  InitialCommand command{};
  if (!stream.read(reinterpret_cast<char*>(&command), sizeof(command)) || command.magic != kTargetMagic || command.version != 1)
    throw std::runtime_error("invalid initial command file");
  return command;
}

LowCmd MakeLowCmd(const InitialCommand& command) {
  LowCmd result;
  result.mode_pr() = 0;
  result.mode_machine() = 0;
  for (std::size_t i = 0; i < kJoints; ++i) {
    auto& motor = result.motor_cmd().at(i);
    motor.q() = command.q[i]; motor.dq() = 0.0F; motor.tau() = 0.0F;
    motor.kp() = command.kp[i]; motor.kd() = command.kd[i];
  }
  result.crc() = Crc32(&result, (sizeof(LowCmd) >> 2U) - 1U);
  return result;
}

double Percentile(std::vector<double> values, double p) {
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  const double at = (values.size() - 1) * p;
  const auto lo = static_cast<std::size_t>(at);
  const auto hi = std::min(lo + 1, values.size() - 1);
  return values[lo] + (values[hi] - values[lo]) * (at - lo);
}

void SummaryJson(std::ofstream& report, std::string_view name, const std::vector<double>& values) {
  report << "\"" << name << "\":{\"p50\":" << Percentile(values, .5)
         << ",\"p95\":" << Percentile(values, .95) << ",\"max\":"
         << (values.empty() ? 0.0 : *std::max_element(values.begin(), values.end())) << "}";
}

int Run(const Arguments& args) {
  std::filesystem::create_directories(args.output);
  InitialCommand command = LoadInitial(args.initial_command);
  sched_param fifo{}; fifo.sched_priority = 10;
  if (sched_setscheduler(0, SCHED_FIFO, &fifo) != 0) std::cerr << "SCHED_FIFO unavailable: " << std::strerror(errno) << "\n";

  unitree::robot::ChannelFactory::Instance()->Init(kSafeDomain, "lo");
  auto dds = std::make_shared<unitree::robot::ChannelPublisher<LowCmd>>(std::string(kSafeTopic));
  dds->InitChannel();
  std::unique_ptr<SampleSource> source;
  if (args.source == "replay") source = std::make_unique<ReplaySampleSource>(args.replay);
  else source = std::make_unique<DdsSampleSource>();

  void* context = zmq_ctx_new();
  if (!context) throw std::runtime_error("zmq_ctx_new failed");
  void* state_socket = zmq_socket(context, ZMQ_PUSH);
  void* target_socket = zmq_socket(context, ZMQ_PULL);
  // 65,536 fixed-size samples are under 16 MiB.  This bounded queue lets the
  // 500 Hz owner continue through an occasional policy stall without blocking
  // or discarding the recorded lowstate stream.
  const int zero = 0, hwm = 65536;
  zmq_setsockopt(state_socket, ZMQ_LINGER, &zero, sizeof(zero));
  zmq_setsockopt(target_socket, ZMQ_LINGER, &zero, sizeof(zero));
  zmq_setsockopt(state_socket, ZMQ_SNDHWM, &hwm, sizeof(hwm));
  zmq_setsockopt(target_socket, ZMQ_RCVHWM, &hwm, sizeof(hwm));
  if (zmq_bind(state_socket, args.state_endpoint.c_str()) != 0 || zmq_bind(target_socket, args.target_endpoint.c_str()) != 0)
    throw std::runtime_error("cannot bind Fix 5 TCP endpoint");

  // Lock memory only after DDS and ZMQ have created their threads. With
  // MCL_FUTURE in force every new thread stack must also be locked, and an
  // unprivileged RLIMIT_MEMLOCK then makes pthread_create fail with EAGAIN,
  // which on the robot stopped CycloneDDS from starting its receive thread.
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) std::cerr << "mlockall unavailable: " << std::strerror(errno) << "\n";

  std::ofstream ticks(args.output + "/loop_ticks.csv");
  if (!ticks) throw std::runtime_error("cannot create loop_ticks.csv");
  ticks << "tick,start_lateness_ms,work_ms,send_gap_ms,target_age_ms,ipc_round_trip_ms\n";
  std::vector<double> lateness, work, gaps, ages, round_trips;
  std::uint64_t missed = 0, dropped_state = 0, received_targets = 0;
  std::uint64_t sequence = 0, last_target_sequence = std::numeric_limits<std::uint64_t>::max();
  std::vector<std::uint64_t> state_sent_mono_ns;
  std::int64_t last_send = 0;
  const std::int64_t start = MonotonicNs();
  const auto max_ticks = args.ticks == 0 ? std::numeric_limits<std::uint64_t>::max() : args.ticks;
  for (std::uint64_t tick = 0; tick < max_ticks; ++tick) {
    const std::int64_t due = start + static_cast<std::int64_t>(tick) * kPeriodNs;
    if (!args.lockstep) SleepUntil(due);
    const std::int64_t tick_start = MonotonicNs();
    const double late_ms = std::max(0.0, static_cast<double>(tick_start - due) / 1e6);
    ReplaySample sample{};
    const bool have_sample = source->Next(sample);
    if (!have_sample && args.source == "replay") break;
    std::array<std::byte, sizeof(std::uint32_t) * 2 + sizeof(std::uint64_t) * 3 + sizeof(ReplaySample)> state{};
    std::byte* cursor = state.data();
    const std::uint32_t state_magic = kStateMagic, version = 1;
    std::memcpy(cursor, &state_magic, sizeof(state_magic)); cursor += sizeof(state_magic);
    std::memcpy(cursor, &version, sizeof(version)); cursor += sizeof(version);
    std::memcpy(cursor, &sequence, sizeof(sequence)); cursor += sizeof(sequence);
    const std::uint64_t sent_mono = static_cast<std::uint64_t>(MonotonicNs());
    std::memcpy(cursor, &sent_mono, sizeof(sent_mono)); cursor += sizeof(sent_mono);
    const std::uint64_t sent_wall = static_cast<std::uint64_t>(RealtimeNs());
    std::memcpy(cursor, &sent_wall, sizeof(sent_wall)); cursor += sizeof(sent_wall);
    std::memcpy(cursor, &sample, sizeof(sample));
    if (have_sample) {
      if (zmq_send(state_socket, state.data(), state.size(), ZMQ_DONTWAIT) < 0 && errno == EAGAIN) ++dropped_state;
      state_sent_mono_ns.push_back(sent_mono);
      ++sequence;
    }

    TargetWire incoming{};
    while (zmq_recv(target_socket, &incoming, sizeof(incoming), ZMQ_DONTWAIT) == static_cast<int>(sizeof(incoming))) {
      if (incoming.magic != kTargetMagic || incoming.version != 1) continue;
      command.q = incoming.q; command.kp = incoming.kp; command.kd = incoming.kd;
      ++received_targets;
      if (incoming.state_sequence != last_target_sequence) {
        last_target_sequence = incoming.state_sequence;
        const auto now_mono = MonotonicNs();
        if (incoming.state_sequence < state_sent_mono_ns.size()) {
          const auto sent_state = static_cast<std::int64_t>(state_sent_mono_ns[incoming.state_sequence]);
          round_trips.push_back(static_cast<double>(now_mono - sent_state) / 1e6);
          // NTP's four-timestamp offset estimate.  On loopback it converts the
          // policy monotonic output timestamp into this native clock without
          // assuming Windows and WSL realtime clocks share an epoch.
          const auto offset = ((static_cast<std::int64_t>(incoming.policy_state_receive_mono_ns) - sent_state) +
                               (static_cast<std::int64_t>(incoming.policy_output_mono_ns) - now_mono)) / 2;
          const auto output_native = static_cast<std::int64_t>(incoming.policy_output_mono_ns) - offset;
          if (output_native <= now_mono) ages.push_back(static_cast<double>(now_mono - output_native) / 1e6);
        }
      }
    }
    dds->Write(MakeLowCmd(command));
    const std::int64_t sent = MonotonicNs();
    const double work_ms = static_cast<double>(sent - tick_start) / 1e6;
    const double gap_ms = last_send == 0 ? 0.0 : static_cast<double>(sent - last_send) / 1e6;
    last_send = sent;
    lateness.push_back(late_ms); work.push_back(work_ms); if (tick) gaps.push_back(gap_ms);
    if (sent > due + kPeriodNs) ++missed;
    ticks << tick << ',' << late_ms << ',' << work_ms << ',' << gap_ms << ','
          << (ages.empty() ? 0.0 : ages.back()) << ',' << (round_trips.empty() ? 0.0 : round_trips.back()) << '\n';
  }
  std::ofstream report(args.output + "/loop_report.json");
  report << std::setprecision(12) << "{\n\"kind\":\"g1_true23_bfm_fix5_native_500hz\",\n"
         << "\"source\":\"" << args.source << "\",\n"
         << "\"dds_safety\":\"domain 232, topic rt/fix5_timing_no_robot_lowcmd, interface lo; compiled fixed\",\n"
         << "\"ticks\":" << lateness.size() << ",\"deadline_misses\":" << missed
         << ",\"state_send_eagain_drops\":" << dropped_state << ",\"targets_received\":" << received_targets << ",\n";
  SummaryJson(report, "start_lateness_ms", lateness); report << ','; SummaryJson(report, "work_ms", work); report << ',';
  SummaryJson(report, "lowcmd_gap_ms", gaps); report << ','; SummaryJson(report, "ipc_round_trip_ms", round_trips); report << ',';
  SummaryJson(report, "target_age_ms", ages); report << "\n}\n";
  zmq_close(state_socket); zmq_close(target_socket); zmq_ctx_term(context);
  std::cout << "Fix 5 native loop completed " << lateness.size() << " ticks; misses=" << missed << "\n";
  return missed == 0 ? 0 : 2;
}

}  // namespace

int main(int argc, char** argv) {
  try { return Run(Parse(argc, argv)); }
  catch (const std::exception& error) { std::cerr << "g1_true23_bfm_lowcmd_loop: " << error.what() << "\n"; return 1; }
}
