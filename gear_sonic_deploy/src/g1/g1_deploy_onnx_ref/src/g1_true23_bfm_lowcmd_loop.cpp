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
#include <mujoco/mujoco.h>
#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <condition_variable>
#include <deque>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <sstream>
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
inline constexpr std::uint32_t kControlMagic = 0x34434742U; // BGC4
inline constexpr std::size_t kJoints = 23;
inline constexpr std::int64_t kPeriodNs = 2'000'000;
inline constexpr std::size_t kSolePoints = 8;
inline constexpr std::int64_t kStateMaxAgeNs = 20'000'000LL;
inline constexpr std::int64_t kTargetMaxAgeNs = 100'000'000LL;
inline constexpr std::int64_t kOperatorLivenessMaxAgeNs = 1'000'000'000LL;
inline constexpr double kDampingKd = 1.0;
inline constexpr double kHoldRampS = 3.0;
inline constexpr double kDefaultPoseRate = .20;
inline constexpr double kBrakeStep = .100;
inline constexpr double kPositionError = .35;
inline constexpr double kVelocityLimit = 6.0;
inline constexpr double kTiltLimit = .35;
inline constexpr std::int64_t kAbortDampingRampNs = 50'000'000LL;
inline constexpr std::int64_t kAbortZeroTorqueAfterNs = 250'000'000LL;

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

// The PC is only an operator input source.  Receipt time is recorded by the
// native loop; no PC clock is trusted for the deadman timeout.  `advance` is
// edge-triggered, while heartbeat and abort may be repeated safely.
struct ControlWire {
  std::uint32_t magic;
  std::uint32_t version;
  std::uint64_t sequence;
  // This is populated only by a same-host loopback verifier.  A PC connected
  // across Ethernet must leave it zero: CLOCK_MONOTONIC epochs are host-local
  // and must never be used to claim a cross-host one-way latency.
  std::uint64_t sent_monotonic_ns;
  std::uint32_t operation;  // 0 heartbeat, 1 advance, 2 manual abort
};

// This uses double intentionally. The estimator state must cross the process
// boundary without converting through the float32 LowState-shaped sample.
struct EstimatorSnapshotWire {
  std::array<double, 3> position_start;
  std::array<double, 3> velocity_start;
  std::array<double, 3> accel_bias_body;
  std::array<double, 4> quaternion_start;
  std::array<double, kSolePoints> support_weights;
  double timestamp_s;
};

struct StateWire {
  std::uint32_t magic;
  std::uint32_t version;
  std::uint64_t sequence;
  std::uint64_t sent_mono_ns;
  std::uint64_t sent_wall_ns;
  ReplaySample sample;
  EstimatorSnapshotWire estimator;
};
#pragma pack(pop)

static_assert(std::is_trivially_copyable_v<ReplaySample>);
static_assert(std::is_trivially_copyable_v<TargetWire>);
static_assert(std::is_trivially_copyable_v<ControlWire>);
static_assert(sizeof(ReplaySample) == 232);
static_assert(sizeof(TargetWire) == 308);
static_assert(sizeof(ControlWire) == 28);
static_assert(sizeof(EstimatorSnapshotWire) == 176);
static_assert(sizeof(StateWire) == 440);

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

class Native23ImuOdometry final {
 public:
  explicit Native23ImuOdometry(const std::string& model_path) {
    char error[1024]{};
    model_ = mj_loadModel(model_path.c_str(), nullptr);
    if (model_ == nullptr) throw std::runtime_error("cannot load compiled MuJoCo model: " + model_path);
    if (model_->nq != 30 || model_->nv != 29 || model_->nu != 23) throw std::runtime_error("odometry model is not native23");
    data_ = mj_makeData(model_);
    if (data_ == nullptr) throw std::runtime_error("mj_makeData failed");
    gravity_ = Eigen::Map<const Eigen::Vector3d>(model_->opt.gravity);
    const int site = mj_name2id(model_, mjOBJ_SITE, "imu_in_pelvis");
    if (site < 0) throw std::runtime_error("imu_in_pelvis site missing");
    imu_offset_ = Eigen::Map<const Eigen::Vector3d>(model_->site_pos + 3 * site);
    const mjtNum* site_quat = model_->site_quat + 4 * site;
    if (std::abs(site_quat[0] - 1.0) > 1e-12 || std::abs(site_quat[1]) > 1e-12 ||
        std::abs(site_quat[2]) > 1e-12 || std::abs(site_quat[3]) > 1e-12) {
      throw std::runtime_error("imu site rotation differs from identity");
    }
    int point_index = 0;
    for (const char* side : {"left", "right"}) {
      const std::string body_name = std::string(side) + "_ankle_roll_link";
      const int body = mj_name2id(model_, mjOBJ_BODY, body_name.c_str());
      if (body < 0) throw std::runtime_error("sole body missing: " + body_name);
      int count = 0;
      for (int geom = 0; geom < model_->ngeom; ++geom) {
        if (model_->geom_bodyid[geom] != body || model_->geom_type[geom] != mjGEOM_SPHERE || model_->geom_contype[geom] == 0) continue;
        if (point_index == static_cast<int>(kSolePoints)) throw std::runtime_error("too many sole points");
        point_bodies_[point_index] = body;
        points_[point_index] = Eigen::Map<const Eigen::Vector3d>(model_->geom_pos + 3 * geom);
        radii_[point_index] = model_->geom_size[3 * geom];
        ++point_index; ++count;
      }
      if (count != 4) throw std::runtime_error("four sole points required per foot");
    }
  }

  ~Native23ImuOdometry() {
    if (data_ != nullptr) mj_deleteData(data_);
    if (model_ != nullptr) mj_deleteModel(model_);
  }

  Native23ImuOdometry(const Native23ImuOdometry&) = delete;
  Native23ImuOdometry& operator=(const Native23ImuOdometry&) = delete;

  std::array<std::array<double, 2>, kJoints> JointLimits() const {
    std::array<std::array<double, 2>, kJoints> limits{};
    // qpos[7 + joint] is the scalar coordinate for model joint 1 + joint;
    // joint zero is the free pelvis joint.
    for (std::size_t joint = 0; joint < kJoints; ++joint) {
      limits[joint] = {model_->jnt_range[2 * (joint + 1)],
                       model_->jnt_range[2 * (joint + 1) + 1]};
    }
    return limits;
  }

  EstimatorSnapshotWire Update(const ReplaySample& sample) {
    const double timestamp = sample.timestamp_s;
    if (!std::isfinite(timestamp)) throw std::runtime_error("nonfinite odometry timestamp");
    Eigen::Vector4d quat;
    for (int i = 0; i < 4; ++i) quat[i] = sample.quat[i];
    Eigen::Vector3d omega, specific;
    for (int i = 0; i < 3; ++i) { omega[i] = sample.gyro[i]; specific[i] = sample.accel[i]; }
    const double norm = quat.norm();
    if (!std::isfinite(norm) || norm < 1e-8 || !omega.allFinite() || !specific.allFinite()) throw std::runtime_error("nonfinite odometry sample");
    quat /= norm;
    const bool first = !has_timestamp_;
    const double dt = first ? 0.0 : timestamp - timestamp_;
    if (!first && !(dt > 0.0 && dt <= .050)) throw std::runtime_error("nonmonotonic or stale odometry sample");
    if (first) {
      const Eigen::Matrix3d rotation = QuaternionMatrix(quat);
      const double yaw = std::atan2(rotation(1, 0), rotation(0, 0));
      initial_heading_ << std::cos(yaw / 2.0), 0.0, 0.0, -std::sin(yaw / 2.0);
    }
    quaternion_start_ = QuaternionMultiply(initial_heading_, quat);
    const Eigen::Matrix3d rotation = QuaternionMatrix(quaternion_start_);

    std::fill_n(data_->qpos, 3, 0.0);
    for (int i = 0; i < 4; ++i) data_->qpos[3 + i] = quaternion_start_[i];
    for (int i = 0; i < 23; ++i) data_->qpos[7 + i] = sample.q[i];
    std::fill_n(data_->qvel, 3, 0.0);
    for (int i = 0; i < 3; ++i) data_->qvel[3 + i] = omega[i];
    for (int i = 0; i < 23; ++i) data_->qvel[6 + i] = sample.dq[i];
    mj_kinematics(model_, data_); mj_comPos(model_, data_);

    Eigen::Matrix<double, kSolePoints, 3> offsets, relative_velocity;
    for (int i = 0; i < static_cast<int>(kSolePoints); ++i) {
      const int body = point_bodies_[i];
      const Eigen::Map<const Eigen::Vector3d> xpos(data_->xpos + 3 * body);
      const Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> xmat(data_->xmat + 9 * body);
      const Eigen::Vector3d offset = xpos + xmat * points_[i];
      offsets.row(i) = offset.transpose();
      mj_jac(model_, data_, jacp_.data(), jacr_.data(), offset.data(), body);
      relative_velocity.row(i) = (jacp_ * Eigen::Map<const Eigen::Matrix<double, 29, 1>>(data_->qvel)).transpose();
    }
    if (first) position_[2] = -(offsets.col(2).array() - radii_.array()).minCoeff();
    const Eigen::Vector3d previous_velocity = x_.head<3>();
    if (!first) {
      const Eigen::Vector3d alpha = (omega - previous_gyro_) / dt;
      const double mix = dt / (.020 + dt);
      filtered_alpha_ = (1.0 - mix) * filtered_alpha_ + mix * alpha;
      const Eigen::Vector3d lever = filtered_alpha_.cross(imu_offset_) + omega.cross(omega.cross(imu_offset_));
      const Eigen::Vector3d acceleration = rotation * (specific - lever - x_.tail<3>()) + gravity_;
      x_.head<3>() += acceleration * dt;
      Eigen::Matrix<double, 6, 6> transition = Eigen::Matrix<double, 6, 6>::Identity();
      transition.block<3, 3>(0, 3) = -rotation * dt;
      Eigen::Matrix<double, 6, 6> process = Eigen::Matrix<double, 6, 6>::Zero();
      for (int i = 0; i < 3; ++i) { process(i, i) = .20 * .20 * dt * dt; process(i + 3, i + 3) = .001 * .001 * dt; }
      covariance_ = transition * covariance_ * transition.transpose() + process;
    }
    const Eigen::Matrix<double, kSolePoints, 1> height = (offsets.col(2).array() - radii_.array() - (offsets.col(2).array() - radii_.array()).minCoeff()).matrix();
    Eigen::Matrix<double, kSolePoints, 1> speed;
    std::array<bool, kSolePoints> active{};
    for (int i = 0; i < static_cast<int>(kSolePoints); ++i) {
      speed[i] = (relative_velocity.row(i).transpose() + x_.head<3>()).norm();
      active[i] = height[i] <= (previous_active_[i] ? .020 : .012) && speed[i] <= (previous_active_[i] ? .40 : .30);
    }
    weights_.setZero();
    bool any_active = std::any_of(active.begin(), active.end(), [](bool value) { return value; });
    if (any_active) {
      double sum = 0.0;
      for (int i = 0; i < static_cast<int>(kSolePoints); ++i) {
        weights_[i] = active[i] ? std::exp(-std::pow(height[i] / .0075, 2) - std::pow(speed[i] / .12, 2)) : 0.0;
        sum += weights_[i];
      }
      if (sum > 1e-20) {
        weights_ /= sum;
        Eigen::Vector3d measured_velocity = Eigen::Vector3d::Zero();
        for (int i = 0; i < static_cast<int>(kSolePoints); ++i) measured_velocity += -relative_velocity.row(i).transpose() * weights_[i];
        double scatter = 0.0;
        for (int i = 0; i < static_cast<int>(kSolePoints); ++i) scatter += weights_[i] * (-relative_velocity.row(i).transpose() - measured_velocity).squaredNorm();
        const Eigen::Matrix3d observed_cov = Eigen::Matrix3d::Identity() * (.025 * .025 + scatter);
        const Eigen::Matrix<double, 3, 6> h = (Eigen::Matrix<double, 3, 6>() << Eigen::Matrix3d::Identity(), Eigen::Matrix3d::Zero()).finished();
        const Eigen::Matrix3d innovation = h * covariance_ * h.transpose() + observed_cov;
        const Eigen::Matrix<double, 6, 3> gain = innovation.ldlt().solve(h * covariance_).transpose();
        x_ += gain * (measured_velocity - x_.head<3>());
        const Eigen::Matrix<double, 6, 6> correction = Eigen::Matrix<double, 6, 6>::Identity() - gain * h;
        covariance_ = correction * covariance_ * correction.transpose() + gain * observed_cov * gain.transpose();
      } else { active.fill(false); any_active = false; }
    }
    if (!any_active) ++no_contact_updates_;
    if (!first) position_ += .5 * (previous_velocity + x_.head<3>()) * dt;
    previous_active_ = active; previous_gyro_ = omega; timestamp_ = timestamp; has_timestamp_ = true; ++updates_;
    if (!x_.allFinite() || !position_.allFinite()) throw std::runtime_error("nonfinite odometry estimate");
    EstimatorSnapshotWire result{};
    for (int i = 0; i < 3; ++i) { result.position_start[i] = position_[i]; result.velocity_start[i] = x_[i]; result.accel_bias_body[i] = x_[i + 3]; }
    for (int i = 0; i < 4; ++i) result.quaternion_start[i] = quaternion_start_[i];
    for (int i = 0; i < static_cast<int>(kSolePoints); ++i) result.support_weights[i] = weights_[i];
    result.timestamp_s = timestamp_;
    return result;
  }

 private:
  static Eigen::Vector4d QuaternionMultiply(const Eigen::Vector4d& left, const Eigen::Vector4d& right) {
    const double w1 = left[0], x1 = left[1], y1 = left[2], z1 = left[3];
    const double w2 = right[0], x2 = right[1], y2 = right[2], z2 = right[3];
    return Eigen::Vector4d(w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2, w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2);
  }
  static Eigen::Matrix3d QuaternionMatrix(Eigen::Vector4d value) {
    value /= value.norm(); const double w=value[0], x=value[1], y=value[2], z=value[3];
    Eigen::Matrix3d result;
    result << 1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y), 2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x), 2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y);
    return result;
  }
  mjModel* model_ = nullptr; mjData* data_ = nullptr;
  Eigen::Vector3d gravity_, imu_offset_, position_ = Eigen::Vector3d::Zero(), previous_gyro_ = Eigen::Vector3d::Zero(), filtered_alpha_ = Eigen::Vector3d::Zero();
  Eigen::Vector4d initial_heading_ = Eigen::Vector4d::Zero(), quaternion_start_ = Eigen::Vector4d::Zero();
  Eigen::Matrix<double, 6, 1> x_ = Eigen::Matrix<double, 6, 1>::Zero();
  Eigen::Matrix<double, 6, 6> covariance_ = (Eigen::Matrix<double, 6, 1>() << .02*.02, .02*.02, .02*.02, .10*.10, .10*.10, .10*.10).finished().asDiagonal();
  // mj_jac fills C-order (3, nv), as does the NumPy scratch array in the
  // reference implementation.  The row-major storage is therefore part of
  // equivalence, not merely an optimisation choice.
  Eigen::Matrix<double, 3, 29, Eigen::RowMajor> jacp_ = Eigen::Matrix<double, 3, 29, Eigen::RowMajor>::Zero(), jacr_ = Eigen::Matrix<double, 3, 29, Eigen::RowMajor>::Zero();
  std::array<int, kSolePoints> point_bodies_{}; std::array<Eigen::Vector3d, kSolePoints> points_{}; Eigen::Matrix<double, kSolePoints, 1> radii_ = Eigen::Matrix<double, kSolePoints, 1>::Zero(), weights_ = Eigen::Matrix<double, kSolePoints, 1>::Zero();
  std::array<bool, kSolePoints> previous_active_{}; double timestamp_ = 0.0; bool has_timestamp_ = false; std::uint64_t updates_ = 0, no_contact_updates_ = 0;
};

struct Arguments {
  std::string source = "replay";
  std::string replay;
  std::string offline_capture;
  std::string initial_command;
  std::string model;
  std::string state_endpoint;
  std::string target_endpoint;
  std::string operator_endpoint;
  std::string output;
  int dds_domain = kSafeDomain;
  std::string dds_interface = "lo";
  bool dds_domain_explicit = false;
  bool dds_interface_explicit = false;
  bool arm = false;
  bool hardware_bringup = false;
  bool bringup_ladder = false;
  std::string operator_token_file;
  std::string operator_token;
  double duration_seconds = 0.0;
  std::uint64_t ticks = 0;
  bool lockstep = false;
  bool record_estimator_snapshots = false;
  bool probe_readonly = false;
  std::uint64_t decimation = 1;
};

bool IsLoopbackInterface(const std::string& interface) {
  return interface == "lo" || interface == "lo0";
}

Arguments Parse(int argc, char** argv) {
  Arguments result;
  const auto value = [&](int& index, std::string_view name) {
    if (++index == argc) throw std::runtime_error(std::string(name) + " requires a value");
    return std::string(argv[index]);
  };
  for (int index = 1; index < argc; ++index) {
    const std::string option = argv[index];
    if (option == "--help" || option == "-h") {
      std::cout << "Fix 5 native 500 Hz loop. A LowCmd publisher exists only for domain 232 on a loopback interface.\n"
                   "--source <replay|dds> --replay <flat.bin> --initial-command <command.bin> "
                   "--model <compiled.mjb> "
                   "--state-endpoint <tcp://bind-address:port> --target-endpoint <tcp://bind-address:port> "
                   "[--bringup-ladder --operator-endpoint <tcp://bind-address:port>] "
                   "--output <dir> [--ticks N] [--lockstep] [--dds-domain N] [--dds-interface IFACE]\n"
                   "--probe-readonly --duration-seconds N --model <compiled.mjb> --output <dir> "
                   "[--dds-domain N] [--dds-interface IFACE]\n"
                   "--offline-capture <LCS1 file> --decimation N --model <compiled.mjb> --output <dir>\n";
      std::exit(0);
    } else if (option == "--source") result.source = value(index, option);
    else if (option == "--replay") result.replay = value(index, option);
    else if (option == "--offline-capture") result.offline_capture = value(index, option);
    else if (option == "--initial-command") result.initial_command = value(index, option);
    else if (option == "--model") result.model = value(index, option);
    else if (option == "--state-endpoint") result.state_endpoint = value(index, option);
    else if (option == "--target-endpoint") result.target_endpoint = value(index, option);
    else if (option == "--operator-endpoint") result.operator_endpoint = value(index, option);
    else if (option == "--output") result.output = value(index, option);
    else if (option == "--dds-domain") { result.dds_domain = std::stoi(value(index, option)); result.dds_domain_explicit = true; }
    else if (option == "--dds-interface") { result.dds_interface = value(index, option); result.dds_interface_explicit = true; }
    else if (option == "--arm") result.arm = true;
    else if (option == "--hardware-bringup") result.hardware_bringup = true;
    else if (option == "--bringup-ladder") result.bringup_ladder = true;
    else if (option == "--operator-token-file") result.operator_token_file = value(index, option);
    else if (option == "--operator-token") result.operator_token = value(index, option);
    else if (option == "--duration-seconds") result.duration_seconds = std::stod(value(index, option));
    else if (option == "--decimation") result.decimation = std::stoull(value(index, option));
    else if (option == "--ticks") result.ticks = std::stoull(value(index, option));
    else if (option == "--lockstep") result.lockstep = true;
    else if (option == "--record-estimator-snapshots") result.record_estimator_snapshots = true;
    else if (option == "--probe-readonly") result.probe_readonly = true;
    else throw std::runtime_error("unknown option: " + option);
  }
  if (result.probe_readonly) {
    if (result.model.empty() || result.output.empty() || !(result.duration_seconds > 0.0) ||
        !std::isfinite(result.duration_seconds)) {
      throw std::runtime_error("--probe-readonly requires --model, --output, and positive --duration-seconds");
    }
    return result;
  }
  if (!result.offline_capture.empty()) {
    if (result.model.empty() || result.output.empty() || result.decimation == 0) {
      throw std::runtime_error("--offline-capture requires --model, --output, and positive --decimation");
    }
    return result;
  }
  if (result.arm && !result.hardware_bringup)
    throw std::runtime_error("--arm is accepted only with --hardware-bringup; no implicit real endpoint exists");
  if (result.hardware_bringup && result.source != "dds")
    throw std::runtime_error("--hardware-bringup requires --source dds; recorded replay cannot arm a real endpoint");
  if ((result.source != "replay" && result.source != "dds") ||
      (result.source == "replay" && result.replay.empty()) || result.initial_command.empty() || result.model.empty() || result.state_endpoint.empty() ||
      result.target_endpoint.empty() || result.output.empty()) throw std::runtime_error("required argument missing");
  if ((result.hardware_bringup || result.bringup_ladder) && result.operator_endpoint.empty())
    throw std::runtime_error("native bring-up ladder requires --operator-endpoint");
  const auto is_tcp = [](const std::string& endpoint) { return endpoint.rfind("tcp://", 0) == 0; };
  if (!is_tcp(result.state_endpoint) || !is_tcp(result.target_endpoint) ||
      (!result.operator_endpoint.empty() && !is_tcp(result.operator_endpoint)))
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

enum class BringupStage : std::uint32_t {
  kDisarmed, kObserve, kZeroTorque, kDamping, kPositionHold, kDefaultPose,
  kPolicy, kAbortDamping, kAbortZeroTorque, kAborted,
};

const char* BringupStageName(BringupStage stage) {
  switch (stage) {
    case BringupStage::kDisarmed: return "disarmed";
    case BringupStage::kObserve: return "observe";
    case BringupStage::kZeroTorque: return "zero_torque";
    case BringupStage::kDamping: return "damping";
    case BringupStage::kPositionHold: return "position_hold";
    case BringupStage::kDefaultPose: return "default_pose";
    case BringupStage::kPolicy: return "policy";
    case BringupStage::kAbortDamping: return "abort_damping";
    case BringupStage::kAbortZeroTorque: return "abort_zero_torque";
    case BringupStage::kAborted: return "aborted";
  }
  return "invalid";
}

// This is the native equivalent of utils/g1_true23_bringup.py.  It owns all
// stage changes, every watchdog, and the final command calculation inside the
// 500 Hz loop.  PC messages can request advance/abort or refresh liveness,
// but cannot decide whether a command remains safe.
class NativeBringupLadder final {
 public:
  struct Transition {
    std::int64_t timestamp_ns;
    BringupStage stage;
    std::string detail;
  };

  NativeBringupLadder(const InitialCommand& contract,
                      const std::array<std::array<double, 2>, kJoints>& limits)
      : contract_(contract), limits_(limits) {
    for (std::size_t i = 0; i < kJoints; ++i) {
      default_q_[i] = contract.q[i]; operating_kp_[i] = contract.kp[i]; operating_kd_[i] = contract.kd[i];
      if (!Finite(default_q_[i]) || !Finite(operating_kp_[i]) || !Finite(operating_kd_[i]) ||
          limits_[i][0] >= limits_[i][1] || default_q_[i] < limits_[i][0] || default_q_[i] > limits_[i][1]) {
        throw std::runtime_error("invalid bring-up contract or model limits");
      }
    }
  }

  void Arm(std::int64_t now_ns) {
    if (stage_ != BringupStage::kDisarmed) throw std::runtime_error("operator must start a new process to arm again");
    Enter(BringupStage::kObserve, now_ns, "operator armed; observe only");
  }

  void Advance(const ReplaySample& state, std::int64_t now_ns) {
    if (Aborted()) return;  // A late PC packet cannot revive a latched abort.
    BringupStage next = BringupStage::kDisarmed;
    switch (stage_) {
      case BringupStage::kObserve: next = BringupStage::kZeroTorque; break;
      case BringupStage::kZeroTorque: next = BringupStage::kDamping; break;
      case BringupStage::kDamping: next = BringupStage::kPositionHold; break;
      case BringupStage::kPositionHold: next = BringupStage::kDefaultPose; break;
      case BringupStage::kDefaultPose: next = BringupStage::kPolicy; break;
      default: return;
    }
    if (next == BringupStage::kPositionHold) {
      for (std::size_t i = 0; i < kJoints; ++i) held_q_[i] = previous_q_[i] = state.q[i];
      have_previous_ = true;
    }
    Enter(next, now_ns, "fresh explicit operator advance");
  }

  void Abort(std::string_view reason, std::int64_t now_ns) {
    if (!Aborted()) {
      abort_reason_ = std::string(reason);
      abort_timestamp_ns_ = now_ns;
      Enter(BringupStage::kAbortDamping, now_ns, std::string("abort: ") + abort_reason_);
    }
  }

  InitialCommand Command(const ReplaySample& state, std::int64_t now_ns,
                         std::int64_t state_received_ns, const std::optional<TargetWire>& target,
                         std::int64_t target_output_ns, bool deadline_missed,
                         std::int64_t operator_liveness_ns) {
    CheckAbort(state, now_ns, state_received_ns, target, target_output_ns, deadline_missed, operator_liveness_ns);
    if (stage_ == BringupStage::kAbortDamping && now_ns - stage_started_ns_ >= kAbortDampingRampNs)
      Enter(BringupStage::kAbortZeroTorque, now_ns, "damping ramp complete");
    if (stage_ == BringupStage::kAbortZeroTorque && now_ns - stage_started_ns_ >= kAbortZeroTorqueAfterNs)
      Enter(BringupStage::kAborted, now_ns, "zero torque continuing; re-arm required");

    InitialCommand output{};
    output.magic = kTargetMagic; output.version = 1;
    for (std::size_t i = 0; i < kJoints; ++i) output.q[i] = have_previous_ ? static_cast<float>(previous_q_[i]) : state.q[i];
    if (stage_ == BringupStage::kDamping || stage_ == BringupStage::kAbortDamping) {
      const double alpha = stage_ == BringupStage::kAbortDamping ? std::clamp(
          static_cast<double>(now_ns - stage_started_ns_) / static_cast<double>(kAbortDampingRampNs), 0.0, 1.0) : 1.0;
      for (float& kd : output.kd) kd = static_cast<float>(kDampingKd * alpha);
    } else if (stage_ == BringupStage::kPositionHold) {
      const double alpha = std::min(static_cast<double>(now_ns - stage_started_ns_) / (kHoldRampS * 1e9), 1.0);
      for (std::size_t i = 0; i < kJoints; ++i) {
        output.q[i] = static_cast<float>(held_q_[i]); output.kp[i] = static_cast<float>(operating_kp_[i] * alpha);
        output.kd[i] = static_cast<float>(operating_kd_[i]);
      }
    } else if (stage_ == BringupStage::kDefaultPose) {
      for (std::size_t i = 0; i < kJoints; ++i) {
        const double previous = have_previous_ ? previous_q_[i] : state.q[i];
        const double next = std::clamp(default_q_[i], previous - kDefaultPoseRate * .002, previous + kDefaultPoseRate * .002);
        output.q[i] = static_cast<float>(std::clamp(next, limits_[i][0], limits_[i][1]));
        output.kp[i] = static_cast<float>(operating_kp_[i]); output.kd[i] = static_cast<float>(operating_kd_[i]);
      }
    } else if (stage_ == BringupStage::kPolicy && target) {
      for (std::size_t i = 0; i < kJoints; ++i) {
        const double previous = have_previous_ ? previous_q_[i] : state.q[i];
        const double next = std::clamp(static_cast<double>(target->q[i]), previous - kBrakeStep, previous + kBrakeStep);
        output.q[i] = static_cast<float>(std::clamp(next, limits_[i][0], limits_[i][1]));
        output.kp[i] = static_cast<float>(operating_kp_[i]); output.kd[i] = static_cast<float>(operating_kd_[i]);
      }
    }
    // Observe, zero torque, and both zero-torque abort terminal stages emit
    // the all-zero command continuously.  This is safer than withholding a
    // frame while checking a live system and is reflected in the Python spec.
    for (std::size_t i = 0; i < kJoints; ++i) {
      if (!Finite(output.q[i]) || !Finite(output.kp[i]) || !Finite(output.kd[i]) ||
          output.q[i] < limits_[i][0] || output.q[i] > limits_[i][1]) {
        Abort("non-finite value in command path or commanded joint outside model limits", now_ns);
        return Command(state, now_ns, state_received_ns, std::nullopt, 0, false, operator_liveness_ns);
      }
      previous_q_[i] = output.q[i];
    }
    have_previous_ = true;
    return output;
  }

  BringupStage stage() const { return stage_; }
  const std::string& last_event() const { return last_event_; }
  const std::vector<Transition>& transitions() const { return transitions_; }
  const std::string& abort_reason() const { return abort_reason_; }
  std::int64_t abort_timestamp_ns() const { return abort_timestamp_ns_; }

 private:
  static bool Finite(double value) { return std::isfinite(value); }
  bool Aborted() const { return stage_ == BringupStage::kAbortDamping || stage_ == BringupStage::kAbortZeroTorque || stage_ == BringupStage::kAborted; }
  void Enter(BringupStage next, std::int64_t now_ns, std::string event) {
    stage_ = next;
    stage_started_ns_ = now_ns;
    last_event_ = std::move(event);
    transitions_.push_back(Transition{now_ns, stage_, last_event_});
  }
  void CheckAbort(const ReplaySample& state, std::int64_t now_ns, std::int64_t state_received_ns,
                  const std::optional<TargetWire>& target, std::int64_t target_output_ns,
                  bool deadline_missed, std::int64_t operator_liveness_ns) {
    if (stage_ == BringupStage::kDisarmed || Aborted()) return;
    if (state_received_ns == 0 || now_ns - state_received_ns > kStateMaxAgeNs) { Abort("LowState is older than 20 ms", now_ns); return; }
    for (std::size_t i = 0; i < kJoints; ++i) {
      if (!Finite(state.q[i]) || !Finite(state.dq[i]) || state.q[i] < limits_[i][0] || state.q[i] > limits_[i][1]) { Abort("non-finite state or measured joint outside model limits", now_ns); return; }
    }
    double quat_norm_sq = 0.0;
    for (float value : state.quat) { if (!Finite(value)) { Abort("non-finite value in command path", now_ns); return; } quat_norm_sq += value * value; }
    const double tilt = std::acos(std::clamp(1.0 - 2.0 * (state.quat[1] * state.quat[1] + state.quat[2] * state.quat[2]), -1.0, 1.0));
    if (!Finite(tilt) || tilt > kTiltLimit) { Abort("estimated tilt exceeds limit", now_ns); return; }
    for (float velocity : state.dq) if (std::abs(velocity) > kVelocityLimit) { Abort("measured joint velocity exceeds limit", now_ns); return; }
    if (have_previous_) for (std::size_t i = 0; i < kJoints; ++i) if (std::abs(static_cast<double>(state.q[i]) - previous_q_[i]) > kPositionError) { Abort("measured joint position error exceeds limit", now_ns); return; }
    deadline_misses_ = deadline_missed ? deadline_misses_ + 1 : 0;
    if (deadline_misses_ > 1) { Abort("more than one consecutive deadline miss", now_ns); return; }
    if (operator_liveness_ns == 0 || now_ns - operator_liveness_ns > kOperatorLivenessMaxAgeNs) { Abort("operator liveness lost", now_ns); return; }
    if (stage_ == BringupStage::kPolicy) {
      if (!target || target_output_ns == 0 || now_ns - target_output_ns > kTargetMaxAgeNs) { Abort("policy target stale beyond 100 ms", now_ns); return; }
      for (std::size_t i = 0; i < kJoints; ++i) {
        if (!Finite(target->q[i]) || target->q[i] < limits_[i][0] || target->q[i] > limits_[i][1]) { Abort("commanded joint outside model limits", now_ns); return; }
        if (have_previous_ && std::abs(static_cast<double>(target->q[i]) - previous_q_[i]) > kBrakeStep + 1e-12) { Abort("commanded step exceeds existing brake bound", now_ns); return; }
      }
    }
  }
  InitialCommand contract_{};
  std::array<std::array<double, 2>, kJoints> limits_{};
  std::array<double, kJoints> default_q_{}, operating_kp_{}, operating_kd_{}, held_q_{}, previous_q_{};
  BringupStage stage_ = BringupStage::kDisarmed;
  std::int64_t stage_started_ns_ = 0;
  std::uint32_t deadline_misses_ = 0;
  bool have_previous_ = false;
  std::string last_event_;
  std::vector<Transition> transitions_;
  std::string abort_reason_;
  std::int64_t abort_timestamp_ns_ = 0;
};

LowCmd MakeLowCmd(const InitialCommand& command, std::uint8_t observed_mode_machine = 0) {
  LowCmd result;
  result.mode_pr() = 0;
  // HG commands must carry the state mode that was accepted at pre-flight;
  // mode 0 is retained for the old loopback-only timing path.
  result.mode_machine() = observed_mode_machine;
  // LowCmd has 35 slots while this deployment owns only 23.  Initialise every
  // unused slot explicitly as disabled/zero before filling the mapped motors;
  // never rely on an IDL constructor's default representation for a command.
  for (auto& motor : result.motor_cmd()) {
    motor.mode() = 0; motor.q() = 0.0F; motor.dq() = 0.0F; motor.tau() = 0.0F;
    motor.kp() = 0.0F; motor.kd() = 0.0F;
  }
  constexpr std::array<int, kJoints> kHardwareSlots = {
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19, 22, 23, 24, 25, 26};
  for (std::size_t i = 0; i < kJoints; ++i) {
    auto& motor = result.motor_cmd().at(kHardwareSlots[i]);
    motor.mode() = 1;
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

struct ProbeObservation {
  ReplaySample sample{};
  std::uint32_t tick = 0;
  std::uint8_t mode_machine = 0;
  std::size_t motor_slots = 0;
  // LowState has no message-time field.  The estimator deliberately uses the
  // monotonic callback time below, so retain it exactly for offline replay.
  std::array<float, 35> motor_q{}, motor_dq{}, motor_ddq{}, motor_tau_est{};
  double quaternion_norm = std::numeric_limits<double>::quiet_NaN();
  std::int64_t received_ns = 0;
  bool convertible = false;
};

#pragma pack(push, 1)
struct LowStateCaptureHeader {
  std::uint32_t magic = 0x3153434cU;  // LCS1
  std::uint32_t version = 1;
  std::uint64_t sample_count = 0;
  std::uint64_t record_bytes = 0;
};

struct LowStateCaptureRecord {
  std::int64_t received_monotonic_ns = 0;
  std::uint32_t tick = 0;
  std::uint8_t mode_machine = 0;
  std::uint8_t motor_slots = 0;
  std::array<float, 4> imu_quat{};
  std::array<float, 3> imu_gyro{}, imu_accel{};
  std::array<float, 35> motor_q{}, motor_dq{}, motor_ddq{}, motor_tau_est{};
};
#pragma pack(pop)

static_assert(std::is_trivially_copyable_v<LowStateCaptureRecord>);

// This subscriber is deliberately independent of SampleSource.  It retains
// every callback until the probe consumes it, rather than the timing loop's
// latest-sample behaviour, so the report can account for all real arrivals.
class ReadOnlyProbeSource final {
 public:
  ReadOnlyProbeSource() {
    subscriber_ = std::make_shared<unitree::robot::ChannelSubscriber<LowState>>(
        std::string(kLowStateTopic));
    subscriber_->InitChannel([this](const void* message) { OnMessage(message); }, 1);
  }
  ~ReadOnlyProbeSource() = default;
  bool WaitNext(ProbeObservation& observation, std::chrono::milliseconds timeout) {
    std::unique_lock lock(mutex_);
    if (!condition_.wait_for(lock, timeout, [this] { return !queue_.empty(); })) return false;
    observation = queue_.front();
    queue_.pop_front();
    return true;
  }
  std::uint64_t queue_drops() const {
    std::lock_guard lock(mutex_);
    return queue_drops_;
  }

 private:
  void OnMessage(const void* message) {
    if (message == nullptr) return;
    const auto& state = *static_cast<const LowState*>(message);
    ProbeObservation observation{};
    observation.received_ns = MonotonicNs();
    observation.sample.timestamp_s = static_cast<double>(observation.received_ns) / 1e9;
    observation.tick = state.tick();
    observation.mode_machine = state.mode_machine();
    observation.motor_slots = state.motor_state().size();
    const auto& quaternion = state.imu_state().quaternion();
    for (std::size_t i = 0; i < observation.sample.quat.size(); ++i) {
      observation.sample.quat[i] = quaternion[i];
    }
    observation.quaternion_norm = std::sqrt(
        std::inner_product(observation.sample.quat.begin(), observation.sample.quat.end(),
                           observation.sample.quat.begin(), 0.0));
    std::copy_n(state.imu_state().gyroscope().begin(), observation.sample.gyro.size(),
                observation.sample.gyro.begin());
    std::copy_n(state.imu_state().accelerometer().begin(), observation.sample.accel.size(),
                observation.sample.accel.begin());
    constexpr std::array<int, kJoints> kHardwareSlots = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19, 22, 23, 24, 25, 26};
    const auto required_slots = *std::max_element(kHardwareSlots.begin(), kHardwareSlots.end()) + 1;
    observation.convertible = observation.motor_slots >= static_cast<std::size_t>(required_slots);
    for (std::size_t slot = 0; slot < state.motor_state().size(); ++slot) {
      const auto& motor = state.motor_state()[slot];
      observation.motor_q[slot] = motor.q();
      observation.motor_dq[slot] = motor.dq();
      observation.motor_ddq[slot] = motor.ddq();
      observation.motor_tau_est[slot] = motor.tau_est();
    }
    if (observation.convertible) {
      for (std::size_t joint = 0; joint < kJoints; ++joint) {
        const auto& motor = state.motor_state()[kHardwareSlots[joint]];
        observation.sample.q[joint] = motor.q();
        observation.sample.dq[joint] = motor.dq();
      }
    }
    std::lock_guard lock(mutex_);
    constexpr std::size_t kMaximumQueuedSamples = 4096;
    if (queue_.size() == kMaximumQueuedSamples) {
      queue_.pop_front();
      ++queue_drops_;
    }
    queue_.push_back(observation);
    condition_.notify_one();
  }
  mutable std::mutex mutex_;
  std::condition_variable condition_;
  std::deque<ProbeObservation> queue_;
  std::uint64_t queue_drops_ = 0;
  std::shared_ptr<unitree::robot::ChannelSubscriber<LowState>> subscriber_;
};

// These checks intentionally remain independent of the publisher predicate.
// A future change must satisfy both this pre-flight and the construction/send
// guard below before it can reach the real HG topic.
std::vector<std::string> RealArmRequestFailures(const Arguments& args) {
  std::vector<std::string> failures;
  if (!args.arm) failures.emplace_back("missing --arm");
  if (!args.dds_domain_explicit) failures.emplace_back("--dds-domain was not explicit");
  if (!args.dds_interface_explicit) failures.emplace_back("--dds-interface was not explicit");
  if (IsLoopbackInterface(args.dds_interface) || args.dds_domain == kSafeDomain)
    failures.emplace_back("real arming requires a non-loopback DDS endpoint");
  if (args.operator_token_file.empty()) failures.emplace_back("missing --operator-token-file");
  if (args.operator_token.empty()) failures.emplace_back("missing --operator-token");
  if (!args.operator_token_file.empty() && !args.operator_token.empty()) {
    try {
      std::ifstream token_file(args.operator_token_file);
      std::string token; std::getline(token_file, token);
      const auto age = std::chrono::duration_cast<std::chrono::seconds>(
          std::filesystem::file_time_type::clock::now() - std::filesystem::last_write_time(args.operator_token_file)).count();
      if (!token_file || age < 0 || age > 60) failures.emplace_back("operator token file is not fresh (maximum age 60 s)");
      else if (token != args.operator_token) failures.emplace_back("operator token does not match token file");
    } catch (const std::exception&) { failures.emplace_back("operator token file cannot be read"); }
  }
  return failures;
}

std::optional<std::uint8_t> LiveBringupPreflight(const Arguments& args, const Native23ImuOdometry& estimator,
                                                  std::vector<std::string>& failures) {
  ReadOnlyProbeSource source;
  ProbeObservation observation{};
  if (!source.WaitNext(observation, std::chrono::milliseconds(20))) {
    failures.emplace_back("no live LowState within 20 ms"); return std::nullopt;
  }
  if (!observation.convertible) failures.emplace_back("LowState does not contain the required 23-joint layout");
  if (observation.mode_machine != 4) failures.emplace_back("LowState mode_machine is not recognised (expected 4)");
  if (!std::isfinite(observation.quaternion_norm) || std::abs(observation.quaternion_norm - 1.0) > .01)
    failures.emplace_back("LowState IMU quaternion is non-finite or not unit length within 0.01");
  const auto limits = estimator.JointLimits();
  for (std::size_t joint = 0; joint < kJoints; ++joint) {
    if (!std::isfinite(observation.sample.q[joint]) || !std::isfinite(observation.sample.dq[joint]) ||
        observation.sample.q[joint] < limits[joint][0] || observation.sample.q[joint] > limits[joint][1]) {
      failures.emplace_back("LowState mapped joint is non-finite or outside model limits"); break;
    }
  }
  if (MonotonicNs() - observation.received_ns > 20'000'000LL)
    failures.emplace_back("LowState is older than 20 ms");
  return failures.empty() ? std::optional<std::uint8_t>(observation.mode_machine) : std::nullopt;
}

void WriteProbeSummary(std::ofstream& report, std::string_view name,
                       const std::vector<double>& values) {
  report << "\"" << name << "\":{\"count\":" << values.size()
         << ",\"min\":" << (values.empty() ? 0.0 : *std::min_element(values.begin(), values.end()))
         << ",\"p50\":" << Percentile(values, .50)
         << ",\"p95\":" << Percentile(values, .95)
         << ",\"p99\":" << Percentile(values, .99)
         << ",\"max\":" << (values.empty() ? 0.0 : *std::max_element(values.begin(), values.end()))
         << "}";
}

int RunReadOnlyProbe(const Arguments& args) {
  std::filesystem::create_directories(args.output);
  std::cout << "DDS endpoint: domain=" << args.dds_domain << " interface=" << args.dds_interface
            << "; mode=read-only probe; LowCmd publisher exists=false\n";
  unitree::robot::ChannelFactory::Instance()->Init(args.dds_domain, args.dds_interface.c_str());
  Native23ImuOdometry estimator(args.model);
  const auto limits = estimator.JointLimits();
  ReadOnlyProbeSource source;
  std::fstream capture(args.output + "/real_lowstate_capture.lcs", std::ios::binary | std::ios::out | std::ios::trunc);
  if (!capture) throw std::runtime_error("cannot create real_lowstate_capture.lcs");
  LowStateCaptureHeader capture_header{};
  capture_header.record_bytes = sizeof(LowStateCaptureRecord);
  capture.write(reinterpret_cast<const char*>(&capture_header), sizeof(capture_header));
  std::ofstream trace(args.output + "/real_lowstate_estimator_trace.csv");
  if (!trace) throw std::runtime_error("cannot create real_lowstate_estimator_trace.csv");
  trace << std::setprecision(17)
        << "sample,elapsed_s,pelvis_height_m,horizontal_position_m,speed_mps,bias_x,bias_y,bias_z"
        << ",weight_0,weight_1,weight_2,weight_3,weight_4,weight_5,weight_6,weight_7\n";

  std::vector<double> intervals_ms, work_ms, quaternion_deviation, height_m, speed_mps, bias_mps2;
  std::array<std::vector<double>, kSolePoints> weights;
  std::array<double, kJoints> joint_min, joint_max;
  joint_min.fill(std::numeric_limits<double>::infinity());
  joint_max.fill(-std::numeric_limits<double>::infinity());
  std::map<unsigned int, std::uint64_t> modes;
  std::uint64_t received = 0, converted = 0, invalid_layout = 0, gaps_over_3ms = 0, gaps_over_10ms = 0;
  std::size_t min_slots = std::numeric_limits<std::size_t>::max(), max_slots = 0;
  std::int64_t first_received_ns = 0, previous_received_ns = 0, last_received_ns = 0;
  Eigen::Vector2d first_horizontal = Eigen::Vector2d::Zero(), last_horizontal = Eigen::Vector2d::Zero();
  const std::int64_t acquisition_started_ns = MonotonicNs();
  const std::int64_t stop_ns = acquisition_started_ns + static_cast<std::int64_t>(args.duration_seconds * 1e9);
  while (MonotonicNs() < stop_ns) {
    ProbeObservation observation;
    if (!source.WaitNext(observation, std::chrono::milliseconds(100))) continue;
    ++received;
    ++modes[observation.mode_machine];
    min_slots = std::min(min_slots, observation.motor_slots);
    max_slots = std::max(max_slots, observation.motor_slots);
    if (std::isfinite(observation.quaternion_norm)) quaternion_deviation.push_back(std::abs(observation.quaternion_norm - 1.0));
    if (previous_received_ns != 0) {
      const double interval = static_cast<double>(observation.received_ns - previous_received_ns) / 1e6;
      intervals_ms.push_back(interval);
      if (interval > 3.0) ++gaps_over_3ms;
      if (interval > 10.0) ++gaps_over_10ms;
    }
    previous_received_ns = observation.received_ns;
    last_received_ns = observation.received_ns;
    if (first_received_ns == 0) first_received_ns = observation.received_ns;
    LowStateCaptureRecord capture_record{};
    capture_record.received_monotonic_ns = observation.received_ns;
    capture_record.tick = observation.tick;
    capture_record.mode_machine = observation.mode_machine;
    capture_record.motor_slots = static_cast<std::uint8_t>(observation.motor_slots);
    capture_record.imu_quat = observation.sample.quat;
    capture_record.imu_gyro = observation.sample.gyro;
    capture_record.imu_accel = observation.sample.accel;
    capture_record.motor_q = observation.motor_q;
    capture_record.motor_dq = observation.motor_dq;
    capture_record.motor_ddq = observation.motor_ddq;
    capture_record.motor_tau_est = observation.motor_tau_est;
    capture.write(reinterpret_cast<const char*>(&capture_record), sizeof(capture_record));
    if (!capture) throw std::runtime_error("failed while writing real_lowstate_capture.lcs");
    ++capture_header.sample_count;
    if (!observation.convertible) { ++invalid_layout; continue; }
    for (std::size_t joint = 0; joint < kJoints; ++joint) {
      joint_min[joint] = std::min(joint_min[joint], static_cast<double>(observation.sample.q[joint]));
      joint_max[joint] = std::max(joint_max[joint], static_cast<double>(observation.sample.q[joint]));
    }
    const auto work_start = MonotonicNs();
    const EstimatorSnapshotWire snapshot = estimator.Update(observation.sample);
    work_ms.push_back(static_cast<double>(MonotonicNs() - work_start) / 1e6);
    ++converted;
    const Eigen::Vector2d horizontal(snapshot.position_start[0], snapshot.position_start[1]);
    if (converted == 1) first_horizontal = horizontal;
    last_horizontal = horizontal;
    height_m.push_back(snapshot.position_start[2]);
    speed_mps.push_back(Eigen::Vector3d(snapshot.velocity_start[0], snapshot.velocity_start[1], snapshot.velocity_start[2]).norm());
    bias_mps2.push_back(Eigen::Vector3d(snapshot.accel_bias_body[0], snapshot.accel_bias_body[1], snapshot.accel_bias_body[2]).norm());
    for (std::size_t point = 0; point < kSolePoints; ++point) weights[point].push_back(snapshot.support_weights[point]);
    if (converted == 1 || converted % 50 == 0) {
      trace << converted << ',' << static_cast<double>(observation.received_ns - first_received_ns) / 1e9
            << ',' << snapshot.position_start[2] << ',' << horizontal.norm() << ',' << speed_mps.back();
      for (double value : snapshot.accel_bias_body) trace << ',' << value;
      for (double value : snapshot.support_weights) trace << ',' << value;
      trace << '\n';
    }
  }
  const double observed_seconds = first_received_ns == 0 || last_received_ns <= first_received_ns ? 0.0 :
      static_cast<double>(last_received_ns - first_received_ns) / 1e9;
  capture.seekp(0);
  capture.write(reinterpret_cast<const char*>(&capture_header), sizeof(capture_header));
  capture.close();
  const double horizontal_drift = converted < 2 ? 0.0 : (last_horizontal - first_horizontal).norm();
  std::ofstream report(args.output + "/real_lowstate_probe_report.json");
  if (!report) throw std::runtime_error("cannot create real_lowstate_probe_report.json");
  report << std::setprecision(12) << "{\n\"kind\":\"g1_true23_real_lowstate_read_only\",\n"
         << "\"dds_domain\":" << args.dds_domain << ",\"dds_interface\":\"" << args.dds_interface << "\",\n"
         << "\"lowcmd_publisher_exists\":false,\"lowcmd_messages_sent\":0,\n"
         << "\"capture_file\":\"real_lowstate_capture.lcs\",\"capture_format\":\"LCS1: callback_monotonic_ns,tick,IMU,q/dq/ddq/tau_est for all 35 motor slots\",\n"
         << "\"lowstate_message_timestamp_available\":false,\"estimator_timestamp_source\":\"callback CLOCK_MONOTONIC\",\n"
         << "\"requested_duration_s\":" << args.duration_seconds << ",\"observed_duration_s\":" << observed_seconds
         << ",\"samples_received\":" << received << ",\"samples_converted\":" << converted
         << ",\"invalid_joint_layout_samples\":" << invalid_layout << ",\"probe_queue_drops\":" << source.queue_drops()
         << ",\"measured_rate_hz\":" << (observed_seconds > 0.0 ? static_cast<double>(received - 1) / observed_seconds : 0.0) << ",\n";
  WriteProbeSummary(report, "inter_sample_interval_ms", intervals_ms);
  report << ",\"gaps_over_3ms\":" << gaps_over_3ms << ",\"gaps_over_10ms\":" << gaps_over_10ms << ",\n";
  report << "\"decode_sanity\":{\"motor_slots_min\":" << (received ? min_slots : 0) << ",\"motor_slots_max\":" << max_slots << ",\"mode_machine_counts\":{";
  bool first_mode = true;
  for (const auto& [mode, count] : modes) { report << (first_mode ? "" : ",") << "\"" << mode << "\":" << count; first_mode = false; }
  report << "},"; WriteProbeSummary(report, "imu_quaternion_norm_deviation", quaternion_deviation); report << "},\n";
  report << "\"joint_position_ranges_rad\":[";
  for (std::size_t joint = 0; joint < kJoints; ++joint) {
    const double excess = received == invalid_layout ? 0.0 : std::max({0.0, limits[joint][0] - joint_min[joint], joint_max[joint] - limits[joint][1]});
    report << (joint ? "," : "") << "{\"joint\":" << joint << ",\"min\":" << joint_min[joint] << ",\"max\":" << joint_max[joint]
           << ",\"model_lower\":" << limits[joint][0] << ",\"model_upper\":" << limits[joint][1] << ",\"max_excess\":" << excess << "}";
  }
  report << "],\n\"estimator\":{\"horizontal_drift_m\":" << horizontal_drift << ",\"horizontal_drift_m_per_min\":"
         << (observed_seconds > 0.0 ? horizontal_drift * 60.0 / observed_seconds : 0.0) << ",";
  WriteProbeSummary(report, "pelvis_height_above_initial_sole_floor_m", height_m); report << ',';
  WriteProbeSummary(report, "speed_magnitude_mps", speed_mps); report << ',';
  WriteProbeSummary(report, "accelerometer_bias_magnitude_mps2", bias_mps2); report << ",\"support_weights\":{";
  for (std::size_t point = 0; point < kSolePoints; ++point) { if (point) report << ','; WriteProbeSummary(report, "point_" + std::to_string(point), weights[point]); }
  report << "}},\n\"timing_comparison\":{";
  WriteProbeSummary(report, "estimator_update_work_ms", work_ms);
  report << ",\"start_lateness_ms\":null,\"lowcmd_gap_ms\":null,\"ipc_round_trip_ms\":null,\"target_age_ms\":null"
         << "}\n}\n";
  std::cout << "Read-only probe completed: received=" << received << " converted=" << converted
            << " rate_hz=" << (observed_seconds > 0.0 ? static_cast<double>(received - 1) / observed_seconds : 0.0)
            << " publisher_exists=false\n";
  return received > 0 && converted == received && source.queue_drops() == 0 ? 0 : 2;
}

// Pure file replay for rate measurements.  This mode does not initialise DDS,
// create ZeroMQ sockets, construct a publisher, or otherwise touch the robot.
int RunOfflineCaptureProbe(const Arguments& args) {
  std::filesystem::create_directories(args.output);
  std::ifstream capture(args.offline_capture, std::ios::binary);
  LowStateCaptureHeader header{};
  if (!capture.read(reinterpret_cast<char*>(&header), sizeof(header)) || header.magic != 0x3153434cU ||
      header.version != 1 || header.record_bytes != sizeof(LowStateCaptureRecord) || header.sample_count < 2) {
    throw std::runtime_error("invalid LCS1 offline capture");
  }
  std::cout << "Offline LCS1 rate probe; DDS disabled; LowCmd publisher exists=false\n";
  Native23ImuOdometry estimator(args.model);
  constexpr std::array<int, kJoints> kHardwareSlots = {
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19, 22, 23, 24, 25, 26};
  std::vector<double> work_ms, speed_mps, bias_mps2;
  std::array<std::vector<double>, kSolePoints> weights;
  Eigen::Vector2d first_horizontal = Eigen::Vector2d::Zero(), last_horizontal = Eigen::Vector2d::Zero();
  std::int64_t first_timestamp_ns = 0, last_timestamp_ns = 0;
  std::uint64_t updates = 0;
  for (std::uint64_t index = 0; index < header.sample_count; ++index) {
    LowStateCaptureRecord record{};
    if (!capture.read(reinterpret_cast<char*>(&record), sizeof(record))) throw std::runtime_error("truncated LCS1 offline capture");
    if (index % args.decimation) continue;
    if (record.motor_slots < 27) throw std::runtime_error("LCS1 record has incomplete motor layout");
    ReplaySample sample{};
    sample.timestamp_s = static_cast<double>(record.received_monotonic_ns) / 1e9;
    sample.quat = record.imu_quat; sample.gyro = record.imu_gyro; sample.accel = record.imu_accel;
    for (std::size_t joint = 0; joint < kJoints; ++joint) {
      sample.q[joint] = record.motor_q[kHardwareSlots[joint]];
      sample.dq[joint] = record.motor_dq[kHardwareSlots[joint]];
    }
    const auto begin_ns = MonotonicNs();
    const EstimatorSnapshotWire snapshot = estimator.Update(sample);
    work_ms.push_back(static_cast<double>(MonotonicNs() - begin_ns) / 1e6);
    const Eigen::Vector2d horizontal(snapshot.position_start[0], snapshot.position_start[1]);
    if (updates == 0) { first_horizontal = horizontal; first_timestamp_ns = record.received_monotonic_ns; }
    last_horizontal = horizontal; last_timestamp_ns = record.received_monotonic_ns;
    speed_mps.push_back(Eigen::Vector3d(snapshot.velocity_start[0], snapshot.velocity_start[1], snapshot.velocity_start[2]).norm());
    bias_mps2.push_back(Eigen::Vector3d(snapshot.accel_bias_body[0], snapshot.accel_bias_body[1], snapshot.accel_bias_body[2]).norm());
    for (std::size_t point = 0; point < kSolePoints; ++point) weights[point].push_back(snapshot.support_weights[point]);
    ++updates;
  }
  const double seconds = static_cast<double>(last_timestamp_ns - first_timestamp_ns) / 1e9;
  const double drift = (last_horizontal - first_horizontal).norm();
  std::ofstream report(args.output + "/offline_capture_rate_report.json");
  if (!report) throw std::runtime_error("cannot write offline_capture_rate_report.json");
  report << std::setprecision(12) << "{\n\"kind\":\"g1_true23_offline_lcs1_rate_probe\",\n"
         << "\"capture\":\"" << args.offline_capture << "\",\"decimation\":" << args.decimation
         << ",\"updates\":" << updates << ",\"elapsed_s\":" << seconds
         << ",\"effective_rate_hz\":" << (seconds > 0.0 ? static_cast<double>(updates - 1) / seconds : 0.0)
         << ",\"lowcmd_publisher_exists\":false,\"lowcmd_messages_sent\":0,\n"
         << "\"horizontal_drift_m\":" << drift << ",\"horizontal_drift_m_per_min\":" << (seconds > 0.0 ? drift * 60.0 / seconds : 0.0) << ",\n";
  WriteProbeSummary(report, "speed_magnitude_mps", speed_mps); report << ',';
  WriteProbeSummary(report, "accelerometer_bias_magnitude_mps2", bias_mps2); report << ',';
  WriteProbeSummary(report, "estimator_update_work_ms", work_ms); report << ",\"support_weights\":{";
  for (std::size_t point = 0; point < kSolePoints; ++point) { if (point) report << ','; WriteProbeSummary(report, "point_" + std::to_string(point), weights[point]); }
  report << "}\n}\n";
  std::cout << "Offline rate probe completed: decimation=" << args.decimation << " updates=" << updates
            << " drift_m_per_min=" << (seconds > 0.0 ? drift * 60.0 / seconds : 0.0)
            << " publisher_exists=false\n";
  return 0;
}

void WriteSafeLoopbackLowCmd(
    const Arguments& args,
    const std::shared_ptr<unitree::robot::ChannelPublisher<LowCmd>>& publisher,
    const InitialCommand& command, bool real_endpoint_armed, std::uint8_t observed_mode_machine) {
  // The predicate is intentionally repeated at the send site.  A null
  // publisher is not the safety mechanism; this explicit domain-and-interface
  // check is, and it remains true even if a future edit changes ownership.
  if (((args.dds_domain == kSafeDomain && IsLoopbackInterface(args.dds_interface)) ||
       (args.hardware_bringup && args.arm && args.dds_domain_explicit && args.dds_interface_explicit &&
        !IsLoopbackInterface(args.dds_interface) && real_endpoint_armed)) && publisher) {
    publisher->Write(MakeLowCmd(command, observed_mode_machine));
  }
}

int Run(const Arguments& args) {
  if (args.probe_readonly) return RunReadOnlyProbe(args);
  if (!args.offline_capture.empty()) return RunOfflineCaptureProbe(args);
  std::filesystem::create_directories(args.output);
  InitialCommand command = LoadInitial(args.initial_command);
  Native23ImuOdometry estimator(args.model);
  const bool ladder_enabled = args.hardware_bringup || args.bringup_ladder;
  std::optional<NativeBringupLadder> ladder;
  if (ladder_enabled) ladder.emplace(command, estimator.JointLimits());
  sched_param fifo{}; fifo.sched_priority = 10;
  if (sched_setscheduler(0, SCHED_FIFO, &fifo) != 0) std::cerr << "SCHED_FIFO unavailable: " << std::strerror(errno) << "\n";

  unitree::robot::ChannelFactory::Instance()->Init(args.dds_domain, args.dds_interface.c_str());
  std::shared_ptr<unitree::robot::ChannelPublisher<LowCmd>> dds;
  bool real_endpoint_armed = false;
  std::uint8_t observed_mode_machine = 0;
  const bool real_endpoint_request = args.hardware_bringup &&
      !(args.dds_domain == kSafeDomain && IsLoopbackInterface(args.dds_interface));
  if (real_endpoint_request) {
    std::vector<std::string> failures = RealArmRequestFailures(args);
    const auto preflight_mode = LiveBringupPreflight(args, estimator, failures);
    if (!failures.empty() || !preflight_mode) {
      std::ostringstream refusal;
      refusal << "real endpoint arming refused:";
      for (const auto& failure : failures) refusal << " " << failure << ";";
      throw std::runtime_error(refusal.str());
    }
    real_endpoint_armed = true;
    observed_mode_machine = *preflight_mode;
  }
  // LowCmd publisher construction is confined to the one harmless endpoint.
  // All other DDS endpoints are subscriber-only as far as this executable is
  // concerned: no publisher object exists, so it has no command path.
  if ((args.dds_domain == kSafeDomain && IsLoopbackInterface(args.dds_interface)) || real_endpoint_armed) {
    dds = std::make_shared<unitree::robot::ChannelPublisher<LowCmd>>(
        real_endpoint_armed ? "rt/lowcmd" : std::string(kSafeTopic));
    dds->InitChannel();
  }
  std::cout << "DDS endpoint: domain=" << args.dds_domain << " interface=" << args.dds_interface
            << "; mode=" << (dds ? "timing loop with loopback test publisher" : "subscriber-only")
            << "; LowCmd publisher exists=" << (dds ? "true" : "false") << "\n";
  std::unique_ptr<SampleSource> source;
  if (args.source == "replay") source = std::make_unique<ReplaySampleSource>(args.replay);
  else source = std::make_unique<DdsSampleSource>();

  void* context = zmq_ctx_new();
  if (!context) throw std::runtime_error("zmq_ctx_new failed");
  void* state_socket = zmq_socket(context, ZMQ_PUSH);
  void* target_socket = zmq_socket(context, ZMQ_PULL);
  void* operator_socket = ladder_enabled ? zmq_socket(context, ZMQ_PULL) : nullptr;
  // 65,536 fixed-size samples are under 16 MiB.  This bounded queue lets the
  // 500 Hz owner continue through an occasional policy stall without blocking
  // or discarding the recorded lowstate stream.
  const int zero = 0, hwm = 65536;
  zmq_setsockopt(state_socket, ZMQ_LINGER, &zero, sizeof(zero));
  zmq_setsockopt(target_socket, ZMQ_LINGER, &zero, sizeof(zero));
  if (operator_socket) zmq_setsockopt(operator_socket, ZMQ_LINGER, &zero, sizeof(zero));
  zmq_setsockopt(state_socket, ZMQ_SNDHWM, &hwm, sizeof(hwm));
  zmq_setsockopt(target_socket, ZMQ_RCVHWM, &hwm, sizeof(hwm));
  if (operator_socket) zmq_setsockopt(operator_socket, ZMQ_RCVHWM, &hwm, sizeof(hwm));
  if (zmq_bind(state_socket, args.state_endpoint.c_str()) != 0 || zmq_bind(target_socket, args.target_endpoint.c_str()) != 0)
    throw std::runtime_error("cannot bind Fix 5 TCP endpoint");
  if (operator_socket && zmq_bind(operator_socket, args.operator_endpoint.c_str()) != 0)
    throw std::runtime_error("cannot bind native operator-control endpoint");

  // Lock memory only after DDS and ZMQ have created their threads. With
  // MCL_FUTURE in force every new thread stack must also be locked, and an
  // unprivileged RLIMIT_MEMLOCK then makes pthread_create fail with EAGAIN,
  // which on the robot stopped CycloneDDS from starting its receive thread.
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) std::cerr << "mlockall unavailable: " << std::strerror(errno) << "\n";

  std::ofstream ticks(args.output + "/loop_ticks.csv");
  if (!ticks) throw std::runtime_error("cannot create loop_ticks.csv");
  ticks << "tick,start_lateness_ms,work_ms,send_gap_ms,target_age_ms,ipc_round_trip_ms\n";
  std::ofstream snapshots;
  if (args.record_estimator_snapshots) {
    snapshots.open(args.output + "/estimator_snapshots.csv");
    if (!snapshots) throw std::runtime_error("cannot create estimator_snapshots.csv");
    snapshots << std::setprecision(17) << "tick,position_x,position_y,position_z,velocity_x,velocity_y,velocity_z,bias_x,bias_y,bias_z,quat_w,quat_x,quat_y,quat_z,weight_0,weight_1,weight_2,weight_3,weight_4,weight_5,weight_6,weight_7,timestamp_s\n";
  }
  std::vector<double> lateness, work, gaps, ages, round_trips, manual_abort_delivery_latencies;
  std::uint64_t missed = 0, dropped_state = 0, received_targets = 0, received_operator_controls = 0;
  std::uint64_t observed_operator_frames = 0;
  int last_operator_frame_bytes = -1;
  std::uint32_t last_operator_frame_magic = 0, last_operator_frame_version = 0;
  std::uint32_t last_operator_frame_first_bytes = 0;
  std::uint64_t sequence = 0, last_target_sequence = std::numeric_limits<std::uint64_t>::max();
  std::vector<std::uint64_t> state_sent_mono_ns;
  std::int64_t last_send = 0;
  std::int64_t last_state_received_ns = 0, last_target_output_ns = 0, last_operator_liveness_ns = 0;
  std::optional<TargetWire> latest_target;
  ReplaySample latest_state{};
  bool have_latest_state = false;
  const std::int64_t start = MonotonicNs();
  const auto max_ticks = args.ticks == 0 ? std::numeric_limits<std::uint64_t>::max() : args.ticks;
  for (std::uint64_t tick = 0; tick < max_ticks; ++tick) {
    const std::int64_t due = start + static_cast<std::int64_t>(tick) * kPeriodNs;
    if (!args.lockstep) SleepUntil(due);
    const std::int64_t tick_start = MonotonicNs();
    const bool deadline_missed = tick_start > due + kPeriodNs;
    const double late_ms = std::max(0.0, static_cast<double>(tick_start - due) / 1e6);
    ReplaySample sample{};
    const bool have_sample = source->Next(sample);
    if (!have_sample && args.source == "replay") break;
    EstimatorSnapshotWire snapshot{};
    if (have_sample) { snapshot = estimator.Update(sample); latest_state = sample; have_latest_state = true; last_state_received_ns = tick_start; }
    StateWire state{};
    state.magic = kStateMagic; state.version = 2; state.sequence = sequence;
    state.sent_mono_ns = static_cast<std::uint64_t>(MonotonicNs());
    state.sent_wall_ns = static_cast<std::uint64_t>(RealtimeNs());
    state.sample = sample; state.estimator = snapshot;
    if (have_sample) {
      if (zmq_send(state_socket, &state, sizeof(state), ZMQ_DONTWAIT) < 0 && errno == EAGAIN) ++dropped_state;
      state_sent_mono_ns.push_back(state.sent_mono_ns);
      ++sequence;
    }
    if (args.record_estimator_snapshots && have_sample && tick % 10 == 0) {
      snapshots << tick;
      for (double value : state.estimator.position_start) snapshots << ',' << value;
      for (double value : state.estimator.velocity_start) snapshots << ',' << value;
      for (double value : state.estimator.accel_bias_body) snapshots << ',' << value;
      for (double value : state.estimator.quaternion_start) snapshots << ',' << value;
      for (double value : state.estimator.support_weights) snapshots << ',' << value;
      snapshots << ',' << state.estimator.timestamp_s << '\n';
    }

    if (ladder_enabled && ladder->stage() == BringupStage::kDisarmed && have_latest_state) {
      ladder->Arm(tick_start);
      // The control process cannot transmit until this loop has bound its
      // PULL socket.  Start the one-second deadman window here rather than
      // treating the first 2 ms tick as a lost supervisor.
      last_operator_liveness_ns = tick_start;
    }
    ControlWire operator_control{};
    // Count every frame the socket yields, not only well-formed ones, and keep
    // the last observed size.  A frame rejected for its size or its header used
    // to vanish without incrementing any counter, which made a silent operator
    // channel indistinguishable from an idle one.  The frame is received into a
    // byte buffer and copied, so the wire layout is explicit rather than a
    // reinterpretation of the receive buffer.
    while (operator_socket) {
      std::array<std::uint8_t, sizeof(ControlWire)> operator_frame{};
      const int operator_bytes = zmq_recv(operator_socket, operator_frame.data(), operator_frame.size(), ZMQ_DONTWAIT);
      if (operator_bytes < 0) break;
      if (operator_bytes == static_cast<int>(operator_frame.size()))
        std::memcpy(&operator_control, operator_frame.data(), sizeof(operator_control));
      last_operator_frame_first_bytes = 0;
      for (int byte = 0; byte < 4 && byte < operator_bytes; ++byte)
        last_operator_frame_first_bytes |= static_cast<std::uint32_t>(operator_frame[byte]) << (8 * byte);
      ++observed_operator_frames;
      last_operator_frame_bytes = operator_bytes;
      if (operator_bytes != static_cast<int>(sizeof(operator_control))) continue;
      last_operator_frame_magic = operator_control.magic;
      last_operator_frame_version = operator_control.version;
      if (operator_control.magic != kControlMagic || operator_control.version != 2) continue;
      ++received_operator_controls;
      last_operator_liveness_ns = MonotonicNs();
      if (operator_control.operation == 2) {
        ladder->Abort("manual operator abort", last_operator_liveness_ns);
        // A value is recorded only by the dedicated same-host loopback test.
        // The production PC client sends zero because cross-host monotonic
        // clocks cannot establish a trustworthy one-way latency.
        if (operator_control.sent_monotonic_ns != 0 &&
            last_operator_liveness_ns >= static_cast<std::int64_t>(operator_control.sent_monotonic_ns)) {
          manual_abort_delivery_latencies.push_back(
              static_cast<double>(last_operator_liveness_ns - static_cast<std::int64_t>(operator_control.sent_monotonic_ns)) / 1e6);
        }
      }
      else if (operator_control.operation == 1 && have_latest_state) ladder->Advance(latest_state, last_operator_liveness_ns);
    }
    TargetWire incoming{};
    while (zmq_recv(target_socket, &incoming, sizeof(incoming), ZMQ_DONTWAIT) == static_cast<int>(sizeof(incoming))) {
      if (incoming.magic != kTargetMagic || incoming.version != 1) continue;
      command.q = incoming.q; command.kp = incoming.kp; command.kd = incoming.kd;
      latest_target = incoming;
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
          if (output_native <= now_mono) {
            ages.push_back(static_cast<double>(now_mono - output_native) / 1e6);
            last_target_output_ns = output_native;
          }
        }
      }
    }
    if (ladder_enabled && have_latest_state) {
      command = ladder->Command(latest_state, MonotonicNs(), last_state_received_ns, latest_target,
                                last_target_output_ns, deadline_missed, last_operator_liveness_ns);
    }
    WriteSafeLoopbackLowCmd(args, dds, command, real_endpoint_armed, observed_mode_machine);
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
         << "\"dds_domain\":" << args.dds_domain << ",\"dds_interface\":\"" << args.dds_interface << "\",\n"
         << "\"dds_safety\":\"LowCmd publisher exists only for domain 232 on loopback; test topic is fixed\",\n"
         << "\"lowcmd_publisher_exists\":" << (dds ? "true" : "false") << ",\n"
         << "\"ticks\":" << lateness.size() << ",\"deadline_misses\":" << missed
         << ",\"state_send_eagain_drops\":" << dropped_state << ",\"targets_received\":" << received_targets
         << ",\"observed_operator_frames\":" << observed_operator_frames
         << ",\"last_operator_frame_magic\":" << last_operator_frame_magic
         << ",\"last_operator_frame_first_bytes\":" << last_operator_frame_first_bytes
         << ",\"last_operator_frame_version\":" << last_operator_frame_version
         << ",\"last_operator_frame_bytes\":" << last_operator_frame_bytes
         << ",\"control_frames_received\":" << received_operator_controls
         << ",\"operator_controls_received\":" << received_operator_controls
         << ",\"current_stage\":\"" << (ladder_enabled ? BringupStageName(ladder->stage()) : "disabled") << "\""
         << ",\"native_bringup_stage\":\"" << (ladder_enabled ? BringupStageName(ladder->stage()) : "disabled") << "\""
         << ",\"native_bringup_event\":\"" << (ladder_enabled ? ladder->last_event() : "") << "\""
         << ",\"abort_reason\":\"" << (ladder_enabled ? ladder->abort_reason() : "") << "\""
         << ",\"abort_timestamp_monotonic_ns\":" << (ladder_enabled ? ladder->abort_timestamp_ns() : 0) << ",\n";
  report << "\"stage_transition_history\":[";
  if (ladder_enabled) {
    for (std::size_t index = 0; index < ladder->transitions().size(); ++index) {
      const auto& transition = ladder->transitions()[index];
      if (index) report << ',';
      report << "{\"timestamp_monotonic_ns\":" << transition.timestamp_ns
             << ",\"stage\":\"" << BringupStageName(transition.stage)
             << "\",\"detail\":\"" << transition.detail << "\"}";
    }
  }
  report << "],\n\"manual_abort_delivery_latency_ms\":{\"samples\":" << manual_abort_delivery_latencies.size() << ',';
  SummaryJson(report, "summary", manual_abort_delivery_latencies);
  report << "},\n";
  SummaryJson(report, "start_lateness_ms", lateness); report << ','; SummaryJson(report, "work_ms", work); report << ',';
  SummaryJson(report, "lowcmd_gap_ms", gaps); report << ','; SummaryJson(report, "ipc_round_trip_ms", round_trips); report << ',';
  SummaryJson(report, "target_age_ms", ages); report << "\n}\n";
  zmq_close(state_socket); zmq_close(target_socket); if (operator_socket) zmq_close(operator_socket); zmq_ctx_term(context);
  std::cout << "Fix 5 native loop completed " << lateness.size() << " ticks; misses=" << missed << "\n";
  return missed == 0 ? 0 : 2;
}

}  // namespace

int main(int argc, char** argv) {
  try { return Run(Parse(argc, argv)); }
  catch (const std::exception& error) { std::cerr << "g1_true23_bfm_lowcmd_loop: " << error.what() << "\n"; return 1; }
}
