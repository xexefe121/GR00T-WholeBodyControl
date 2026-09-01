// Native G1 EDU rev-1.0 True23 first-actuation controller.
//
// Safety ordering is deliberate: immutable artifacts -> ONNX dry run -> DDS
// LowState subscriber -> five advancing CRC-valid mode_machine==4 samples ->
// motion-mode release -> LowCmd publisher.  After release, final-boundary
// validation forbids every controlled-joint kp=0 packet. Recoverable exits
// hold with positive gains and return ownership to captured Unitree mode.

#include "true23_active_gantry_core.hpp"

#include <onnxruntime_cxx_api.h>
#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/b2/motion_switcher/motion_switcher_client.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/robot/g1/loco/g1_loco_client.hpp>
#include <zmq.h>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cerrno>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

namespace fs = std::filesystem;
namespace true23 = gear_sonic::true23;
namespace active = gear_sonic::true23::active;
namespace live = gear_sonic::true23::live;
using LowCmd = unitree_hg::msg::dds_::LowCmd_;
using LowState = unitree_hg::msg::dds_::LowState_;
using nlohmann::json;

inline constexpr std::string_view kLowStateTopic = "rt/lowstate";
inline constexpr std::string_view kLowCmdTopic = "rt/lowcmd";
inline constexpr std::string_view kFrozenLoraKind =
    "g1_true23_frozen_lora_happy_residual_diagnostic_decoder_onnx";
inline constexpr std::string_view kFrozenLoraPromotionKind =
    "g1_true23_frozen_lora_dance_shadow_admission_v1";
inline constexpr std::string_view kFrozenLoraActivePromotionKind =
    "g1_true23_frozen_lora_dance_gantry_active_promotion_v2";
inline constexpr std::string_view kFrozenLoraLiveActivePromotionKind =
    "g1_true23_frozen_lora_live_gantry_active_promotion_v1";
inline constexpr std::string_view kDirectDanceCommand = "DANCE";
inline constexpr std::string_view kFrozenLoraEncoderSha256 =
    "733353148bef1eb8dd83a96416b7a89f0b5c3530ceb9e0cec9c25fdb04f56ff2";
inline constexpr std::string_view kFrozenLoraDecoderSha256 =
    "44d1fb2701f1e65460f1c2c23f676bce4f1d4a44b3b112798dc5034af37946b8";
inline constexpr std::string_view kFrozenLoraReportSha256 =
    "02197e5682a9bddc8f11aa6fa9c32ba909b97ec7d1c316c9a0d660cba2d25b7d";
inline constexpr std::string_view kFrozenLoraSummarySha256 =
    "ed3f5513ed7da8625195b37b674163d64948697d311722ad936be3f4668db801";
inline constexpr std::string_view kFrozenLoraHappyReportSha256 =
    "bcd674314a48f86ce111ea13a5389ac0acf7a9549ffef6de9f45a059014a1a4f";
inline constexpr std::string_view kFrozenLoraHappyTrajectorySha256 =
    "b5d415dacfcd175da08fefeac54cabeb9387e912ec78f49e126a14d65913b697";
inline constexpr std::string_view kFrozenLoraLiveQualificationSha256 =
    "ab1b8493d20d5a4e92b2e432f7ab5eeb8c16862115ff3b9c579d7b1781216d35";
inline constexpr std::string_view kFrozenLoraPacketBundleSha256 =
    "237910ad5dfc370db9645e52f08ba0ca3b0f409a1383d692e1ce1937c5e3dc9d";
inline constexpr int kStateGateTimeoutSeconds = 8;
inline constexpr int kNormalReturnHoldCycles = 250;
inline constexpr int kMinimumPreArmHoldCycles = 25;
inline constexpr int kMotionRestoreStableSamples = 100;
inline constexpr int kMotionRestorePollLimit = 300;
inline constexpr std::int64_t kMaximumFirstHoldWriteDelayNs = 20'000'000;
inline constexpr auto kWriterQuiesceTimeout = std::chrono::milliseconds(100);
[[gnu::used]] const char kCompiledCausalRuntimeSurface[] =
    "WaitJoinCausalReference BuildCausalEncoderInput";
volatile std::sig_atomic_t g_stop_requested = 0;

void HandleSignal(int) { g_stop_requested = 1; }

std::int64_t NowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

std::uint32_t Crc32(const void* bytes, std::uint32_t words) {
  const auto* input = static_cast<const std::uint8_t*>(bytes);
  std::uint32_t crc = 0xffffffffU;
  constexpr std::uint32_t polynomial = 0x04c11db7U;
  for (std::uint32_t index = 0; index < words; ++index) {
    std::uint32_t data = 0;
    std::memcpy(&data, input + index * sizeof(data), sizeof(data));
    std::uint32_t bit = 1U << 31U;
    for (std::uint32_t count = 0; count < 32U; ++count) {
      crc = (crc & 0x80000000U) != 0U
                ? (crc << 1U) ^ polynomial
                : crc << 1U;
      if ((data & bit) != 0U) {
        crc ^= polynomial;
      }
      bit >>= 1U;
    }
  }
  return crc;
}

struct Arguments {
  std::string network;
  fs::path encoder;
  fs::path decoder;
  fs::path metadata;
  fs::path promotion;
  fs::path active_promotion;
  fs::path live_shadow_evidence;
  fs::path execution_evidence;
  std::string pico_endpoint;
  std::string authorization_id;
  std::string gantry_authorization;
  std::string direct_dance_command;
  int post_arm_duration_seconds = 0;
  bool frozen_lora_policy = false;
  bool validate_only = false;
  bool execute_stage_one = false;
  bool help = false;
};

std::string Usage(std::string_view executable) {
  return
      "Usage: " + std::string(executable) +
      " --network <interface> --encoder <pair.encoder.onnx>"
      " --decoder <pair.decoder.onnx> --metadata <candidate.json>"
      " --promotion <promotion.json>"
      " --active-promotion <gantry-active-promotion.json>"
      " --live-shadow-evidence <passed-shadow.jsonl>"
      " --authorization-id <session-id>"
      " [--frozen-lora-policy]"
      " [--validate-only | --network <interface>"
      " --pico-endpoint <tcp://host:port> --execute-stage-one"
      " --evidence <new.jsonl> --post-arm-duration-seconds"
      " <causal-wireless:20..30|frozen-live:1..10|direct-dance:1..5>"
      " --gantry-authorize " +
      std::string(active::kGantryAuthorizationPhrase) +
      " [--direct-dance-command DANCE]" +
      "]"
      "\n\n"
      "Frozen-LoRA policy supports either exact DANCE direct mode or wireless "
      "L2/A live mode, selected by its bound active sidecar. Process signal, "
      "B/R2, state/policy fault, reviewed duration, or physical e-stop stops. "
      "Stage one is gantry-only. Diagnostic ONNX, generic "
      "29-output policy, non-mode-4 robot, stale input, and CRC bypass are "
      "unconditionally rejected. Causal artifacts require "
      "g1_true23_causal_history_reference_terms; future-command packets are "
      "rejected.";
}

Arguments ParseArguments(int argc, char** argv) {
  Arguments result;
  std::unordered_set<std::string> seen_options;
  const auto value = [&](int& index, std::string_view option) {
    if (++index >= argc) {
      throw std::runtime_error(std::string(option) + " requires a value");
    }
    return std::string(argv[index]);
  };
  for (int index = 1; index < argc; ++index) {
    const std::string option = argv[index];
    if (!seen_options.insert(option).second) {
      throw std::runtime_error("duplicate option rejected: " + option);
    }
    if (option == "--help" || option == "-h") {
      result.help = true;
    } else if (option == "--network") {
      result.network = value(index, option);
    } else if (option == "--encoder") {
      result.encoder = value(index, option);
    } else if (option == "--decoder") {
      result.decoder = value(index, option);
    } else if (option == "--metadata") {
      result.metadata = value(index, option);
    } else if (option == "--promotion") {
      result.promotion = value(index, option);
    } else if (option == "--active-promotion") {
      result.active_promotion = value(index, option);
    } else if (option == "--live-shadow-evidence") {
      result.live_shadow_evidence = value(index, option);
    } else if (option == "--evidence") {
      result.execution_evidence = value(index, option);
    } else if (option == "--pico-endpoint") {
      result.pico_endpoint = value(index, option);
    } else if (option == "--authorization-id") {
      result.authorization_id = value(index, option);
    } else if (option == "--gantry-authorize") {
      result.gantry_authorization = value(index, option);
    } else if (option == "--direct-dance-command") {
      result.direct_dance_command = value(index, option);
    } else if (option == "--post-arm-duration-seconds") {
      const auto duration = value(index, option);
      std::size_t parsed = 0;
      try {
        result.post_arm_duration_seconds = std::stoi(duration, &parsed);
      } catch (const std::exception&) {
        throw std::runtime_error(
            "--post-arm-duration-seconds must be an integer");
      }
      if (parsed != duration.size()) {
        throw std::runtime_error(
            "--post-arm-duration-seconds must be an integer");
      }
    } else if (option == "--validate-only") {
      result.validate_only = true;
    } else if (option == "--frozen-lora-policy") {
      result.frozen_lora_policy = true;
    } else if (option == "--execute-stage-one") {
      result.execute_stage_one = true;
    } else {
      throw std::runtime_error("unknown option: " + option);
    }
  }
  if (result.help) {
    return result;
  }
  if (result.encoder.empty() || result.decoder.empty() ||
      result.metadata.empty() ||
      result.promotion.empty() || result.active_promotion.empty() ||
      result.live_shadow_evidence.empty() ||
      !active::IsSafeAuthorizationId(result.authorization_id)) {
    throw std::runtime_error("all artifact and evidence arguments are required");
  }
  if (result.validate_only) {
    if (result.execute_stage_one || !result.gantry_authorization.empty() ||
        !result.direct_dance_command.empty() ||
        !result.execution_evidence.empty() ||
        result.post_arm_duration_seconds != 0) {
      throw std::runtime_error(
          "--validate-only rejects execution, evidence, duration, and "
          "authorization arguments");
    }
  }
  if (result.network.empty() || result.pico_endpoint.empty()) {
    throw std::runtime_error(
        "network and PICO bindings are required for validation/execution");
  }
  if (result.validate_only) {
    if (!result.pico_endpoint.starts_with("tcp://") &&
        !result.pico_endpoint.starts_with("ipc://")) {
      throw std::runtime_error(
          "PICO endpoint must use explicit tcp:// or ipc:// URI");
    }
    return result;
  }
  if (!result.execute_stage_one) {
    throw std::runtime_error("--execute-stage-one is required");
  }
  if (result.gantry_authorization != active::kGantryAuthorizationPhrase) {
    throw std::runtime_error("exact explicit gantry authorization phrase is required");
  }
  if (result.execution_evidence.empty()) {
    throw std::runtime_error("--evidence is required for stage-one execution");
  }
  const bool direct_dance = !result.direct_dance_command.empty();
  if (direct_dance && result.direct_dance_command != kDirectDanceCommand) {
    throw std::runtime_error("exact direct dance command DANCE is required");
  }
  if (direct_dance && !result.frozen_lora_policy) {
    throw std::runtime_error(
        "--direct-dance-command is restricted to frozen-LoRA policy");
  }
  const int minimum_duration =
      direct_dance || result.frozen_lora_policy
          ? 1
          : active::kMinimumStageOnePostArmSeconds;
  const int maximum_duration =
      direct_dance
          ? 5
          : (result.frozen_lora_policy
                 ? 10
                 : active::kMaximumStageOnePostArmSeconds);
  if (result.post_arm_duration_seconds < minimum_duration ||
      result.post_arm_duration_seconds > maximum_duration) {
    throw std::runtime_error(
        direct_dance
            ? "--post-arm-duration-seconds must be within dance-reviewed 1..5 range"
            : (result.frozen_lora_policy
                   ? "--post-arm-duration-seconds must be within frozen-live 1..10 range"
                   : "--post-arm-duration-seconds must be within reviewed 20..30 range"));
  }
  if (!result.pico_endpoint.starts_with("tcp://") &&
      !result.pico_endpoint.starts_with("ipc://")) {
    throw std::runtime_error("PICO endpoint must use explicit tcp:// or ipc:// URI");
  }
  return result;
}

fs::path ResolveFile(const fs::path& argument, std::string_view role) {
  std::error_code error;
  const auto path = fs::canonical(argument, error);
  if (error || !fs::is_regular_file(path, error)) {
    throw std::runtime_error(
        std::string(role) + " is not a resolvable regular file: " +
        argument.string());
  }
  return path;
}

json ParseStrictJson(std::string_view bytes, std::string_view role) {
  std::vector<std::unordered_set<std::string>> keys;
  const json::parser_callback_t callback =
      [&](int, json::parse_event_t event, json& parsed) {
        if (event == json::parse_event_t::object_start) {
          keys.emplace_back();
        } else if (event == json::parse_event_t::key) {
          if (keys.empty()) {
            throw std::runtime_error(std::string(role) + " parser scope error");
          }
          const auto key = parsed.get<std::string>();
          if (!keys.back().insert(key).second) {
            throw std::runtime_error(
                std::string(role) + " contains duplicate key: " + key);
          }
        } else if (event == json::parse_event_t::object_end) {
          if (keys.empty()) {
            throw std::runtime_error(std::string(role) + " parser underflow");
          }
          keys.pop_back();
        }
        return true;
      };
  auto result = json::parse(bytes, callback, true, false);
  if (!result.is_object()) {
    throw std::runtime_error(std::string(role) + " root must be object");
  }
  return result;
}

json LoadJson(const fs::path& path, std::string_view role) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    throw std::runtime_error("cannot open " + std::string(role));
  }
  const std::string bytes{
      std::istreambuf_iterator<char>(stream),
      std::istreambuf_iterator<char>()};
  return ParseStrictJson(bytes, role);
}

std::string LoadBoundedBytes(
    const fs::path& path,
    std::size_t maximum_bytes,
    std::string_view role) {
  std::error_code error;
  const auto size = fs::file_size(path, error);
  if (error || size == 0 || size > maximum_bytes) {
    throw std::runtime_error(
        std::string(role) + " is empty, unreadable, or oversized");
  }
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    throw std::runtime_error("cannot open " + std::string(role));
  }
  std::string bytes(static_cast<std::size_t>(size), '\0');
  stream.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  if (!stream || stream.peek() != std::ifstream::traits_type::eof()) {
    throw std::runtime_error(
        std::string(role) + " changed or failed during bounded read");
  }
  return bytes;
}

Ort::SessionOptions SessionOptions() {
  Ort::SessionOptions options;
  options.SetIntraOpNumThreads(1);
  options.SetInterOpNumThreads(1);
  options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
  return options;
}

class OnnxModel {
 public:
  OnnxModel(Ort::Env& environment, fs::path path,
            bool allow_missing_metadata = false)
      : path_(std::move(path)),
        allow_missing_metadata_(allow_missing_metadata),
        options_(SessionOptions()),
        session_(environment, path_.c_str(), options_) {
    Inspect();
  }

  [[nodiscard]] const true23::ModelSignature& signature() const {
    return signature_;
  }
  [[nodiscard]] const json& embedded() const { return embedded_; }

  template <std::size_t Input, std::size_t Output>
  std::array<float, Output> Run(const std::array<float, Input>& input) {
    std::array<float, Output> result{};
    auto memory = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);
    auto shape = signature_.input_shape;
    auto tensor = Ort::Value::CreateTensor<float>(
        memory, const_cast<float*>(input.data()), input.size(),
        shape.data(), shape.size());
    const char* inputs[] = {signature_.input_name.c_str()};
    const char* outputs[] = {signature_.output_name.c_str()};
    auto values = session_.Run(
        Ort::RunOptions{nullptr}, inputs, &tensor, 1, outputs, 1);
    if (values.size() != 1 || !values.front().IsTensor()) {
      throw std::runtime_error("ONNX inference returned invalid output count/type");
    }
    const auto info = values.front().GetTensorTypeAndShapeInfo();
    if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        info.GetElementCount() != Output) {
      throw std::runtime_error("ONNX inference output shape/type changed");
    }
    const auto* data = values.front().template GetTensorData<float>();
    std::copy_n(data, Output, result.begin());
    if (!live::IsFinite(result)) {
      throw std::runtime_error("ONNX inference produced non-finite output");
    }
    return result;
  }

 private:
  void Inspect() {
    Ort::AllocatorWithDefaultOptions allocator;
    signature_.input_count = session_.GetInputCount();
    signature_.output_count = session_.GetOutputCount();
    if (signature_.input_count == 1) {
      auto name = session_.GetInputNameAllocated(0, allocator);
      signature_.input_name = name ? name.get() : "";
      const auto type_info = session_.GetInputTypeInfo(0);
      const auto info = type_info.GetTensorTypeAndShapeInfo();
      signature_.input_shape = info.GetShape();
      signature_.input_float32 =
          info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
    if (signature_.output_count == 1) {
      auto name = session_.GetOutputNameAllocated(0, allocator);
      signature_.output_name = name ? name.get() : "";
      const auto type_info = session_.GetOutputTypeInfo(0);
      const auto info = type_info.GetTensorTypeAndShapeInfo();
      signature_.output_shape = info.GetShape();
      signature_.output_float32 =
          info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
    signature_.default_opsets =
        true23::InspectDefaultOnnxOpsetsFile(path_.string());
    auto metadata = session_.GetModelMetadata();
    auto metadata_keys = metadata.GetCustomMetadataMapKeysAllocated(allocator);
    if (allow_missing_metadata_ && metadata_keys.empty()) {
      embedded_ = json::object();
      return;
    }
    if (metadata_keys.size() != 1 || !metadata_keys.front() ||
        std::string_view(metadata_keys.front().get()) !=
            true23::kOnnxMetadataKey) {
      throw std::runtime_error("ONNX embedded metadata key set is not exact");
    }
    auto value = metadata.LookupCustomMetadataMapAllocated(
        std::string(true23::kOnnxMetadataKey).c_str(), allocator);
    if (!value) {
      throw std::runtime_error("ONNX embedded metadata is missing");
    }
    embedded_ = ParseStrictJson(value.get(), "embedded ONNX metadata");
  }

  fs::path path_;
  bool allow_missing_metadata_ = false;
  Ort::SessionOptions options_;
  Ort::Session session_;
  true23::ModelSignature signature_;
  json embedded_;
};

template <std::size_t Size>
std::array<float, Size> JsonFloatArray(
    const json& root, std::string_view key) {
  const auto iterator = root.find(std::string(key));
  if (iterator == root.end() || !iterator->is_array() ||
      iterator->size() != Size) {
    throw std::runtime_error(
        "PICO semantic field has wrong size/type: " + std::string(key));
  }
  std::array<float, Size> result{};
  for (std::size_t index = 0; index < Size; ++index) {
    result[index] = iterator->at(index).get<float>();
  }
  return result;
}

live::PicoEncoderTerms ParsePicoTerms(std::string_view message) {
  const auto root = ParseStrictJson(message, "PICO semantic terms");
  live::PicoEncoderTerms result;
  result.schema_version = root.at("schema_version").get<int>();
  result.kind = root.at("kind").get<std::string>();
  result.source_frame_index = root.at("source_frame_index").get<std::uint64_t>();
  result.source_monotonic_ns = root.at("source_monotonic_ns").get<std::int64_t>();
  const auto offsets = root.at("future_frame_offsets_s");
  if (!offsets.is_array() || offsets.size() != result.future_frame_offsets_s.size()) {
    throw std::runtime_error("PICO future_frame_offsets_s shape mismatch");
  }
  for (std::size_t index = 0; index < offsets.size(); ++index) {
    result.future_frame_offsets_s[index] = offsets[index].get<double>();
  }
  result.command_multi_future_lower_body =
      JsonFloatArray<live::kCommandDim>(root, "command_multi_future_lower_body");
  result.vr_3point_local_target =
      JsonFloatArray<live::kVrPositionDim>(root, "vr_3point_local_target");
  result.vr_3point_local_orn_target =
      JsonFloatArray<live::kVrOrientationDim>(root, "vr_3point_local_orn_target");
  result.motion_anchor_ori_b =
      JsonFloatArray<live::kAnchorOrientationDim>(root, "motion_anchor_ori_b");
  const auto errors = live::ValidatePicoEncoderTerms(result);
  if (!errors.empty()) {
    throw std::runtime_error("PICO semantic terms rejected: " + errors.front());
  }
  return result;
}

live::CausalPicoReferenceTerms ParseCausalPicoReferenceTerms(
    std::string_view message) {
  const auto root = ParseStrictJson(message, "causal PICO reference terms");
  return live::ParseCausalPicoReferenceTermsDocument(root);
}

enum class CausalJoinStatus {
  Ready,
  AwaitingCoverage,
  InvalidCoveredRange,
};

struct CausalJoinAttempt {
  CausalJoinStatus status = CausalJoinStatus::AwaitingCoverage;
  std::optional<live::CausalEncoderJoin> joined;
};

class StateMonitor {
 public:
  explicit StateMonitor(active::GantrySafetyCore& core,
                        bool direct_dance_command)
      : core_(core), direct_dance_command_(direct_dance_command) {}
  ~StateMonitor() {
    if (subscriber_) {
      subscriber_->CloseChannel();
    }
  }

  void Start() {
    subscriber_ =
        std::make_shared<unitree::robot::ChannelSubscriber<LowState>>(
            std::string(kLowStateTopic));
    subscriber_->InitChannel(
        [this](const void* message) { OnMessage(message); }, 1);
  }

  bool WaitForMutationGate(std::chrono::steady_clock::time_point deadline) {
    std::unique_lock lock(mutex_);
    return condition_.wait_until(lock, deadline, [&] {
      return core_.mutation_surface_allowed() || core_.stopped() ||
             g_stop_requested != 0;
    }) && core_.mutation_surface_allowed();
  }

  std::optional<live::TimedProprioSample> LatestProprio() const {
    std::lock_guard lock(mutex_);
    return proprio_;
  }

  active::WirelessOperatorState LatestOperator() const {
    std::lock_guard lock(mutex_);
    return operator_state_;
  }

  CausalJoinAttempt WaitJoinCausalReference(
      const live::CausalPicoReferenceTerms& reference,
      std::chrono::steady_clock::time_point deadline) {
    std::unique_lock lock(mutex_);
    condition_.wait_until(lock, deadline, [&] {
      return causal_history_.Covers(
                 reference.pico_anchor_monotonic_ns,
                 reference.control_monotonic_ns) ||
             core_.stopped() || g_stop_requested != 0;
    });
    if (!causal_history_.Covers(
            reference.pico_anchor_monotonic_ns,
            reference.control_monotonic_ns)) {
      return {};
    }
    try {
      return {
          .status = CausalJoinStatus::Ready,
          .joined = live::JoinCausalReferenceWithLowState(
              reference, causal_history_),
      };
    } catch (const std::invalid_argument&) {
      return {.status = CausalJoinStatus::InvalidCoveredRange};
    }
  }

  void WithCore(const auto& function) {
    std::lock_guard lock(mutex_);
    function(core_);
  }

  [[nodiscard]] active::Fault fault() const {
    std::lock_guard lock(mutex_);
    return core_.fault();
  }

 private:
  void OnMessage(const void* message) {
    if (message == nullptr) {
      return;
    }
    LowState state = *static_cast<const LowState*>(message);
    const auto received = NowNs();
    const auto valid_crc =
        state.crc() == Crc32(&state, (sizeof(LowState) >> 2U) - 1U);
    std::lock_guard lock(mutex_);
    if (!valid_crc) {
      core_.ObserveCrcFailure();
      condition_.notify_all();
      return;
    }

    active::StateSample sample;
    sample.tick = state.tick();
    sample.mode_machine = state.mode_machine();
    sample.crc_valid = true;
    sample.received_monotonic_ns = received;
    for (std::size_t slot = 0; slot < active::kMotorSlotCount; ++slot) {
      const auto& motor = state.motor_state()[slot];
      sample.q[slot] = motor.q();
      sample.dq[slot] = motor.dq();
      sample.tau_est[slot] = motor.tau_est();
    }
    core_.ObserveState(sample, received);

    live::TimedProprioSample proprio;
    proprio.received_monotonic_ns = received;
    for (std::size_t compact = 0; compact < 23; ++compact) {
      const auto slot = static_cast<std::size_t>(
          true23::kHardwareJointIds[compact]);
      proprio.hardware_q[compact] = state.motor_state()[slot].q();
      proprio.hardware_dq[compact] = state.motor_state()[slot].dq();
    }
    const auto& imu = state.imu_state();
    std::copy_n(imu.gyroscope().begin(), 3, proprio.gyroscope.begin());
    std::copy_n(imu.quaternion().begin(), 4, proprio.quaternion_wxyz.begin());
    const bool advancing_history_tick =
        !history_tick_.has_value() ||
        static_cast<std::int32_t>(state.tick() - *history_tick_) > 0;
    if (state.mode_machine() == true23::kRequiredModeMachine &&
        !core_.stopped() && advancing_history_tick) {
      if (!causal_history_.Push(proprio)) {
        core_.ObserveInternalFailure();
      } else {
        history_tick_ = state.tick();
        proprio_ = proprio;
      }
    }

    std::array<std::uint8_t, 40> remote{};
    std::copy_n(state.wireless_remote().begin(), remote.size(), remote.begin());
    const auto buttons = active::DecodeWirelessOperator(remote);
    operator_state_ = buttons;
    const bool arm_edge =
        !direct_dance_command_ && buttons.arm_pressed && !arm_pressed_;
    arm_pressed_ = buttons.arm_pressed;
    core_.ObserveOperator(
        {.arm_edge = arm_edge,
         .deadman_held =
             direct_dance_command_ ? true : buttons.deadman_held,
         .stop_requested = buttons.stop_pressed},
        received);
    condition_.notify_all();
  }

  active::GantrySafetyCore& core_;
  std::shared_ptr<unitree::robot::ChannelSubscriber<LowState>> subscriber_;
  mutable std::mutex mutex_;
  std::condition_variable condition_;
  std::optional<live::TimedProprioSample> proprio_;
  live::CausalLowStateHistory causal_history_;
  std::optional<std::uint32_t> history_tick_;
  active::WirelessOperatorState operator_state_;
  bool arm_pressed_ = false;
  bool direct_dance_command_ = false;
};

class ZmqSocket {
 public:
  explicit ZmqSocket(const std::string& endpoint) {
    context_ = zmq_ctx_new();
    if (context_ == nullptr) {
      throw std::runtime_error("cannot create PICO ZMQ context");
    }
    socket_ = zmq_socket(context_, ZMQ_SUB);
    if (socket_ == nullptr) {
      throw std::runtime_error("cannot create PICO ZMQ subscriber");
    }
    constexpr int timeout_ms = 10;
    constexpr int receive_hwm = 1000;
    constexpr int linger_ms = 0;
    if (zmq_setsockopt(socket_, ZMQ_SUBSCRIBE, "", 0) != 0 ||
        zmq_setsockopt(socket_, ZMQ_RCVTIMEO, &timeout_ms,
                       sizeof(timeout_ms)) != 0 ||
        zmq_setsockopt(socket_, ZMQ_RCVHWM, &receive_hwm,
                       sizeof(receive_hwm)) != 0 ||
        zmq_setsockopt(socket_, ZMQ_LINGER, &linger_ms,
                       sizeof(linger_ms)) != 0 ||
        zmq_connect(socket_, endpoint.c_str()) != 0) {
      throw std::runtime_error("cannot configure/connect PICO ZMQ subscriber");
    }
  }
  ~ZmqSocket() {
    if (socket_ != nullptr) {
      zmq_close(socket_);
    }
    if (context_ != nullptr) {
      zmq_ctx_term(context_);
    }
  }

  std::optional<std::string> Receive() {
    std::array<char, 1 << 20> buffer{};
    const auto size = zmq_recv(socket_, buffer.data(), buffer.size(), 0);
    if (size < 0) {
      if (zmq_errno() == EAGAIN) {
        return std::nullopt;
      }
      throw std::runtime_error("PICO ZMQ receive failed");
    }
    if (static_cast<std::size_t>(size) >= buffer.size()) {
      throw std::runtime_error("PICO semantic packet exceeds 1 MiB limit");
    }
    return std::string(buffer.data(), static_cast<std::size_t>(size));
  }

 private:
  void* context_ = nullptr;
  void* socket_ = nullptr;
};

struct ReleasedMotionMode {
  std::string form;
  std::string name;
  int locomotion_fsm_id = -1;
  int locomotion_fsm_mode = -1;
};

std::pair<int, int> ReadLocomotionFsm() {
  unitree::robot::g1::LocoClient client;
  client.SetTimeout(3.0F);
  client.Init();
  int fsm_id = -1;
  int fsm_mode = -1;
  if (client.GetFsmId(fsm_id) != 0 || client.GetFsmMode(fsm_mode) != 0) {
    throw std::runtime_error("locomotion FSM query failed");
  }
  return {fsm_id, fsm_mode};
}

ReleasedMotionMode ReleaseMotionModeAfterGate() {
  auto client =
      std::make_unique<unitree::robot::b2::MotionSwitcherClient>();
  client->SetTimeout(3.0F);
  client->Init();
  std::string form;
  std::string name;
  if (client->CheckMode(form, name) != 0) {
    throw std::runtime_error("motion-mode pre-release CheckMode RPC failed");
  }
  if (name.empty()) {
    throw std::runtime_error(
        "motion-mode pre-release name is empty; exact restoration unavailable");
  }
  const auto [fsm_id, fsm_mode] = ReadLocomotionFsm();
  if (fsm_id == 1) {
    throw std::runtime_error(
        "pre-release locomotion FSM is damped (fsm_id=1); normal standing required");
  }
  const ReleasedMotionMode released{.form = form,
                                    .name = name,
                                    .locomotion_fsm_id = fsm_id,
                                    .locomotion_fsm_mode = fsm_mode};
  if (client->ReleaseMode() != 0) {
    throw std::runtime_error("motion-mode release failed");
  }
  name.clear();
  if (client->CheckMode(form, name) != 0) {
    throw std::runtime_error("motion-mode post-release CheckMode RPC failed");
  }
  if (!name.empty()) {
    throw std::runtime_error("motion mode remains active after release");
  }
  return released;
}

struct MotionRestoreResult {
  int select_mode_attempts = 0;
  int internal_control_attempts = 0;
  int poll_attempts = 0;
  int stable_samples = 0;
};

MotionRestoreResult RestoreMotionModeAfterNormalHold(
    const ReleasedMotionMode& released) {
  auto client =
      std::make_unique<unitree::robot::b2::MotionSwitcherClient>();
  client->SetTimeout(3.0F);
  client->Init();
  unitree::robot::g1::LocoClient locomotion;
  locomotion.SetTimeout(3.0F);
  locomotion.Init();
  std::string form;
  std::string name;
  if (client->CheckMode(form, name) != 0) {
    throw std::runtime_error("motion-mode pre-restore CheckMode RPC failed");
  }
  bool mode_selected = false;
  bool select_rpc_accepted = name == released.name;
  bool internal_control_accepted = false;
  active::MotionRestoreStabilityGate stability_gate(
      kMotionRestoreStableSamples);
  MotionRestoreResult result;
  for (int attempt = 0; attempt < kMotionRestorePollLimit; ++attempt) {
    form.clear();
    name.clear();
    const bool mode_read = client->CheckMode(form, name) == 0;
    if (mode_read && name == released.name) {
      mode_selected = true;
      if (!internal_control_accepted) {
        ++result.internal_control_attempts;
        internal_control_accepted =
            locomotion.SwitchToInternalCtrl(
                unitree::robot::g1::InternalFsmMode::LAST) == 0;
      }
      int fsm_id = -1;
      int fsm_mode = -1;
      const bool exact_restore =
          internal_control_accepted && locomotion.GetFsmId(fsm_id) == 0 &&
          locomotion.GetFsmMode(fsm_mode) == 0 &&
          active::ExactMotionModeRestored(
              released.name, released.locomotion_fsm_id,
              released.locomotion_fsm_mode, name, fsm_id, fsm_mode);
      if (stability_gate.Observe(exact_restore)) {
        result.poll_attempts = attempt + 1;
        result.stable_samples = stability_gate.consecutive_samples();
        return result;
      }
    } else {
      stability_gate.Observe(false);
      internal_control_accepted = false;
      if (mode_read && (!select_rpc_accepted || attempt % 5 == 0)) {
        ++result.select_mode_attempts;
        if (client->SelectMode(released.name) == 0) {
          select_rpc_accepted = true;
        }
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  if (!select_rpc_accepted) {
    throw std::runtime_error(
        "motion-mode restore SelectMode RPC failed after bounded retries");
  }
  if (!internal_control_accepted) {
    throw std::runtime_error(
        "SwitchToInternalCtrl(LAST) RPC failed after motion-mode selection");
  }
  throw std::runtime_error(
      mode_selected
          ? "internal control did not remain in captured locomotion FSM for 10 seconds"
          : "motion mode did not return to captured pre-release mode");
}

bool WaitForWriterQuiescence(
    const active::ModeHandoffInterlock& interlock) {
  const auto deadline = std::chrono::steady_clock::now() +
                        kWriterQuiesceTimeout;
  while (!interlock.restore_allowed() &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return interlock.restore_allowed();
}

LowCmd ToLowCmd(const active::MotorCommand& command) {
  LowCmd result;
  result.mode_pr() = 0;
  result.mode_machine() = true23::kRequiredModeMachine;
  for (std::size_t slot = 0; slot < command.size(); ++slot) {
    auto& output = result.motor_cmd().at(slot);
    output.mode() = command[slot].mode;
    output.q() = static_cast<float>(command[slot].q);
    output.dq() = static_cast<float>(command[slot].dq);
    output.kp() = static_cast<float>(command[slot].kp);
    output.kd() = static_cast<float>(command[slot].kd);
    output.tau() = static_cast<float>(command[slot].tau);
  }
  result.crc() = Crc32(&result, (sizeof(LowCmd) >> 2U) - 1U);
  return result;
}

struct LoadedArtifacts {
  fs::path encoder;
  fs::path decoder;
  fs::path metadata;
  fs::path promotion;
  fs::path active_promotion;
  fs::path live_shadow_evidence;
  json candidate_json;
  json promotion_json;
  json active_json;
  std::string encoder_sha;
  std::string decoder_sha;
  std::string metadata_sha;
  std::string promotion_sha;
  std::string active_sha;
  std::string live_shadow_evidence_sha;
  std::string live_shadow_evidence_bytes;
};

fs::path ResolveNewEvidencePath(const fs::path& argument) {
  if (argument.empty() || argument.filename().empty() ||
      argument.filename() == "." || argument.filename() == "..") {
    throw std::runtime_error("execution evidence path is invalid");
  }
  std::error_code error;
  const auto parent = fs::canonical(
      argument.has_parent_path() ? argument.parent_path() : fs::current_path(),
      error);
  if (error || !fs::is_directory(parent, error)) {
    throw std::runtime_error(
        "execution evidence parent is not a resolvable directory");
  }
  const auto path = parent / argument.filename();
  error.clear();
  const auto status = fs::symlink_status(path, error);
  if (!error && status.type() != fs::file_type::not_found) {
    throw std::runtime_error(
        "execution evidence destination already exists");
  }
  if (error && error != std::errc::no_such_file_or_directory) {
    throw std::runtime_error(
        "cannot inspect execution evidence destination");
  }
  return path;
}

class ExecutionEvidenceLog {
 public:
  ExecutionEvidenceLog(
      fs::path path,
      std::string authorization_id,
      json session_start)
      : path_(ResolveNewEvidencePath(path)),
        authorization_id_(std::move(authorization_id)) {
    fd_ = ::open(
        path_.c_str(),
        O_WRONLY | O_CREAT | O_EXCL | O_APPEND | O_CLOEXEC | O_NOFOLLOW,
        0600);
    if (fd_ < 0) {
      throw std::runtime_error(
          "cannot exclusively create execution evidence: " +
          std::string(std::strerror(errno)));
    }
    try {
      AppendEvent("session_start", std::move(session_start));
    } catch (...) {
      ::close(fd_);
      fd_ = -1;
      throw;
    }
  }

  ExecutionEvidenceLog(const ExecutionEvidenceLog&) = delete;
  ExecutionEvidenceLog& operator=(const ExecutionEvidenceLog&) = delete;

  ~ExecutionEvidenceLog() {
    if (fd_ < 0) {
      return;
    }
    if (!terminal_) {
      try {
        AppendEvent(
            "session_failed",
            {{"passed", false},
             {"reason", "controller_scope_exited_without_terminal_record"}});
      } catch (...) {
      }
    }
    (void)::fsync(fd_);
    (void)::close(fd_);
  }

  [[nodiscard]] const fs::path& path() const { return path_; }

  void AppendEvent(std::string_view event, json payload = json::object()) {
    std::lock_guard lock(mutex_);
    AppendEventLocked(event, std::move(payload));
  }

  void Finalize(std::string_view event, json payload) {
    std::lock_guard lock(mutex_);
    VerifyPathIdentityLocked();
    AppendEventLocked(event, std::move(payload));
    VerifyPathIdentityLocked();
    terminal_ = true;
  }

 private:
  void AppendEventLocked(std::string_view event, json payload) {
    if (!payload.is_object()) {
      throw std::runtime_error("execution evidence payload must be object");
    }
    if (terminal_) {
      throw std::runtime_error("execution evidence is already terminal");
    }
    for (const auto key : {
             "schema_version", "kind", "event", "authorization_id",
             "monotonic_ns"}) {
      if (payload.contains(key)) {
        throw std::runtime_error(
            "execution evidence payload contains reserved field");
      }
    }
    json record = {
        {"schema_version", 1},
        {"kind", "g1_true23_stage1_gantry_execution_evidence"},
        {"event", event},
        {"authorization_id", authorization_id_},
        {"monotonic_ns", NowNs()},
    };
    record.update(payload);
    WriteLocked(record.dump() + "\n");
    if (::fsync(fd_) != 0) {
      throw std::runtime_error("cannot fsync execution evidence");
    }
  }

  void VerifyPathIdentityLocked() const {
    struct stat descriptor_status {};
    struct stat path_status {};
    if (::fstat(fd_, &descriptor_status) != 0 ||
        ::lstat(path_.c_str(), &path_status) != 0 ||
        !S_ISREG(descriptor_status.st_mode) ||
        !S_ISREG(path_status.st_mode) ||
        descriptor_status.st_dev != path_status.st_dev ||
        descriptor_status.st_ino != path_status.st_ino ||
        descriptor_status.st_nlink != 1) {
      throw std::runtime_error(
          "execution evidence path identity changed during session");
    }
  }
  void WriteLocked(std::string_view bytes) {
    std::size_t offset = 0;
    while (offset < bytes.size()) {
      const auto written = ::write(
          fd_, bytes.data() + offset, bytes.size() - offset);
      if (written < 0) {
        if (errno == EINTR) {
          continue;
        }
        throw std::runtime_error("cannot append execution evidence");
      }
      if (written == 0) {
        throw std::runtime_error("zero-byte execution evidence write");
      }
      offset += static_cast<std::size_t>(written);
    }
  }

  fs::path path_;
  std::string authorization_id_;
  int fd_ = -1;
  std::mutex mutex_;
  bool terminal_ = false;
};

LoadedArtifacts LoadArtifacts(const Arguments& arguments) {
  LoadedArtifacts result;
  result.encoder = ResolveFile(arguments.encoder, "encoder");
  result.decoder = ResolveFile(arguments.decoder, "decoder");
  result.metadata = ResolveFile(arguments.metadata, "candidate metadata");
  result.promotion = ResolveFile(arguments.promotion, "MuJoCo promotion");
  result.active_promotion =
      ResolveFile(arguments.active_promotion, "active promotion");
  result.live_shadow_evidence = ResolveFile(
      arguments.live_shadow_evidence, "live shadow evidence");
  std::error_code evidence_time_error;
  const auto evidence_write_time = fs::last_write_time(
      result.live_shadow_evidence, evidence_time_error);
  if (evidence_time_error) {
    throw std::runtime_error("cannot inspect live shadow evidence freshness");
  }
  const auto evidence_age_seconds =
      std::chrono::duration<double>(
          fs::file_time_type::clock::now() - evidence_write_time)
          .count();
  if (!std::isfinite(evidence_age_seconds) || evidence_age_seconds < -5.0 ||
      evidence_age_seconds > active::kMaximumShadowEvidenceAgeSeconds) {
    throw std::runtime_error("live shadow evidence is not fresh");
  }
  const std::array<fs::path, 6> paths = {
      result.encoder, result.decoder, result.metadata, result.promotion,
      result.active_promotion, result.live_shadow_evidence};
  for (std::size_t left = 0; left < paths.size(); ++left) {
    for (std::size_t right = left + 1; right < paths.size(); ++right) {
      if (paths[left] == paths[right]) {
        throw std::runtime_error("all artifact paths must resolve distinctly");
      }
    }
  }
  result.encoder_sha = true23::Sha256File(result.encoder.string());
  result.decoder_sha = true23::Sha256File(result.decoder.string());
  result.metadata_sha = true23::Sha256File(result.metadata.string());
  result.promotion_sha = true23::Sha256File(result.promotion.string());
  result.active_sha = true23::Sha256File(result.active_promotion.string());
  result.live_shadow_evidence_sha =
      true23::Sha256File(result.live_shadow_evidence.string());
  result.live_shadow_evidence_bytes = LoadBoundedBytes(
      result.live_shadow_evidence,
      active::kMaximumShadowEvidenceBytes,
      "live shadow evidence");
  if (true23::Sha256Bytes(result.live_shadow_evidence_bytes) !=
      result.live_shadow_evidence_sha) {
    throw std::runtime_error(
        "live shadow evidence byte/file hash changed during load");
  }
  result.candidate_json = LoadJson(result.metadata, "candidate metadata");
  result.promotion_json = LoadJson(result.promotion, "MuJoCo promotion");
  result.active_json = LoadJson(result.active_promotion, "active promotion");
  return result;
}

void VerifyFilesUnchanged(const LoadedArtifacts& files) {
  if (true23::Sha256File(files.encoder.string()) != files.encoder_sha ||
      true23::Sha256File(files.decoder.string()) != files.decoder_sha ||
      true23::Sha256File(files.metadata.string()) != files.metadata_sha ||
      true23::Sha256File(files.promotion.string()) != files.promotion_sha ||
      true23::Sha256File(files.active_promotion.string()) != files.active_sha ||
      true23::Sha256File(files.live_shadow_evidence.string()) !=
          files.live_shadow_evidence_sha) {
    throw std::runtime_error("artifact changed during validation/load");
  }
}

void RequireExactKeys(
    const json& value,
    std::initializer_list<std::string_view> keys,
    std::string_view context) {
  if (!value.is_object() || value.size() != keys.size()) {
    throw std::runtime_error(
        std::string(context) + " field set is not exact");
  }
  for (const auto key : keys) {
    if (!value.contains(std::string(key))) {
      throw std::runtime_error(
          std::string(context) + " missing field: " + std::string(key));
    }
  }
}

std::string RequireSha256(const json& value, std::string_view context) {
  if (!value.is_string()) {
    throw std::runtime_error(std::string(context) + " must be SHA-256");
  }
  const auto result = value.get<std::string>();
  if (!active::IsLowerSha256(result)) {
    throw std::runtime_error(
        std::string(context) + " must be lowercase SHA-256");
  }
  return result;
}

void ValidateFrozenLoraShadowAdmission(
    const LoadedArtifacts& files,
    const OnnxModel& encoder,
    const OnnxModel& decoder) {
  if (files.encoder_sha != kFrozenLoraEncoderSha256 ||
      files.decoder_sha != kFrozenLoraDecoderSha256 ||
      files.metadata_sha != kFrozenLoraReportSha256) {
    throw std::runtime_error("frozen-LoRA selected artifact SHA-256 mismatch");
  }
  const auto& report = files.candidate_json;
  RequireExactKeys(
      report,
      {"active_motor_control_authorized", "closed_loop_happy_dance_passed",
       "decoder", "deployment_ready", "diagnostic_only",
       "hardware_authorized", "kind", "promotion_eligible",
       "robot_network_commands", "schema_version", "source"},
      "frozen-LoRA decoder report");
  const auto& graph = report.at("decoder");
  RequireExactKeys(
      graph,
      {"filename", "input_name", "input_shape", "opset", "output_name",
       "output_shape", "sha256"},
      "frozen-LoRA decoder graph");
  if (report.at("schema_version") != 1 ||
      report.at("kind") != kFrozenLoraKind ||
      report.at("closed_loop_happy_dance_passed") != true ||
      report.at("diagnostic_only") != true ||
      report.at("deployment_ready") != false ||
      report.at("promotion_eligible") != false ||
      report.at("hardware_authorized") != false ||
      report.at("active_motor_control_authorized") != false ||
      report.at("robot_network_commands") != false ||
      graph.at("filename") != files.decoder.filename().string() ||
      graph.at("input_name") != "obs_dict" ||
      graph.at("input_shape") != json::array({1, 994}) ||
      graph.at("output_name") != "action" ||
      graph.at("output_shape") != json::array({1, 23}) ||
      graph.at("opset") != 13 || graph.at("sha256") != files.decoder_sha) {
    throw std::runtime_error("frozen-LoRA decoder report contract mismatch");
  }
  std::vector<std::string> signature_errors;
  true23::ValidatePairModelSignatures(
      encoder.signature(), decoder.signature(), signature_errors);
  if (!signature_errors.empty()) {
    throw std::runtime_error(signature_errors.front());
  }
  if (!decoder.embedded().empty()) {
    throw std::runtime_error(
        "selected frozen-LoRA decoder unexpectedly gained embedded metadata");
  }
  const auto& promotion = files.promotion_json;
  RequireExactKeys(
      promotion,
      {"schema_version", "kind", "robot_model", "required_mode_machine",
       "native_action_dof", "deployment_bytes_authorized_for_shadow",
       "active_motor_control_authorized", "gantry_or_rated_support_required",
       "free_standing_authorized", "reference_profile",
       "decoder_output_semantics", "runtime_policy_semantics",
       "external_safe_target_transform_required",
       "safe_target_transform_sha256",
       "source_artifacts", "qualification", "stage_one_envelope",
       "promotion_payload_sha256"},
      "frozen-LoRA dance shadow admission");
  if (promotion.at("schema_version") != 1 ||
      promotion.at("kind") != kFrozenLoraPromotionKind ||
      promotion.at("robot_model") != true23::kRobotModel ||
      promotion.at("required_mode_machine") != true23::kRequiredModeMachine ||
      promotion.at("native_action_dof") != true23::kDecoderOutputDim ||
      promotion.at("deployment_bytes_authorized_for_shadow") != true ||
      promotion.at("active_motor_control_authorized") != false ||
      promotion.at("gantry_or_rated_support_required") != true ||
      promotion.at("free_standing_authorized") != false ||
      promotion.at("reference_profile") != live::kCausalReferenceProfile ||
      promotion.at("decoder_output_semantics") !=
          live::kRawNativeActionSemantics ||
      promotion.at("runtime_policy_semantics") !=
          live::kAppliedSafeNativeActionSemantics ||
      promotion.at("external_safe_target_transform_required") != true ||
      promotion.at("safe_target_transform_sha256") !=
          live::kExternalRawSafeTargetTransformSha256) {
    throw std::runtime_error("frozen-LoRA shadow admission contract mismatch");
  }
  const auto& source = promotion.at("source_artifacts");
  RequireExactKeys(
      source,
      {"encoder_sha256", "decoder_sha256", "decoder_report_sha256",
       "candidate_summary_sha256", "happy_dance_report_sha256",
       "happy_dance_trajectory_sha256", "live_qualification_sha256",
       "packet_bundle_sha256"},
      "frozen-LoRA source artifacts");
  if (source.at("encoder_sha256") != kFrozenLoraEncoderSha256 ||
      source.at("decoder_sha256") != kFrozenLoraDecoderSha256 ||
      source.at("decoder_report_sha256") != kFrozenLoraReportSha256 ||
      source.at("candidate_summary_sha256") != kFrozenLoraSummarySha256 ||
      source.at("happy_dance_report_sha256") != kFrozenLoraHappyReportSha256 ||
      source.at("happy_dance_trajectory_sha256") !=
          kFrozenLoraHappyTrajectorySha256 ||
      source.at("live_qualification_sha256") !=
          kFrozenLoraLiveQualificationSha256 ||
      source.at("packet_bundle_sha256") != kFrozenLoraPacketBundleSha256) {
    throw std::runtime_error("frozen-LoRA admission source binding mismatch");
  }
  const auto& qualification = promotion.at("qualification");
  if (qualification.value("happy_dance_passed", false) != true ||
      qualification.value("happy_dance_completed_transitions", 0) != 535 ||
      qualification.value("saved_pico_walk001_completed_transitions", 0) != 684 ||
      qualification.value("software_live_transport_fault_drills_passed", false) !=
          true) {
    throw std::runtime_error("frozen-LoRA dance qualification mismatch");
  }
  const auto& envelope = promotion.at("stage_one_envelope");
  if (envelope.value("action_fraction", 0.0) != 0.10 ||
      envelope.value("maximum_target_rate_rad_per_second", 0.0) != 0.25 ||
      envelope.value("maximum_post_arm_duration_seconds", 0) != 10 ||
      envelope.value("wireless_deadman_required", false) != true ||
      envelope.value("wireless_stop_required", false) != true) {
    throw std::runtime_error("frozen-LoRA stage-one envelope mismatch");
  }
  const auto expected = RequireSha256(
      promotion.at("promotion_payload_sha256"), "promotion payload");
  auto unhashed = promotion;
  unhashed.erase("promotion_payload_sha256");
  if (true23::Sha256CanonicalJson(unhashed) != expected) {
    throw std::runtime_error("frozen-LoRA promotion payload hash mismatch");
  }
}

active::ActiveArtifactBinding ParseFrozenLoraActivePromotion(
    const LoadedArtifacts& files,
    const Arguments& arguments) {
  const auto& root = files.active_json;
  const bool direct_dance = !arguments.direct_dance_command.empty();
  const auto expected_kind = direct_dance
                                 ? kFrozenLoraActivePromotionKind
                                 : kFrozenLoraLiveActivePromotionKind;
  RequireExactKeys(
      root,
      {"schema_version", "kind", "robot_model", "required_mode_machine",
       "native_action_dof", "deployment_ready",
       "active_motor_control_authorized", "gantry_authorized",
       "free_standing_authorized", "decoder_output_semantics",
       "runtime_policy_semantics", "previous_action_semantics",
       "external_safe_target_transform_required",
       "safe_target_transform_sha256", "source_promotion_sha256",
       "encoder_sha256", "decoder_sha256", "decoder_report_sha256",
       "live_shadow_evidence_sha256", "authorization_id",
       "stage_one_envelope", "promotion_payload_sha256"},
      "frozen-LoRA active promotion");
  if (root.at("schema_version") != 1 ||
      root.at("kind") != expected_kind ||
      root.at("robot_model") != true23::kRobotModel ||
      root.at("required_mode_machine") != true23::kRequiredModeMachine ||
      root.at("native_action_dof") != true23::kDecoderOutputDim ||
      root.at("deployment_ready") != true ||
      root.at("active_motor_control_authorized") != true ||
      root.at("gantry_authorized") != true ||
      root.at("free_standing_authorized") != false ||
      root.at("decoder_output_semantics") !=
          live::kRawNativeActionSemantics ||
      root.at("runtime_policy_semantics") !=
          live::kAppliedSafeNativeActionSemantics ||
      root.at("previous_action_semantics") !=
          live::kAppliedSafeNativeActionSemantics ||
      root.at("external_safe_target_transform_required") != true ||
      root.at("safe_target_transform_sha256") !=
          live::kExternalRawSafeTargetTransformSha256 ||
      root.at("source_promotion_sha256") != files.promotion_sha ||
      root.at("encoder_sha256") != files.encoder_sha ||
      root.at("decoder_sha256") != files.decoder_sha ||
      root.at("decoder_report_sha256") != files.metadata_sha ||
      root.at("live_shadow_evidence_sha256") !=
          files.live_shadow_evidence_sha ||
      root.at("authorization_id") != arguments.authorization_id) {
    throw std::runtime_error("frozen-LoRA active promotion binding mismatch");
  }
  const auto& envelope = root.at("stage_one_envelope");
  RequireExactKeys(
      envelope,
      {"action_fraction", "maximum_target_rate_rad_per_second",
       "maximum_post_arm_duration_seconds", "wireless_deadman_required",
       "wireless_stop_required", "direct_dance_command_required",
       "physical_estop_required", "process_signal_stop_required"},
      "frozen-LoRA operator-mode envelope");
  const json expected_direct_command =
      direct_dance ? json(kDirectDanceCommand) : json(false);
  if (envelope.value("action_fraction", 0.0) !=
          active::kStageOneActionFraction ||
      envelope.value("maximum_target_rate_rad_per_second", 0.0) !=
          active::kStageOneTargetRateRadPerSecond ||
      envelope.value("maximum_post_arm_duration_seconds", 0) !=
          (direct_dance ? 5 : 10) ||
      envelope.value("wireless_deadman_required", false) != !direct_dance ||
      envelope.value("wireless_stop_required", false) != !direct_dance ||
      envelope.at("direct_dance_command_required") !=
          expected_direct_command ||
      envelope.value("physical_estop_required", false) != true ||
      envelope.value("process_signal_stop_required", false) != true) {
    throw std::runtime_error("frozen-LoRA active stage-one envelope mismatch");
  }
  const auto expected = RequireSha256(
      root.at("promotion_payload_sha256"), "active promotion payload");
  auto unhashed = root;
  unhashed.erase("promotion_payload_sha256");
  if (true23::Sha256CanonicalJson(unhashed) != expected) {
    throw std::runtime_error("frozen-LoRA active promotion payload hash mismatch");
  }
  return {
      .base_promotion_valid = true,
      .stage = active::ArtifactStage::MujocoPromotion,
      .decoder_output_dim = 23,
      .mode_machine = 4,
      .action_clip_value = 20.0,
      .deployment_ready = true,
      .active_motor_control_authorized = true,
      .gantry_authorized = true,
      .free_standing_authorized = false,
      .decoder_output_semantics =
          std::string(live::kAppliedSafeNativeActionSemantics),
      .previous_action_semantics =
          std::string(live::kAppliedSafeNativeActionSemantics),
      .external_safe_target_transform_allowed = false,
      .safe_target_transform_sha256 =
          std::string(live::kSafeTargetTransformSha256),
      .source_promotion_sha256 = files.promotion_sha,
      .checkpoint_sha256 = std::string(kFrozenLoraSummarySha256),
      .lineage_sha256 = std::string(kFrozenLoraReportSha256),
      .policy_state_sha256 = files.decoder_sha,
      .encoder_onnx_sha256 = files.encoder_sha,
      .decoder_onnx_sha256 = files.decoder_sha,
      .metadata_sha256 = files.metadata_sha,
      .full_campaign_aggregate_sha256 =
          std::string(kFrozenLoraHappyReportSha256),
      .full_campaign_shard_manifest_sha256 =
          std::string(kFrozenLoraPacketBundleSha256),
      .live_shadow_evidence_sha256 = files.live_shadow_evidence_sha,
      .authorization_id = arguments.authorization_id,
  };
}

void ValidateCausalDiagnosticPair(
    const LoadedArtifacts& files,
    const OnnxModel& encoder,
    const OnnxModel& decoder) {
  const auto& root = files.candidate_json;
  RequireExactKeys(
      root,
      {"schema_version", "kind", "diagnostic_only", "deployment_ready",
       "promotion_eligible", "active_motor_control_authorized",
       "checkpoint_role", "allowed_uses", "forbidden_uses",
       "no_robot_or_network_commands_performed", "source", "contract",
       "artifacts", "hashes", "validation", "metadata_payload_sha256"},
      "causal diagnostic bundle");
  if (root.at("schema_version") != live::kSafeTargetDiagnosticSchemaVersion ||
      root.at("kind") != "g1_true23_mjlab_diagnostic_onnx_pair" ||
      root.at("diagnostic_only") != true ||
      root.at("deployment_ready") != false ||
      root.at("promotion_eligible") != false ||
      root.at("active_motor_control_authorized") != false ||
      root.at("checkpoint_role") != "training_resume_only" ||
      root.at("allowed_uses") != json::array(
          {"mujoco_sim2sim_diagnostic", "pico_shadow_diagnostic"}) ||
      root.at("forbidden_uses") != json::array(
          {"active_motor_control", "deployment", "promotion"}) ||
      root.at("no_robot_or_network_commands_performed") != true) {
    throw std::runtime_error("causal diagnostic safety contract mismatch");
  }

  const auto& source = root.at("source");
  RequireExactKeys(
      source,
      {"checkpoint_filename", "checkpoint_update_count", "reference_profile",
       "simulation_candidate_review_allowed"},
      "causal diagnostic source");
  if (!source.at("checkpoint_filename").is_string() ||
      source.at("checkpoint_filename").get<std::string>().empty() ||
      !source.at("checkpoint_update_count").is_number_integer() ||
      source.at("checkpoint_update_count").get<int>() <= 0 ||
      source.at("reference_profile") != live::kCausalReferenceProfile ||
      source.at("simulation_candidate_review_allowed") != true) {
    throw std::runtime_error("causal diagnostic source mismatch");
  }

  const auto& contract = root.at("contract");
  RequireExactKeys(
      contract,
      {"robot_model", "required_mode_machine", "native_action_dof",
       "history_length", "observation_layout", "onnx_opset", "encoder",
       "decoder", "decoder_output_semantics",
       "external_safe_target_transform_allowed",
       "previous_action_semantics", "safe_target_transform"},
      "causal diagnostic contract");
  if (contract.at("robot_model") != true23::kRobotModel ||
      contract.at("required_mode_machine") != true23::kRequiredModeMachine ||
      contract.at("native_action_dof") != true23::kDecoderOutputDim ||
      contract.at("history_length") != true23::kHistoryLength ||
      contract.at("observation_layout") !=
          "canonical_il29_fixed_slots_v1" ||
      contract.at("onnx_opset") != true23::kOnnxOpset) {
    throw std::runtime_error("causal diagnostic embodiment mismatch");
  }
  std::vector<std::string> signature_errors;
  true23::ValidatePairModelSignatures(
      encoder.signature(), decoder.signature(), signature_errors);
  if (!signature_errors.empty()) {
    throw std::runtime_error(signature_errors.front());
  }

  const auto& artifacts = root.at("artifacts");
  RequireExactKeys(
      artifacts,
      {"encoder_onnx_filename", "decoder_onnx_filename", "metadata_filename"},
      "causal diagnostic artifacts");
  if (artifacts.at("encoder_onnx_filename") !=
          files.encoder.filename().string() ||
      artifacts.at("decoder_onnx_filename") !=
          files.decoder.filename().string() ||
      artifacts.at("metadata_filename") !=
          files.metadata.filename().string()) {
    throw std::runtime_error("causal diagnostic filename binding mismatch");
  }

  const auto& hashes = root.at("hashes");
  RequireExactKeys(
      hashes,
      {"checkpoint_sha256", "lineage_sha256", "policy_state_sha256",
       "encoder_state_sha256", "decoder_state_sha256",
       "encoder_onnx_sha256", "decoder_onnx_sha256",
       "encoder_embedded_metadata_sha256",
       "decoder_embedded_metadata_sha256", "safe_target_transform_sha256"},
      "causal diagnostic hashes");
  for (const auto& [key, value] : hashes.items()) {
    (void)RequireSha256(value, key);
  }
  if (hashes.at("encoder_onnx_sha256") != files.encoder_sha ||
      hashes.at("decoder_onnx_sha256") != files.decoder_sha ||
      true23::Sha256CanonicalJson(encoder.embedded()) !=
          hashes.at("encoder_embedded_metadata_sha256") ||
      true23::Sha256CanonicalJson(decoder.embedded()) !=
          hashes.at("decoder_embedded_metadata_sha256")) {
    throw std::runtime_error("causal diagnostic ONNX/hash binding mismatch");
  }
  if (encoder.embedded().at("schema_version") !=
          live::kSafeTargetDiagnosticSchemaVersion ||
      decoder.embedded().at("schema_version") !=
          live::kSafeTargetDiagnosticSchemaVersion) {
    throw std::runtime_error("causal diagnostic embedded schema mismatch");
  }
  live::ValidateAppliedSafeDecoderContract(
      contract, hashes, encoder.embedded(), decoder.embedded());

  const auto expected_payload = RequireSha256(
      root.at("metadata_payload_sha256"), "metadata_payload_sha256");
  auto unhashed = root;
  unhashed.erase("metadata_payload_sha256");
  if (true23::Sha256CanonicalJson(unhashed) != expected_payload) {
    throw std::runtime_error("causal diagnostic payload hash mismatch");
  }
}

void ValidateCausalMujocoPromotion(
    const LoadedArtifacts& files) {
  const auto& root = files.promotion_json;
  RequireExactKeys(
      root,
      {"schema_version", "kind", "robot_model", "promotion_stage",
       "deployment_bytes_authorized", "active_motor_control_authorized",
       "gantry_or_rated_support_required", "free_standing_authorized",
       "required_mode_machine", "decoder_output_semantics",
       "previous_action_semantics", "external_safe_target_transform_allowed",
       "safe_target_transform_sha256", "source_artifact",
       "full_campaign_evidence", "promotion_payload_sha256"},
      "causal MuJoCo promotion");
  if (root.at("schema_version") != 2 ||
      root.at("kind") != "g1_true23_causal_mujoco_promotion" ||
      root.at("robot_model") != true23::kRobotModel ||
      root.at("promotion_stage") != "causal_mujoco_full" ||
      root.at("deployment_bytes_authorized") != true ||
      root.at("active_motor_control_authorized") != false ||
      root.at("gantry_or_rated_support_required") != true ||
      root.at("free_standing_authorized") != false ||
      root.at("required_mode_machine") != true23::kRequiredModeMachine ||
      root.at("decoder_output_semantics") !=
          live::kAppliedSafeNativeActionSemantics ||
      root.at("previous_action_semantics") !=
          live::kAppliedSafeNativeActionSemantics ||
      root.at("external_safe_target_transform_allowed") != false ||
      root.at("safe_target_transform_sha256") !=
          live::kSafeTargetTransformSha256) {
    throw std::runtime_error("causal MuJoCo promotion contract mismatch");
  }

  const auto& candidate = files.candidate_json;
  const auto& source = root.at("source_artifact");
  RequireExactKeys(
      source,
      {"checkpoint_filename", "checkpoint_sha256", "checkpoint_update_count",
       "lineage_sha256", "policy_state_sha256", "reference_profile",
       "causal_reference_contract_sha256", "encoder_onnx_filename",
       "encoder_onnx_sha256", "decoder_onnx_filename",
       "decoder_onnx_sha256", "metadata_filename", "metadata_sha256",
       "metadata_payload_sha256"},
      "causal promotion source artifact");
  for (const auto key : {
           "checkpoint_sha256", "lineage_sha256", "policy_state_sha256",
           "causal_reference_contract_sha256", "encoder_onnx_sha256",
           "decoder_onnx_sha256", "metadata_sha256",
           "metadata_payload_sha256"}) {
    (void)RequireSha256(source.at(key), key);
  }
  if (source.at("checkpoint_filename") !=
          candidate.at("source").at("checkpoint_filename") ||
      source.at("checkpoint_update_count") !=
          candidate.at("source").at("checkpoint_update_count") ||
      source.at("checkpoint_sha256") !=
          candidate.at("hashes").at("checkpoint_sha256") ||
      source.at("lineage_sha256") !=
          candidate.at("hashes").at("lineage_sha256") ||
      source.at("policy_state_sha256") !=
          candidate.at("hashes").at("policy_state_sha256") ||
      source.at("reference_profile") != live::kCausalReferenceProfile ||
      source.at("causal_reference_contract_sha256") !=
          live::kCausalReferenceContractSha256 ||
      source.at("encoder_onnx_filename") != files.encoder.filename().string() ||
      source.at("decoder_onnx_filename") != files.decoder.filename().string() ||
      source.at("metadata_filename") != files.metadata.filename().string() ||
      source.at("encoder_onnx_sha256") != files.encoder_sha ||
      source.at("decoder_onnx_sha256") != files.decoder_sha ||
      source.at("metadata_sha256") != files.metadata_sha ||
      source.at("metadata_payload_sha256") !=
          candidate.at("metadata_payload_sha256")) {
    throw std::runtime_error("causal promotion source binding mismatch");
  }

  const auto& campaign = root.at("full_campaign_evidence");
  RequireExactKeys(
      campaign,
      {"aggregate_report_filename", "aggregate_report_sha256",
       "aggregate_report_payload_sha256", "campaign_layout",
       "shard_manifest_sha256", "trace_manifest_sha256", "shards",
       "deterministic_seeds", "scenarios", "run_count",
       "episodes_per_scenario", "total_episodes", "total_records",
       "all_strict_gates_pass", "summary_metrics", "provenance"},
      "causal full campaign evidence");
  for (const auto key : {
           "aggregate_report_sha256", "aggregate_report_payload_sha256",
           "shard_manifest_sha256", "trace_manifest_sha256"}) {
    (void)RequireSha256(campaign.at(key), key);
  }
  if (campaign.at("campaign_layout") != "monolithic" ||
      campaign.at("shards") != json::array() ||
      campaign.at("shard_manifest_sha256") !=
          true23::Sha256CanonicalJson(json::array()) ||
      campaign.at("deterministic_seeds") != json::array({1729, 2718, 3141}) ||
      campaign.at("scenarios") != json::array(
          {"nominal", "push_50", "push_100", "domain_push_100"}) ||
      campaign.at("run_count") != 12 ||
      campaign.at("episodes_per_scenario") != 66 ||
      campaign.at("total_episodes") != 264 ||
      campaign.at("total_records") != 66000 ||
      campaign.at("all_strict_gates_pass") != true) {
    throw std::runtime_error("causal full campaign fixed contract mismatch");
  }

  const auto& provenance = campaign.at("provenance");
  RequireExactKeys(
      provenance,
      {"base_sim2sim_config_relpath", "base_sim2sim_config_sha256",
       "mjcf_relpath", "mjcf_sha256", "domain_source_relpath",
       "domain_source_sha256", "mjlab_env_relpath", "mjlab_env_sha256",
       "executed_producer_archive_manifest_relpath",
       "executed_producer_archive_manifest_sha256",
       "executed_runtime_filename", "executed_runtime_sha256",
       "executed_runner_filename", "executed_runner_sha256"},
      "causal campaign provenance");
  for (const auto key : {
           "base_sim2sim_config_sha256", "mjcf_sha256",
           "domain_source_sha256", "mjlab_env_sha256",
           "executed_producer_archive_manifest_sha256",
           "executed_runtime_sha256", "executed_runner_sha256"}) {
    (void)RequireSha256(provenance.at(key), key);
  }
  for (const auto key : {
           "base_sim2sim_config_relpath", "mjcf_relpath",
           "domain_source_relpath", "mjlab_env_relpath",
           "executed_producer_archive_manifest_relpath"}) {
    if (!provenance.at(key).is_string()) {
      throw std::runtime_error("causal provenance path must be a string");
    }
    const fs::path relative(provenance.at(key).get<std::string>());
    if (relative.empty() || relative.is_absolute() ||
        relative.lexically_normal() != relative) {
      throw std::runtime_error("causal provenance path is not canonical relative");
    }
    for (const auto& component : relative) {
      if (component == "..") {
        throw std::runtime_error("causal provenance path escapes root");
      }
    }
  }
  for (const auto key : {
           "executed_runtime_filename", "executed_runner_filename"}) {
    if (!provenance.at(key).is_string()) {
      throw std::runtime_error("causal provenance filename must be a string");
    }
    const auto filename = provenance.at(key).get<std::string>();
    if (filename.empty() || fs::path(filename).filename() != filename) {
      throw std::runtime_error("causal provenance filename is not a basename");
    }
  }

  const auto& metrics = campaign.at("summary_metrics");
  RequireExactKeys(
      metrics,
      {"run_count", "episode_count", "record_count", "termination_count",
       "fall_count", "policy_output_nonfinite_count",
       "joint_limit_violation_count", "joint_velocity_bound_violation_count",
       "effort_bound_violation_count", "minimum_recovery_fraction",
       "maximum_recovery_time_s", "max_action_saturation_fraction",
       "max_joint_velocity_ratio", "max_effort_ratio",
       "max_abs_native_action_raw", "minimum_base_height_m",
       "maximum_tilt_rad"},
      "causal campaign summary metrics");
  if (metrics.at("run_count") != 12 || metrics.at("episode_count") != 264 ||
      metrics.at("record_count") != 66000 ||
      metrics.at("termination_count") != 0 || metrics.at("fall_count") != 0 ||
      metrics.at("policy_output_nonfinite_count") != 0 ||
      metrics.at("joint_limit_violation_count") != 0 ||
      metrics.at("joint_velocity_bound_violation_count") != 0 ||
      metrics.at("effort_bound_violation_count") != 0 ||
      metrics.at("minimum_recovery_fraction") != 1.0 ||
      metrics.at("maximum_recovery_time_s").get<double>() > 2.0 ||
      metrics.at("max_action_saturation_fraction") != 0.0 ||
      metrics.at("max_joint_velocity_ratio").get<double>() > 1.0 ||
      metrics.at("max_effort_ratio").get<double>() > 1.0) {
    throw std::runtime_error("causal campaign strict metrics failed");
  }
  for (const auto key : {
           "maximum_recovery_time_s", "max_joint_velocity_ratio",
           "max_effort_ratio", "max_abs_native_action_raw",
           "minimum_base_height_m", "maximum_tilt_rad"}) {
    if (!metrics.at(key).is_number() ||
        !std::isfinite(metrics.at(key).get<double>())) {
      throw std::runtime_error("causal campaign metric is non-finite");
    }
  }

  const auto expected_payload = RequireSha256(
      root.at("promotion_payload_sha256"), "promotion_payload_sha256");
  auto unhashed = root;
  unhashed.erase("promotion_payload_sha256");
  if (true23::Sha256CanonicalJson(unhashed) != expected_payload) {
    throw std::runtime_error("causal promotion payload hash mismatch");
  }
}

int Run(const Arguments& arguments) {
  auto files = LoadArtifacts(arguments);
  const bool frozen_lora_policy =
      files.candidate_json.value("kind", std::string{}) == kFrozenLoraKind;
  const bool direct_dance = !arguments.direct_dance_command.empty();
  if (frozen_lora_policy != arguments.frozen_lora_policy) {
    throw std::runtime_error(
        "--frozen-lora-policy must exactly match selected artifact class");
  }
  Ort::Env environment(ORT_LOGGING_LEVEL_WARNING, "g1_true23_active_gantry");
  OnnxModel encoder(environment, files.encoder);
  OnnxModel decoder(environment, files.decoder, frozen_lora_policy);

  if (frozen_lora_policy) {
    ValidateFrozenLoraShadowAdmission(files, encoder, decoder);
  } else {
    ValidateCausalDiagnosticPair(files, encoder, decoder);
    ValidateCausalMujocoPromotion(files);
  }
  const auto shadow_summary = active::ValidateLiveShadowEvidenceJsonl(
      files.live_shadow_evidence_bytes,
      {
          .encoder_sha256 = files.encoder_sha,
          .decoder_sha256 = files.decoder_sha,
          .metadata_sha256 = files.metadata_sha,
           .promotion_sha256 = files.promotion_sha,
           .network = arguments.network,
           .pico_endpoint = arguments.pico_endpoint,
           .external_safe_target_transform_applied = frozen_lora_policy,
       });
  const bool causal_reference_artifact = true;
  auto artifact = frozen_lora_policy
      ? ParseFrozenLoraActivePromotion(files, arguments)
      : active::ParseActivePromotion(
            files.active_json, true, files.promotion_sha,
            files.encoder_sha, files.decoder_sha, files.metadata_sha,
            files.live_shadow_evidence_sha);
  if (artifact.authorization_id != arguments.authorization_id) {
    throw std::runtime_error(
        "operator authorization-id does not match active promotion");
  }
  if (!frozen_lora_policy) {
    const auto& candidate_hashes = files.candidate_json.at("hashes");
    const auto& promotion_source =
        files.promotion_json.at("source_artifact");
    const auto& campaign =
        files.promotion_json.at("full_campaign_evidence");
    if (artifact.checkpoint_sha256 !=
            candidate_hashes.at("checkpoint_sha256") ||
        artifact.lineage_sha256 != candidate_hashes.at("lineage_sha256") ||
        artifact.policy_state_sha256 !=
            candidate_hashes.at("policy_state_sha256") ||
        artifact.safe_target_transform_sha256 !=
            candidate_hashes.at("safe_target_transform_sha256") ||
        artifact.checkpoint_sha256 !=
            promotion_source.at("checkpoint_sha256") ||
        artifact.lineage_sha256 != promotion_source.at("lineage_sha256") ||
        artifact.policy_state_sha256 !=
            promotion_source.at("policy_state_sha256") ||
        artifact.full_campaign_aggregate_sha256 !=
            campaign.at("aggregate_report_sha256") ||
        artifact.full_campaign_shard_manifest_sha256 !=
            campaign.at("shard_manifest_sha256")) {
      throw std::runtime_error(
          "active promotion material/campaign binding mismatch");
    }
  }
  const auto active_errors = active::ValidateActiveArtifactBinding(artifact);
  if (!active_errors.empty()) {
    throw std::runtime_error("active promotion rejected: " + active_errors.front());
  }

  const std::array<float, true23::kEncoderInputDim> zero_encoder{};
  const auto token = encoder.Run<true23::kEncoderInputDim,
                                 true23::kEncoderOutputDim>(zero_encoder);
  std::array<float, true23::kDecoderInputDim> zero_decoder{};
  std::copy(token.begin(), token.end(), zero_decoder.begin());
  (void)decoder.Run<true23::kDecoderInputDim,
                    true23::kDecoderOutputDim>(zero_decoder);
  VerifyFilesUnchanged(files);
  if (arguments.validate_only) {
    std::cout
        << "[PASS] exact True23 "
        << (frozen_lora_policy ? "frozen-LoRA" : "causal")
        << " gantry promotion and "
        << shadow_summary.action_frames
        << "-frame promoted shadow PASS validated; no DDS, motion-mode "
           "client, or LowCmd publisher was created.\n";
    return 0;
  }

  std::error_code executable_error;
  const auto executable_path = fs::canonical("/proc/self/exe", executable_error);
  if (executable_error || !fs::is_regular_file(executable_path)) {
    throw std::runtime_error("cannot resolve running controller binary");
  }
  const auto executable_sha = true23::Sha256File(executable_path.string());
  ExecutionEvidenceLog execution_evidence(
      arguments.execution_evidence,
      arguments.authorization_id,
      {
          {"controller_binary_path", executable_path.string()},
          {"controller_binary_sha256", executable_sha},
          {"encoder_sha256", files.encoder_sha},
          {"decoder_sha256", files.decoder_sha},
          {"metadata_sha256", files.metadata_sha},
          {"promotion_sha256", files.promotion_sha},
          {"active_promotion_sha256", files.active_sha},
          {"live_shadow_evidence_sha256", files.live_shadow_evidence_sha},
          {"live_shadow_action_frames", shadow_summary.action_frames},
          {"network", arguments.network},
          {"pico_endpoint", arguments.pico_endpoint},
          {"post_arm_duration_seconds",
           arguments.post_arm_duration_seconds},
          {"operator_contract",
           direct_dance
               ? "bounded_direct_dance_command_v1"
               : "wireless_deadman_v1"},
          {"minimum_policy_command_frames",
           active::kMinimumPromotedShadowActionFrames},
          {"mutation_gate_open", false},
          {"motion_mode_released", false},
          {"lowcmd_publisher_created", false},
      });
  execution_evidence.AppendEvent(
      "artifact_gate_passed",
      {{"onnx_dry_run_passed", true},
       {"artifact_bytes_reverified", true},
       {"active_promotion_authorized", true},
       {"live_shadow_evidence_validated", true}});

  active::GantrySafetyCore core(std::move(artifact));
  if (core.stopped()) {
    throw std::runtime_error("active artifact safety core refused authorization");
  }

  // Start subscriber-only. Policy history and ONNX inference warm completely
  // while Unitree motion mode still owns posture. LowCmd writer stays closed.
  unitree::robot::ChannelFactory::Instance()->Init(0, arguments.network);
  StateMonitor monitor(core, direct_dance);
  monitor.Start();
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::seconds(kStateGateTimeoutSeconds);
  if (!monitor.WaitForMutationGate(deadline)) {
    throw std::runtime_error(
        "five advancing CRC-valid mode_machine==4 states were not obtained");
  }
  execution_evidence.AppendEvent(
      "mutation_gate_open",
      {{"stable_advancing_lowstate_samples", 5},
       {"required_mode_machine", true23::kRequiredModeMachine},
       {"crc_valid", true}});
  // Publisher construction is non-mutating. It occurs before motion release so
  // first sampled-hold packet can follow ReleaseMode without a DDS setup gap.
  auto publisher =
      std::make_shared<unitree::robot::ChannelPublisher<LowCmd>>(
          std::string(kLowCmdTopic));
  publisher->InitChannel();
  execution_evidence.AppendEvent(
      "lowcmd_publisher_created",
      {{"topic", std::string(kLowCmdTopic)},
       {"writes_before_event", 0},
       {"motion_mode_released", false}});
  if (g_stop_requested != 0) {
    throw std::runtime_error(
        "signal stopped controller before read-only policy prewarm");
  }
  std::atomic<bool> stop_threads{false};
  std::atomic<bool> first_policy_ready_for_arm{false};
  std::atomic<bool> motion_mode_released{false};
  std::atomic<std::int64_t> first_policy_write_ns{0};
  std::atomic<std::int64_t> first_pre_arm_hold_write_ns{0};
  std::atomic<std::uint64_t> publisher_write_count{0};
  std::atomic<std::uint64_t> pre_arm_hold_frames{0};
  std::atomic<std::uint64_t> normal_return_hold_frames{0};
  std::atomic<std::uint64_t> startup_damping_frames{0};
  std::atomic<bool> motion_mode_restored{false};
  std::atomic<bool> software_return_requested{false};
  std::atomic<bool> emergency_mode_restore_requested{false};
  std::atomic<std::uint64_t> rejected_non_positive_gain_commands{0};
  active::ModeHandoffInterlock mode_handoff_interlock;
  std::int64_t motion_mode_released_ns = 0;
  std::uint64_t policy_command_frames = 0;
  constexpr int damping_frames_after_stop = 0;
  double maximum_target_delta_from_state_rad = 0.0;
  double maximum_target_slew_rad = 0.0;
  double maximum_abs_predicted_effort_nm = 0.0;
  double maximum_abs_feedforward_tau_nm = 0.0;
  std::array<double, 23> previous_active_targets{};
  bool have_previous_active_targets = false;
  std::string inference_error;
  std::string writer_error;
  bool publisher_write_failed = false;
  bool writer_quiesced_before_restore = false;
  bool lowcmd_publisher_closed_before_restore = false;
  int restore_select_mode_attempts = 0;
  int restore_internal_control_attempts = 0;
  int restore_poll_attempts = 0;
  int restore_stable_samples = 0;
  std::uint64_t accepted_inference_frames = 0;
  std::int64_t maximum_inference_duration_ns = 0;
  std::int64_t maximum_packet_age_ns = 0;

  std::jthread inference_thread([&](std::stop_token stop_token) {
    try {
      const auto recover_or_latch = [&](auto&& latch_fault) {
        bool recovering = false;
        const auto recovery_ns = NowNs();
        monitor.WithCore([&](active::GantrySafetyCore& value) {
          recovering = value.BeginSoftwareFaultReturnHold(recovery_ns);
          if (!recovering) {
            latch_fault(value);
          }
        });
        if (recovering) {
          software_return_requested.store(true, std::memory_order_release);
        }
        return recovering;
      };
      ZmqSocket socket(arguments.pico_endpoint);
      live::ProprioHistory history;
      active::RealProprioWarmupGate real_history_gate;
      std::array<float, 23> previous_action{};
      std::optional<std::uint64_t> last_frame;
      std::optional<std::int64_t> last_source_monotonic_ns;
      while (!stop_token.stop_requested() && !stop_threads.load()) {
        const auto message = socket.Receive();
        if (!message.has_value()) {
          continue;
        }
        std::optional<live::PicoEncoderTerms> released_terms;
        std::optional<live::CausalPicoReferenceTerms> causal_reference;
        std::uint64_t frame_index = 0;
        std::int64_t source_monotonic_ns = 0;
        if (causal_reference_artifact) {
          causal_reference = ParseCausalPicoReferenceTerms(*message);
          frame_index = causal_reference->control_source_frame_index;
          source_monotonic_ns = causal_reference->control_monotonic_ns;
        } else {
          released_terms = ParsePicoTerms(*message);
          frame_index = released_terms->source_frame_index;
          source_monotonic_ns = released_terms->source_monotonic_ns;
        }
        const auto now = NowNs();
        const auto age = now - source_monotonic_ns;
        maximum_packet_age_ns = std::max(maximum_packet_age_ns, age);
        if (age < -active::kFutureClockToleranceNs ||
            age > active::kPolicyFreshnessNs) {
          inference_error =
              "pico_age_out_of_range frame=" + std::to_string(frame_index) +
              " age_ns=" + std::to_string(age);
          std::cerr << "PICO/inference fault: " << inference_error << '\n';
          recover_or_latch([](active::GantrySafetyCore& value) {
            value.ObservePicoTermsFailure();
          });
          break;
        }
        if (last_frame.has_value() && frame_index != *last_frame + 1U) {
          inference_error =
              "pico_frame_discontinuity previous=" +
              std::to_string(*last_frame) + " current=" +
              std::to_string(frame_index);
          std::cerr << "PICO/inference fault: " << inference_error << '\n';
          recover_or_latch([](active::GantrySafetyCore& value) {
            value.ObservePicoFrameRegression();
          });
          break;
        }
        if (last_source_monotonic_ns.has_value() &&
            source_monotonic_ns !=
                *last_source_monotonic_ns + active::kShadowControlPeriodNs) {
          inference_error =
              "pico_timestamp_discontinuity previous=" +
              std::to_string(*last_source_monotonic_ns) + " current=" +
              std::to_string(source_monotonic_ns);
          std::cerr << "PICO/inference fault: " << inference_error << '\n';
          recover_or_latch([](active::GantrySafetyCore& value) {
            value.ObservePicoTermsFailure();
          });
          break;
        }
        last_frame = frame_index;
        last_source_monotonic_ns = source_monotonic_ns;

        live::TimedProprioSample proprio;
        std::array<float, true23::kEncoderInputDim> encoder_input{};
        if (causal_reference_artifact) {
          const auto attempt = monitor.WaitJoinCausalReference(
              *causal_reference,
              std::chrono::steady_clock::now() +
                  std::chrono::milliseconds(35));
          if (attempt.status != CausalJoinStatus::Ready ||
              !attempt.joined.has_value()) {
            if (monitor.fault() != active::Fault::None ||
                stop_threads.load() || g_stop_requested != 0) {
              break;
            }
            const std::string join_failure =
                attempt.status == CausalJoinStatus::AwaitingCoverage
                    ? "lowstate_coverage_timeout"
                    : "lowstate_covered_range_invalid";
            bool reacquiring = false;
            monitor.WithCore([&](active::GantrySafetyCore& value) {
              reacquiring = value.BeginPreArmPolicyReacquisition();
            });
            if (reacquiring) {
              history = live::ProprioHistory{};
              real_history_gate = active::RealProprioWarmupGate{};
              previous_action.fill(0.0F);
              first_policy_ready_for_arm.store(false, std::memory_order_release);
              execution_evidence.AppendEvent(
                  "pre_arm_causal_reacquisition",
                  {{"frame_index", frame_index},
                   {"reason", join_failure}});
              std::cerr << "[WAIT] " << join_failure << " frame="
                        << frame_index
                        << "; Unitree mode retained while real-proprio history "
                           "reacquires.\n";
              continue;
            }
            inference_error =
                join_failure + " frame=" + std::to_string(frame_index);
            std::cerr << "PICO/inference fault: " << inference_error << '\n';
            recover_or_latch([](active::GantrySafetyCore& value) {
              value.ObservePicoTermsFailure();
            });
            break;
          }
          proprio = attempt.joined->control_proprio_q10;
          encoder_input = live::BuildCausalEncoderInput(
              attempt.joined->encoder_terms);
        } else {
          const auto latest = monitor.LatestProprio();
          if (!latest.has_value() ||
              now - latest->received_monotonic_ns >
                  active::kStateFreshnessNs) {
            continue;
          }
          proprio = *latest;
          encoder_input = live::BuildEncoderInput(*released_terms);
        }
        live::ProprioSource source{
            .hardware_q = proprio.hardware_q,
            .hardware_dq = proprio.hardware_dq,
            .imu_gyroscope = proprio.gyroscope,
            .imu_quaternion_wxyz = proprio.quaternion_wxyz,
            .previous_action_native = previous_action,
        };
        history.Push(live::BuildProprioFrame(source));
        if (!real_history_gate.Observe(frame_index)) {
          if (real_history_gate.rejected()) {
            inference_error =
                "real_proprio_warmup_rejected frame=" +
                std::to_string(frame_index);
            std::cerr << "PICO/inference fault: " << inference_error << '\n';
            recover_or_latch([](active::GantrySafetyCore& value) {
              value.ObservePicoFrameRegression();
            });
            break;
          }
          continue;
        }
        if (!history.ready()) {
          inference_error =
              "proprio_history_not_ready frame=" +
              std::to_string(frame_index);
          std::cerr << "PICO/inference fault: " << inference_error << '\n';
          recover_or_latch([](active::GantrySafetyCore& value) {
            value.ObservePicoTermsFailure();
          });
          break;
        }
        const auto inference_started_ns = NowNs();
        const auto live_token =
            encoder.Run<true23::kEncoderInputDim,
                        true23::kEncoderOutputDim>(encoder_input);
        const auto decoder_input =
            live::BuildDecoderInput(live_token, history.Flatten());
        const auto decoder_action =
            decoder.Run<true23::kDecoderInputDim,
                        true23::kDecoderOutputDim>(decoder_input);
        if (frozen_lora_policy) {
          double raw_max_abs = 0.0;
          for (const auto value : decoder_action) {
            raw_max_abs = std::max(
                raw_max_abs, std::abs(static_cast<double>(value)));
          }
          if (!std::isfinite(raw_max_abs) || raw_max_abs > 20.0) {
            throw std::runtime_error(
                "raw SONIC decoder action exceeded finite magnitude gate");
          }
        }
        const auto action = frozen_lora_policy
            ? live::RawNativeActionToAppliedSafeNativeAction(decoder_action)
                  .applied_safe_native_action
            : decoder_action;
        const auto produced_ns = NowNs();
        maximum_inference_duration_ns = std::max(
            maximum_inference_duration_ns,
            produced_ns - inference_started_ns);
        bool policy_ready = false;
        monitor.WithCore([&](active::GantrySafetyCore& value) {
          value.SubmitPolicy(
              {.native_action = action,
               .produced_monotonic_ns = produced_ns},
              produced_ns);
          policy_ready = value.policy_ready_for_arm(produced_ns);
        });
        ++accepted_inference_frames;
        if (policy_ready &&
            !first_policy_ready_for_arm.load(std::memory_order_relaxed)) {
          execution_evidence.AppendEvent(
              "first_policy_ready_for_arm",
              {{"real_history_frames", true23::kHistoryLength},
               {"policy_freshness_limit_ns", active::kPolicyFreshnessNs}});
          first_policy_ready_for_arm.store(true, std::memory_order_release);
        }
        previous_action = action;
      }
    } catch (const std::exception& error) {
      std::cerr << "PICO/inference fault: " << error.what() << '\n';
      inference_error = error.what();
      const auto recovery_ns = NowNs();
      monitor.WithCore([&](active::GantrySafetyCore& value) {
        if (!value.BeginSoftwareFaultReturnHold(recovery_ns)) {
          value.ObservePicoTermsFailure();
        } else {
          software_return_requested.store(true, std::memory_order_release);
        }
      });
    }
  });

  std::jthread writer_thread([&](std::stop_token stop_token) {
    const auto request_emergency_handoff = [&](std::string failure) {
      writer_error = std::move(failure);
      bool positive_return_started = false;
      try {
        const auto recovery_ns = NowNs();
        monitor.WithCore([&](active::GantrySafetyCore& value) {
          positive_return_started =
              value.BeginSoftwareFaultReturnHold(recovery_ns);
        });
      } catch (...) {
        writer_error += "; failed to start positive-gain return";
      }
      if (positive_return_started) {
        software_return_requested.store(true, std::memory_order_release);
      }
      // Writer is no longer trustworthy. Never synthesize a damping tail.
      // Main thread immediately re-selects captured Unitree service/FSM.
      mode_handoff_interlock.Request();
      emergency_mode_restore_requested.store(true,
                                             std::memory_order_release);
    };
    try {
      auto next_write_ns = NowNs();
      while (!stop_token.stop_requested() && !stop_threads.load()) {
        if (mode_handoff_interlock.writer_should_quiesce()) {
          break;
        }
        if (motion_mode_restored.load(std::memory_order_acquire)) {
          break;
        }
        if (!motion_mode_released.load(std::memory_order_acquire)) {
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
          next_write_ns = NowNs();
          continue;
        }
        std::this_thread::sleep_until(
            std::chrono::steady_clock::time_point(
                std::chrono::nanoseconds(next_write_ns)));
        active::MotorCommand command;
        active::Fault fault = active::Fault::None;
        std::optional<active::StateSample> command_state;
        bool policy_command = false;
        bool pre_arm_hold_command = false;
        bool normal_return_hold_command = false;
        monitor.WithCore([&](active::GantrySafetyCore& value) {
          const bool armed_before_build = value.armed();
          const bool hold_before_build =
              !armed_before_build && value.pre_arm_hold_prepared();
          const bool normal_return_before_build =
              value.normal_return_active();
          command = value.BuildCommand(NowNs());
          fault = value.fault();
          command_state = value.latest_state();
          policy_command =
              armed_before_build && value.armed() &&
              fault == active::Fault::None;
          pre_arm_hold_command =
              hold_before_build && !value.armed() &&
              !normal_return_before_build && fault == active::Fault::None;
          normal_return_hold_command =
              value.normal_return_active() &&
              fault == active::Fault::None;
          pre_arm_hold_command =
              pre_arm_hold_command && !normal_return_hold_command;
          if (!active::IsPositiveGainRuntimeCommand(command)) {
            rejected_non_positive_gain_commands.fetch_add(
                1, std::memory_order_relaxed);
            throw std::runtime_error(
                "outgoing non-positive-gain LowCmd rejected before DDS");
          }
          try {
            publisher->Write(ToLowCmd(command));
            publisher_write_count.fetch_add(1, std::memory_order_relaxed);
          } catch (...) {
            publisher_write_failed = true;
            throw;
          }
        });
        const auto completed_write_ns = NowNs();
        next_write_ns =
            active::NextNoCatchUpWriterDeadlineNs(completed_write_ns);
        if (pre_arm_hold_command) {
          if (!command_state.has_value()) {
            throw std::runtime_error(
                "pre-arm hold command lacks bound LowState sample");
          }
          for (std::size_t compact = 0; compact < 23; ++compact) {
            const auto slot = static_cast<std::size_t>(
                true23::kHardwareJointIds[compact]);
            const auto target = command[slot].q;
            const auto predicted_effort =
                command[slot].kp * (target - command_state->q[slot]) -
                command[slot].kd * command_state->dq[slot];
            maximum_target_delta_from_state_rad = std::max(
                maximum_target_delta_from_state_rad,
                std::abs(target - command_state->q[slot]));
            maximum_abs_predicted_effort_nm = std::max(
                maximum_abs_predicted_effort_nm,
                std::abs(predicted_effort));
            maximum_abs_feedforward_tau_nm = std::max(
                maximum_abs_feedforward_tau_nm,
                std::abs(command[slot].tau));
            previous_active_targets[compact] = target;
          }
          have_previous_active_targets = true;
          const auto hold_frame = pre_arm_hold_frames.fetch_add(
                                      1, std::memory_order_release) +
                                  1;
          if (hold_frame == 1) {
            first_pre_arm_hold_write_ns.store(
                completed_write_ns, std::memory_order_release);
          }
        }
        if (normal_return_hold_command) {
          if (!command_state.has_value()) {
            throw std::runtime_error(
                "normal return hold command lacks bound LowState sample");
          }
          for (std::size_t compact = 0; compact < 23; ++compact) {
            const auto slot = static_cast<std::size_t>(
                true23::kHardwareJointIds[compact]);
            const auto& joint = command[slot];
            if (joint.mode != 1 || !(joint.kp > 0.0) ||
                joint.tau != 0.0 || !std::isfinite(joint.q)) {
              throw std::runtime_error(
                  "normal return emitted non-hold command");
            }
            maximum_abs_feedforward_tau_nm = std::max(
                maximum_abs_feedforward_tau_nm, std::abs(joint.tau));
          }
          normal_return_hold_frames.fetch_add(1, std::memory_order_release);
        }
        if (policy_command) {
          if (!command_state.has_value()) {
            throw std::runtime_error(
                "policy command lacks bound LowState sample");
          }
          for (std::size_t compact = 0; compact < 23; ++compact) {
            const auto slot = static_cast<std::size_t>(
                true23::kHardwareJointIds[compact]);
            const auto target = command[slot].q;
            maximum_target_delta_from_state_rad = std::max(
                maximum_target_delta_from_state_rad,
                std::abs(target - command_state->q[slot]));
            const auto previous_target = have_previous_active_targets
                                             ? previous_active_targets[compact]
                                             : command_state->q[slot];
            maximum_target_slew_rad = std::max(
                maximum_target_slew_rad,
                std::abs(target - previous_target));
            const auto predicted_effort =
                command[slot].kp * (target - command_state->q[slot]) -
                command[slot].kd * command_state->dq[slot] +
                command[slot].tau;
            maximum_abs_predicted_effort_nm = std::max(
                maximum_abs_predicted_effort_nm,
                std::abs(predicted_effort));
            maximum_abs_feedforward_tau_nm = std::max(
                maximum_abs_feedforward_tau_nm,
                std::abs(command[slot].tau));
            previous_active_targets[compact] = target;
          }
          have_previous_active_targets = true;
          ++policy_command_frames;
          if (policy_command_frames == 1) {
            first_policy_write_ns.store(completed_write_ns);
            execution_evidence.AppendEvent(
                "first_armed_policy_command_written",
                {{"policy_command_frame", 1},
                 {"native_action_dof", 23},
                 {"feedforward_tau_zero", true}});
          }
        }
      }
    } catch (const std::exception& error) {
      request_emergency_handoff(error.what());
    } catch (...) {
      request_emergency_handoff("unknown command-writer exception");
    }
    mode_handoff_interlock.MarkWriterQuiesced();
  });

  std::cout
      << "[PREWARM] Unitree motion mode retained; zero LowCmd writes while "
         "waiting for first fresh 10-frame causal policy.\n";
  while (!first_policy_ready_for_arm.load(std::memory_order_acquire) &&
         !stop_threads.load() &&
         g_stop_requested == 0 && monitor.fault() == active::Fault::None) {
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  if (!first_policy_ready_for_arm.load(std::memory_order_acquire) ||
      stop_threads.load() || g_stop_requested != 0 ||
      monitor.fault() != active::Fault::None) {
    stop_threads.store(true);
    throw std::runtime_error(
        "read-only policy prewarm stopped before motion-mode release");
  }
  bool hold_prepared = false;
  const auto hold_prepare_ns = NowNs();
  monitor.WithCore([&](active::GantrySafetyCore& value) {
    hold_prepared = value.PreparePreArmHold(hold_prepare_ns);
  });
  if (!hold_prepared || publisher_write_count.load() != 0) {
    stop_threads.store(true);
    throw std::runtime_error(
        "pre-arm posture hold was not ready with zero LowCmd writes");
  }
  execution_evidence.AppendEvent(
      "pre_arm_hold_prepared",
      {{"sampled_hardware_joints", 23},
       {"kp_fraction", active::kPreArmHoldKpFraction},
       {"feedforward_tau_zero", true},
       {"pre_release_lowcmd_writes", 0}});
  if (g_stop_requested != 0) {
    stop_threads.store(true);
    throw std::runtime_error(
        "signal stopped controller before motion-mode release");
  }
  const auto released_motion_mode = ReleaseMotionModeAfterGate();
  motion_mode_released_ns = NowNs();
  motion_mode_released.store(true, std::memory_order_release);
  execution_evidence.AppendEvent(
      "motion_mode_released",
      {{"post_release_mode_name_empty", true},
       {"captured_pre_release_form", released_motion_mode.form},
       {"captured_pre_release_name", released_motion_mode.name},
       {"captured_pre_release_fsm_id", released_motion_mode.locomotion_fsm_id},
       {"captured_pre_release_fsm_mode", released_motion_mode.locomotion_fsm_mode},
       {"pre_release_lowcmd_writes", 0},
       {"first_post_release_command", "sampled_posture_hold"}});

  while (pre_arm_hold_frames.load(std::memory_order_acquire) <
             static_cast<std::uint64_t>(kMinimumPreArmHoldCycles) &&
         !stop_threads.load() && g_stop_requested == 0 &&
         monitor.fault() == active::Fault::None) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  const auto first_hold_ns =
      first_pre_arm_hold_write_ns.load(std::memory_order_acquire);
  const auto first_hold_delay_ns =
      first_hold_ns > 0 ? first_hold_ns - motion_mode_released_ns : 0;
  const bool startup_hold_gate_open =
      pre_arm_hold_frames.load(std::memory_order_acquire) >=
          static_cast<std::uint64_t>(kMinimumPreArmHoldCycles) &&
      startup_damping_frames.load(std::memory_order_acquire) == 0 &&
      first_hold_delay_ns >= 0 &&
      first_hold_delay_ns <= kMaximumFirstHoldWriteDelayNs &&
      g_stop_requested == 0 && monitor.fault() == active::Fault::None;
  execution_evidence.AppendEvent(
      startup_hold_gate_open ? "pre_arm_hold_gate_open"
                             : "pre_arm_hold_gate_failed",
      {{"pre_arm_hold_frames", pre_arm_hold_frames.load()},
       {"required_pre_arm_hold_frames", kMinimumPreArmHoldCycles},
       {"startup_damping_frames", startup_damping_frames.load()},
       {"first_hold_write_monotonic_ns", first_hold_ns},
       {"release_to_first_hold_write_ns", first_hold_delay_ns},
       {"maximum_first_hold_write_delay_ns",
        kMaximumFirstHoldWriteDelayNs},
       {"kp_positive", true},
       {"feedforward_tau_zero", true}});
  if (startup_hold_gate_open) {
    monitor.WithCore([](active::GantrySafetyCore& value) {
      value.EnableOperatorArming();
    });
    if (direct_dance) {
      bool direct_armed = false;
      const auto direct_arm_ns = NowNs();
      monitor.WithCore([&](active::GantrySafetyCore& value) {
        value.ObserveOperator(
            {.arm_edge = true,
             .deadman_held = true,
             .stop_requested = false},
            direct_arm_ns);
        direct_armed = value.armed();
        if (!direct_armed) {
          value.ObserveInternalFailure();
        }
        // Record acceptance while core lock still excludes writer. This makes
        // evidence order causal: acceptance always precedes first policy write.
        execution_evidence.AppendEvent(
            direct_armed ? "direct_dance_command_accepted"
                         : "direct_dance_command_rejected",
            {{"command", std::string(kDirectDanceCommand)},
             {"policy_ready", true},
             {"gantry_only", true},
             {"physical_estop_required", true}});
      });
      if (direct_armed) {
        std::cout
            << "[DANCE] Direct command accepted after READY; bounded motion "
               "started. Process signal, B/R2, fault, duration, or physical "
               "e-stop stops.\n";
      }
    } else {
      std::cout
          << "[READY] Sampled posture hold active; fresh policy ready. Release "
             "A if already held, hold L2, then press A once. B/R2 or L2 "
             "release stops.\n";
    }
  } else if (monitor.fault() == active::Fault::None) {
    monitor.WithCore([](active::GantrySafetyCore& value) {
      value.ObserveInternalFailure();
    });
  }
  auto last_operator = monitor.LatestOperator();
  const auto report_operator = [](const active::WirelessOperatorState& state) {
    std::cout << "[REMOTE] L2=" << (state.deadman_held ? "held" : "released")
              << " A=" << (state.arm_pressed ? "pressed" : "released")
              << " STOP=" << (state.stop_pressed ? "pressed" : "released")
              << '\n';
  };
  if (!direct_dance) {
    report_operator(last_operator);
  }
  std::string stop_reason;
  std::int64_t stop_requested_ns = 0;
  bool normal_return_event_written = false;
  bool motion_restore_attempted = false;
  while (!stop_threads.load()) {
    const auto now_ns = NowNs();
    const auto current_operator = monitor.LatestOperator();
    if (!direct_dance &&
        (current_operator.arm_pressed != last_operator.arm_pressed ||
        current_operator.deadman_held != last_operator.deadman_held ||
         current_operator.stop_pressed != last_operator.stop_pressed)) {
      report_operator(current_operator);
      last_operator = current_operator;
    }
    if (g_stop_requested != 0 && stop_reason.empty()) {
      stop_reason = "process_signal";
      stop_requested_ns = now_ns;
      monitor.WithCore([&](active::GantrySafetyCore& value) {
        if (!value.BeginNormalReturnHold(now_ns)) {
          value.Stop();
        }
      });
    }
    const auto first_write = first_policy_write_ns.load();
    if (first_write > 0 && stop_reason.empty() &&
        now_ns - first_write >=
            static_cast<std::int64_t>(arguments.post_arm_duration_seconds) *
                1'000'000'000LL) {
      stop_reason = "reviewed_post_arm_duration_complete";
      stop_requested_ns = now_ns;
      monitor.WithCore([&](active::GantrySafetyCore& value) {
        if (!value.BeginNormalReturnHold(now_ns)) {
          value.Stop();
        }
      });
    }
    bool normal_return_active = false;
    monitor.WithCore([&](active::GantrySafetyCore& value) {
      normal_return_active = value.normal_return_active();
    });
    if (normal_return_active && stop_reason.empty()) {
      stop_requested_ns = now_ns;
      if (software_return_requested.load(std::memory_order_acquire)) {
        stop_reason = "software_transport_normal_return";
      } else {
        stop_reason = current_operator.stop_pressed
                          ? "wireless_operator_stop"
                          : "wireless_deadman_released";
      }
    }
    if (normal_return_active && !normal_return_event_written) {
      execution_evidence.AppendEvent(
          "normal_return_hold_started",
          {{"sampled_hardware_joints", 23},
           {"kp_fraction", active::kPreArmHoldKpFraction},
           {"feedforward_tau_zero", true},
           {"damping_frames_before_return", damping_frames_after_stop}});
      normal_return_event_written = true;
    }
    if (emergency_mode_restore_requested.load(std::memory_order_acquire) &&
        !motion_restore_attempted) {
      motion_restore_attempted = true;
      if (stop_reason.empty()) {
        stop_reason = "writer_emergency_mode_handoff";
        stop_requested_ns = now_ns;
      }
      try {
        mode_handoff_interlock.Request();
        writer_quiesced_before_restore =
            WaitForWriterQuiescence(mode_handoff_interlock);
        if (!writer_quiesced_before_restore) {
          throw std::runtime_error(
              "LowCmd writer did not quiesce before emergency mode restore");
        }
        publisher.reset();
        lowcmd_publisher_closed_before_restore = !publisher;
        const auto restore =
            RestoreMotionModeAfterNormalHold(released_motion_mode);
        restore_select_mode_attempts = restore.select_mode_attempts;
        restore_internal_control_attempts = restore.internal_control_attempts;
        restore_poll_attempts = restore.poll_attempts;
        restore_stable_samples = restore.stable_samples;
        motion_mode_restored.store(true, std::memory_order_release);
        motion_mode_released.store(false, std::memory_order_release);
        execution_evidence.AppendEvent(
            "emergency_motion_mode_restored",
            {{"restored_form", released_motion_mode.form},
             {"restored_name", released_motion_mode.name},
             {"restored_fsm_id", released_motion_mode.locomotion_fsm_id},
             {"restored_fsm_mode", released_motion_mode.locomotion_fsm_mode},
             {"normal_return_hold_frames",
              normal_return_hold_frames.load()},
             {"damping_frames_after_stop", damping_frames_after_stop},
             {"rejected_non_positive_gain_commands",
              rejected_non_positive_gain_commands.load()},
             {"writer_quiesced_before_select", true},
             {"lowcmd_publisher_closed_before_select",
              lowcmd_publisher_closed_before_restore},
              {"select_mode_attempts", restore_select_mode_attempts},
              {"internal_control_handoff", "last"},
              {"internal_control_attempts",
               restore_internal_control_attempts},
              {"restore_poll_attempts", restore_poll_attempts},
              {"stable_restore_samples", restore_stable_samples},
              {"required_stable_restore_samples",
               kMotionRestoreStableSamples}});
      } catch (const std::exception& error) {
        writer_error += std::string("; emergency motion-mode restore failed: ") +
                        error.what();
      }
      stop_threads.store(true, std::memory_order_release);
    }
    if (normal_return_active && !motion_restore_attempted &&
        normal_return_hold_frames.load(std::memory_order_acquire) >=
            static_cast<std::uint64_t>(kNormalReturnHoldCycles)) {
      motion_restore_attempted = true;
      try {
        mode_handoff_interlock.Request();
        writer_quiesced_before_restore =
            WaitForWriterQuiescence(mode_handoff_interlock);
        if (!writer_quiesced_before_restore) {
          throw std::runtime_error(
              "LowCmd writer did not quiesce before motion-mode restore");
        }
        publisher.reset();
        lowcmd_publisher_closed_before_restore = !publisher;
        const auto restore =
            RestoreMotionModeAfterNormalHold(released_motion_mode);
        restore_select_mode_attempts = restore.select_mode_attempts;
        restore_internal_control_attempts = restore.internal_control_attempts;
        restore_poll_attempts = restore.poll_attempts;
        restore_stable_samples = restore.stable_samples;
        motion_mode_restored.store(true, std::memory_order_release);
        motion_mode_released.store(false, std::memory_order_release);
        execution_evidence.AppendEvent(
            "motion_mode_restored",
            {{"restored_form", released_motion_mode.form},
             {"restored_name", released_motion_mode.name},
             {"restored_fsm_id", released_motion_mode.locomotion_fsm_id},
             {"restored_fsm_mode", released_motion_mode.locomotion_fsm_mode},
             {"normal_return_hold_frames",
              normal_return_hold_frames.load()},
             {"required_normal_return_hold_frames",
              kNormalReturnHoldCycles},
             {"startup_damping_frames", startup_damping_frames.load()},
             {"damping_frames_after_stop", damping_frames_after_stop},
             {"writer_quiesced_before_select", true},
             {"lowcmd_publisher_closed_before_select",
              lowcmd_publisher_closed_before_restore},
              {"select_mode_attempts", restore_select_mode_attempts},
              {"internal_control_handoff", "last"},
              {"internal_control_attempts",
               restore_internal_control_attempts},
              {"restore_poll_attempts", restore_poll_attempts},
              {"stable_restore_samples", restore_stable_samples},
              {"required_stable_restore_samples",
               kMotionRestoreStableSamples}});
        stop_threads.store(true, std::memory_order_release);
      } catch (const std::exception& error) {
        writer_error = std::string("motion-mode restore failed: ") +
                       error.what();
        // A failed normal-return handoff is not an actuation fault.  Never
        // replace the positive-gain return hold with an intentional damping
        // tail.  Stop publishing and report failed evidence; the selected AI
        // service or operator retains recovery ownership.
        stop_threads.store(true, std::memory_order_release);
      }
    }
    const auto observed_fault = monitor.fault();
    if (observed_fault != active::Fault::None && stop_reason.empty()) {
      stop_reason = observed_fault == active::Fault::OperatorStop
                        ? "wireless_operator_stop"
                        : std::string(active::FaultName(observed_fault));
      stop_requested_ns = now_ns;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  inference_thread.request_stop();
  writer_thread.request_stop();
  if (inference_thread.joinable()) {
    inference_thread.join();
  }
  if (writer_thread.joinable()) {
    writer_thread.join();
  }
  const auto fault = monitor.fault();
  if (stop_reason.empty()) {
    stop_reason = fault == active::Fault::None
                      ? "writer_stopped_without_latched_fault"
                      : std::string(active::FaultName(fault));
  }
  const bool armed_transition_observed = first_policy_write_ns.load() > 0;
  const bool enough_policy_commands =
      policy_command_frames >= static_cast<std::uint64_t>(
                                   active::kMinimumPromotedShadowActionFrames);
  const bool normal_return_completed =
      normal_return_hold_frames.load() >=
          static_cast<std::uint64_t>(kNormalReturnHoldCycles) &&
      writer_quiesced_before_restore && motion_mode_restored.load() &&
      lowcmd_publisher_closed_before_restore &&
      restore_internal_control_attempts >= 1 &&
      restore_stable_samples >= kMotionRestoreStableSamples &&
      damping_frames_after_stop == 0;
  const bool safe_terminal_fault = fault == active::Fault::None;
  const auto first_write_ns = first_policy_write_ns.load();
  const auto post_arm_elapsed_ns =
      first_write_ns > 0 && stop_requested_ns >= first_write_ns
          ? stop_requested_ns - first_write_ns
          : 0;
  const auto required_post_arm_duration_ns =
      static_cast<std::int64_t>(arguments.post_arm_duration_seconds) *
      1'000'000'000LL;
  const bool reviewed_duration_completed =
      stop_reason == "reviewed_post_arm_duration_complete" &&
      post_arm_elapsed_ns >= required_post_arm_duration_ns;
  const bool intentional_live_stop_completed =
      !direct_dance &&
      (stop_reason == "wireless_operator_stop" ||
       stop_reason == "wireless_deadman_released" ||
       stop_reason == "process_signal") &&
      post_arm_elapsed_ns > 0;
  const bool passed =
      startup_hold_gate_open && armed_transition_observed &&
      enough_policy_commands &&
      normal_return_completed && safe_terminal_fault &&
      (reviewed_duration_completed || intentional_live_stop_completed) &&
      inference_error.empty() && writer_error.empty();
  json terminal = {
      {"passed", passed},
      {"policy_prewarmed_before_motion_release", true},
      {"pre_release_lowcmd_writes", 0},
      {"pre_arm_hold_gate_open", startup_hold_gate_open},
      {"pre_arm_hold_frames", pre_arm_hold_frames.load()},
      {"required_pre_arm_hold_frames", kMinimumPreArmHoldCycles},
      {"startup_damping_frames", startup_damping_frames.load()},
      {"rejected_non_positive_gain_commands",
       rejected_non_positive_gain_commands.load()},
      {"release_to_first_hold_write_ns", first_hold_delay_ns},
      {"maximum_first_hold_write_delay_ns",
       kMaximumFirstHoldWriteDelayNs},
      {"armed_transition_observed", armed_transition_observed},
      {"policy_command_frames", policy_command_frames},
      {"minimum_policy_command_frames",
       active::kMinimumPromotedShadowActionFrames},
      {"damping_frames_after_stop", damping_frames_after_stop},
      {"required_damping_frames_after_stop", 0},
      {"normal_return_hold_frames", normal_return_hold_frames.load()},
      {"required_normal_return_hold_frames", kNormalReturnHoldCycles},
      {"motion_mode_restored", motion_mode_restored.load()},
      {"restored_motion_mode_form", released_motion_mode.form},
      {"restored_motion_mode_name", released_motion_mode.name},
      {"restored_locomotion_fsm_id", released_motion_mode.locomotion_fsm_id},
      {"restored_locomotion_fsm_mode", released_motion_mode.locomotion_fsm_mode},
      {"maximum_target_delta_from_state_rad",
       maximum_target_delta_from_state_rad},
      {"maximum_target_slew_rad", maximum_target_slew_rad},
      {"maximum_abs_predicted_effort_nm",
       maximum_abs_predicted_effort_nm},
      {"maximum_abs_feedforward_tau_nm",
       maximum_abs_feedforward_tau_nm},
      {"final_fault", std::string(active::FaultName(fault))},
      {"stop_reason", stop_reason},
      {"post_arm_elapsed_ns", post_arm_elapsed_ns},
      {"required_post_arm_duration_ns", required_post_arm_duration_ns},
      {"inference_error", inference_error},
      {"writer_error", writer_error},
      {"publisher_write_failed", publisher_write_failed},
      {"writer_quiesced_before_restore", writer_quiesced_before_restore},
      {"lowcmd_publisher_closed_before_restore",
       lowcmd_publisher_closed_before_restore},
      {"restore_select_mode_attempts", restore_select_mode_attempts},
      {"restore_internal_control_handoff", "last"},
      {"restore_internal_control_attempts",
       restore_internal_control_attempts},
      {"restore_poll_attempts", restore_poll_attempts},
      {"stable_restore_samples", restore_stable_samples},
      {"required_stable_restore_samples", kMotionRestoreStableSamples},
      {"publisher_write_count", publisher_write_count.load()},
      {"accepted_inference_frames", accepted_inference_frames},
      {"maximum_inference_duration_ns", maximum_inference_duration_ns},
      {"maximum_packet_age_ns", maximum_packet_age_ns},
  };
  if (!armed_transition_observed) {
    execution_evidence.Finalize("session_no_actuation", terminal);
    std::cerr << "[NO ACTUATION] no armed policy command was written; evidence: "
              << execution_evidence.path() << '\n';
    return 3;
  }
  if (!passed) {
    execution_evidence.Finalize("session_failed", terminal);
    std::cerr << "[STOPPED] stage-one evidence failed: fault="
              << active::FaultName(fault)
              << " policy_command_frames=" << policy_command_frames
              << " damping_frames=" << damping_frames_after_stop << '\n';
    return 2;
  }
  execution_evidence.Finalize("session_complete", terminal);
  std::cout << "[COMPLETE] policy_command_frames=" << policy_command_frames
            << " normal_return_hold_frames="
            << normal_return_hold_frames.load()
            << " motion_mode_restored=" << std::boolalpha
            << motion_mode_restored.load()
            << " evidence=" << execution_evidence.path() << '\n';
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto arguments = ParseArguments(argc, argv);
    if (arguments.help) {
      std::cout << Usage(argv[0]) << '\n';
      return 0;
    }
    std::signal(SIGINT, HandleSignal);
    std::signal(SIGTERM, HandleSignal);
    return Run(arguments);
  } catch (const std::exception& error) {
    std::cerr << "[BLOCKED] true23 active gantry controller: "
              << error.what() << '\n';
    if (argc > 0) {
      std::cerr << Usage(argv[0]) << '\n';
    }
    return 1;
  }
}
