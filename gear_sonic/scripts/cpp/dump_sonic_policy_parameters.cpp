// Offline arithmetic witness only. No Unitree SDK, network or robot commands.
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>
#include "policy_parameters.hpp"

template <typename T, std::size_t N>
void array_json(const std::array<T, N>& values) {
  std::cout << '[';
  for (std::size_t i = 0; i < N; ++i) {
    if (i) std::cout << ',';
    std::cout << values[i];
  }
  std::cout << ']';
}

int main(int argc, char** argv) {
  std::cout << std::setprecision(std::numeric_limits<double>::max_digits10);
  if (argc == 2 && std::strcmp(argv[1], "--targets") == 0) {
    std::array<float, 29> actions{}, targets{};
    for (auto& action : actions) {
      if (!(std::cin >> action) || !std::isfinite(action)) return 2;
    }
    // Same scalar expression and final float cast as CreatePolicyCommand.
    for (int i = 0; i < 29; ++i) {
      const double action_value = static_cast<double>(actions[isaaclab_to_mujoco[i]]) * g1_action_scale[i];
      targets[i] = static_cast<float>(default_angles[i] + action_value);
    }
    array_json(targets);
    std::cout << '\n';
    return 0;
  }
  if (argc != 1) return 2;
  std::cout << "{\"kind\":\"g1_sonic_cpp_policy_parameters_v1\",\"kps\":";
  array_json(kps);
  std::cout << ",\"kds\":";
  array_json(kds);
  std::cout << ",\"action_scale\":";
  array_json(g1_action_scale);
  std::cout << ",\"default_angles\":";
  array_json(default_angles);
  std::cout << ",\"isaaclab_to_mujoco\":";
  array_json(isaaclab_to_mujoco);
  std::cout << ",\"mujoco_to_isaaclab\":";
  array_json(mujoco_to_isaaclab);
  // These are header motor constants, NOT a C++ runtime torque-clip array.
  std::cout << ",\"motor_effort_constants\":";
  array_json(std::array<double, 4>{EFFORT_LIMIT_5020, EFFORT_LIMIT_7520_14,
                                  EFFORT_LIMIT_7520_22, EFFORT_LIMIT_4010});
  std::cout << "}\n";
}
