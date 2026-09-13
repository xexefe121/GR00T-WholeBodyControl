// Read-only arithmetic oracle. No SDK, DDS, motor or robot connection.
#include <vector>
#include <iostream>
#include <iomanip>
#include "policy_parameters.hpp"

template <typename T, size_t N>
void array_json(const std::array<T, N>& values) {
    std::cout << "[";
    for (size_t i = 0; i < N; ++i) {
        if (i) std::cout << ",";
        std::cout << values[i];
    }
    std::cout << "]";
}

int main() {
    std::cout << std::setprecision(17) << "{\"scale_hardware29\":";
    array_json(g1_action_scale);
    std::cout << ",\"default_hardware29\":";
    array_json(default_angles);
    std::cout << ",\"source_il29_for_hardware29\":";
    array_json(isaaclab_to_mujoco);
    std::cout << ",\"kp_hardware29\":";
    array_json(kps);
    std::cout << ",\"kd_hardware29\":";
    array_json(kds);
    std::cout << "}\n";
}
