`zmq_manager` does not exist and was not built. This checkout defines no executable target with that name. The executable that implements `--input-type zmq_manager` was built successfully: `/root/builds/teleop23_20260915/source/target/release/g1_deploy_onnx_ref`.

## What the source builds

`gear_sonic_deploy/deploy.sh` parses `--input-type zmq_manager`, then always builds with `just build` and launches `just run g1_deploy_onnx_ref ... --input-type zmq_manager`. It does not execute `target/release/zmq_manager`.

The CMake executable target is `g1_deploy_onnx_ref`, defined in `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/CMakeLists.txt`. The `zmq_manager` name is an input-interface value parsed by `src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp`; its implementation is the header-only `include/input_interface/zmq_manager.hpp`. `cmake --build ... --target help` listed `g1_deploy_onnx_ref` and no `zmq_manager` target.

The main target links `g1_deploy_shared`, `audio_thread`, vendored `unitree_sdk2`, Zlib, ONNX Runtime, `TRTInference`, CUDA runtime, TensorRT `nvinfer` and `nvonnxparser`, and libzmq. It also needs Eigen3 and msgpack headers. ROS 2 is conditional: with `HAS_ROS2` unset or `0`, CMake does not call `find_package(rclcpp)` and builds without `ROS2InputHandler`. Unitree SDK imports its bundled CycloneDDS libraries (`libddsc.so` and `libddscxx.so`); it does not require a separately installed CycloneDDS or Fast DDS package.

## Dependency check

All dependencies needed by `g1_deploy_onnx_ref` are present in WSL Ubuntu-22.04 (x86_64):

- CMake 3.22.1, GCC 11.4.0, Clang 14.0.0, Ninja 1.10.1, and `just` 1.43.0.
- CUDA Toolkit 12.9.86 at `/usr/local/cuda-12.9`; GPU is an RTX 3070 under driver 576.02.
- TensorRT 10.13.3.9 for CUDA 12.9: headers in `/usr/include/x86_64-linux-gnu`, `libnvinfer.so.10`, and `libnvonnxparser.so.10` in `/usr/lib/x86_64-linux-gnu`.
- ONNX Runtime C++ 1.16.3: headers in `/opt/onnxruntime/include`, library `/usr/local/lib/libonnxruntime.so` resolving to `/opt/onnxruntime/lib/libonnxruntime.so.1.16.3`. The existing Python venv has ONNX Runtime 1.23.2; that Python package is not the C++ build dependency.
- `libzmq3-dev` 4.3.4, `libmsgpack-dev` 3.3.0, `libeigen3-dev` 3.4.0, and `zlib1g-dev` 1.2.11.
- No ROS 2, CycloneDDS, or Fast DDS Debian packages are installed. This does not block this target: CMake configured with `ROS2 disabled` and imported the vendored Unitree/CycloneDDS libraries.

The TensorRT finder reports an absent `nvparsers` component, but configuration succeeds and `g1_deploy_onnx_ref` does not link it. No missing package blocks this C++ build.

## Build performed

Build cache and copied source were kept on WSL ext4, not on `Z:`. Save this script as `/root/builds/teleop23_20260915/build_g1_deploy_onnx_ref.sh` inside WSL:

```bash
#!/usr/bin/env bash
set -euo pipefail
mkdir -p /root/builds/teleop23_20260915
rsync -a --exclude build --exclude target \
  /mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/gear_sonic_deploy/ \
  /root/builds/teleop23_20260915/source/
export CUDAToolkit_ROOT=/usr/local/cuda-12.9
export CUDA_HOME=/usr/local/cuda-12.9
cmake -S /root/builds/teleop23_20260915/source \
  -B /root/builds/teleop23_20260915/cmake-release \
  -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SRCS=ON \
  -DCMAKE_BUILD_RPATH=/usr/local/lib
cmake --build /root/builds/teleop23_20260915/cmake-release \
  --target g1_deploy_onnx_ref --parallel 2
```

Then run it from Windows exactly as required:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash -lc 'bash /root/builds/teleop23_20260915/build_g1_deploy_onnx_ref.sh'
```

The equivalent commands completed here. Output was a 5,517,080-byte x86-64 ELF executable at `/root/builds/teleop23_20260915/source/target/release/g1_deploy_onnx_ref`. `g1_deploy_onnx_ref --help` exited successfully and lists `zmq_manager` among accepted input types. No hardware or headset was contacted.

`-DCMAKE_BUILD_RPATH=/usr/local/lib` is intentional. Windows checkout handling leaves the vendored Unitree `libddsc.so.0` SONAME link as a text file. Without the RPATH override, the loader selects that bad path and exits 127 with `file too short`. With the override, `/usr/local/lib/libddsc.so.0` is selected first and `--help` succeeds without setting `LD_LIBRARY_PATH`.

## Smallest path forward

There is no dependency-install step needed to build the existing deployment program. For the documented PICO mode, use the built `g1_deploy_onnx_ref` program with `--input-type zmq_manager`; do not look for a separate `zmq_manager` executable in this revision.

If an external tutorial truly requires a process named `zmq_manager`, that is a source/documentation mismatch rather than a missing package. The repository contains no target, source file, or CLI contract for such a standalone program. Do not create a renamed copy of `g1_deploy_onnx_ref` without confirming that external contract. The smallest code change, if a filename alias is explicitly required, would be to add a second CMake executable target using the same `g1_deploy_onnx_ref.cpp` entry point and link set; no dependency installation would be involved.

One launcher caveat remains: the `just build` recipe used by `deploy.sh` configures `gear_sonic_deploy/build` in the mounted repository, which is on `Z:`. That conflicts with the low-free-space constraint. Use the ext4 CMake build above until the launcher is changed to point its build directory at `/root/builds/teleop23_20260915/cmake-release`. No repository source file was modified in this work.
