Isolated mjbatch0.1.0 build for the native MuJoCo3.2.3 referee passes physics conformance. Existing MuJoCo3.11 mjbatch and3.5 training environments/source remain unchanged.

Windows location: E:\codex_sonic_runtime\mjbatch323_20260910 . Source, virtual environment, build, cache and temporary files are E-backed. The WSL mount may disappear when the distribution exits; start each fresh invocation with:

```
mountpoint -q /mnt/e || mount -t drvfs E: /mnt/e
```

Python: /mnt/e/codex_sonic_runtime/mjbatch323_20260910/venv/bin/python . Versions: Python3.11.15, MuJoCo3.2.3, mjbatch0.1.0, NumPy1.26.4, SciPy1.15.3. Source commit77966f85bcd8f7ef4351cb4a1a6f42e133d19725.

Changes affect only the copied pyproject.toml and C++ batch.h API compatibility branches: MuJoCo version pin, isolated build directory, older field/option macros and scalar types, older fatal-error callback, and absent newer sleep/mesh scalar fields. No MuJoCo engine source or native model physics changed. A toy upstream test keyframe now explicitly includes its third zero joint value, because3.2.3 rejects the newer implicit-padding form. Exact patch, input/output hashes, linked library and test results are in E:\codex_sonic_runtime\mjbatch323_20260910\build_receipt.json and compatibility.patch.

Native torque Batch versus ordinary MuJoCo3.2.3: eight states ×100 physical2ms steps; qpos, qvel, commanded torque, actuator torque and time are bit-exact. Affine PD versus manual clipped PD:128 varied clipped/saturated one-step cases and eight simulations ×1000 physical steps. Worst continuous errors: qpos2.69e-14, qvel3.36e-12, torque4.39e-12; time exact. Every original physical model field is preserved. Results: native_torque_parity/report.json and servo_conformance323.json under the E-backed build directory.

Upstream tests:22 passed,16 deselected, including state restoration, substep history, lockstep simulation, per-simulation options and worker error recovery. Initial10 fixture setup errors were the toy keyframe padding issue, corrected only in the source copy. This is numerical engine/batch conformance, not robot tracking or deployment qualification.

Build command used the E-backed environment with existing build dependencies:

```
UV_CACHE_DIR=/mnt/e/codex_sonic_runtime/mjbatch323_20260910/uvcache TMPDIR=/mnt/e/codex_sonic_runtime/mjbatch323_20260910/tmp CMAKE_BUILD_PARALLEL_LEVEL=4 /root/.local/bin/uv pip install --python /mnt/e/codex_sonic_runtime/mjbatch323_20260910/venv/bin/python --no-build-isolation --no-deps /mnt/e/codex_sonic_runtime/mjbatch323_20260910/source
```
