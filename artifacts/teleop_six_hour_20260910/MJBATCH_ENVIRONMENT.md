Isolated CPU batch simulator installed, 2026-09-10 UTC.

Python: `/root/.venvs/g1_true23_mjbatch/bin/python` in WSL Ubuntu-22.04.
Python 3.11.15; mjbatch 0.1.0; MuJoCo 3.11.0; NumPy 2.4.6;
SciPy 1.17.1. Existing GPU training environment remains MuJoCo 3.5.0.
No packages were installed into the training environment.

Upstream source: https://github.com/kevinzakka/mjbatch, local source directory
`artifacts/teleop_six_hour_20260910/mjbatch_source`. Its build and runtime pin
MuJoCo 3.11.0. Upstream explicitly skips Windows wheels because the MuJoCo
Windows wheel lacks an import library and the extension uses rpath.

Installation, already completed:

```sh
/root/.local/bin/uv venv --python /root/.venvs/g1_true23_mjlab/bin/python /root/.venvs/g1_true23_mjbatch
CMAKE_BUILD_PARALLEL_LEVEL=2 /root/.local/bin/uv pip install --python /root/.venvs/g1_true23_mjbatch/bin/python /mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/mjbatch_source pytest scipy
```

Native model loader available without Torch:
`gear_sonic.scripts.benchmark_g1_true23_mjbatch.prepare_native()`.
It applies the existing native armature, damping, friction, joint effort caps
and `mj_setConst`, using the original referee XML under
`/mnt/z/codex/GR00T-WholeBodyControl/gear_sonic/data/robots/g1/`.

API checked:

```python
from mjbatch import Batch
batch = Batch(model, num_sims=64, num_threads=8)
qpos, qvel, ctrl = (batch.bind(k) for k in ("qpos", "qvel", "ctrl"))
qpos[:] = initial_qpos
qvel[:] = initial_qvel
batch.forward()
ctrl[:] = torque
batch.step()               # One 2 ms step per simulator.
batch.step(nstep=10)       # Ten steps with held ctrl; Python PD is not recomputed.
state = batch.bind("state")  # Integration-state rows for copying/restoring.
```

`forward=True` adds `mj_forward` after each step call for synchronized derived
fields. Default derived fields have the same one-substep age as `mj_step`.
The model is copied during Batch construction: apply actuator changes first.

Validation: eight native23 states, 100 physics substeps, PD recomputed at 500 Hz
with targets held at 50 Hz. Batch and single MuJoCo 3.11 qpos, qvel, ctrl,
actuator forces and time all bit-exact. Twelve upstream state/lockstep/history/
forward tests also passed.

Measured during concurrent GPU training and CPU work, three repeats, 300 steps:

| Batch size | Threads | Physics steps/s | Speedup over serial |
| --- | --- | --- | --- |
| 1 | 1 | 11,341 | 1.11x |
| 8 | 8 | 13,108 | 1.24x |
| 32 | 8 | 32,758 | 3.76x |
| 64 | 8 | 43,836 | 5.15x |

This includes Python PD recomputation each 2 ms; it excludes policy inference.
An affine position-servo model may hold joint targets through `nstep=10` while
MuJoCo recomputes actuator PD internally, provided its torque/state parity is
verified independently.

Evidence: `mjbatch_native23_benchmark_v1/report.json`,
`mujoco350_native23_compare_v1/report.json`, and
`mjbatch_original_xml_parity_v2/report.json`.
The initial benchmark used the checkout XML (SHA38d6...), which differs from
the original referee XML (SHA16e304...) only in CRLF/LF line endings. The v2
check uses the exact original XML. No model geometry was changed.

MuJoCo 3.5 versus 3.11 on the short deterministic trace differs slightly:
max qpos 6.83e-7, qvel 9.62e-5, torque 1.81e-4. This is cross-version evidence,
not a claim of Windows-engine equivalence or full-horizon dynamics qualification.
