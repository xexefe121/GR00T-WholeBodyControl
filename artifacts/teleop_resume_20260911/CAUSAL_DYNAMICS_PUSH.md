# Accepted full-body causal dynamics plan — implementation

Started 2026-09-12 02:46:40 UTC. Six-hour decision deadline 08:46:40 UTC.
Laptop only. No paid compute, headset, robot, DDS, or physical control.

New code: received-only 1323-feature builder, reuse of native23 PacketGate,
generated input-loss standing references, full-range 23-target actor initialized
from earlier81000 checkpoint, MJLab Simulation/Warp dynamics PPO, native3.2.3
full-motion evaluator. Original tracking and physical thresholds unchanged.

Data: E:/codex-artifacts/sonic23_teleop_resume_20260911/causal_dynamics_v1/bank.
8666 existing actual expert rows across walk003/walk002/Pico. walk008 is held
out. Six additional generated stop-reference variants preserve original motion
prefixes; original evaluation references remain unchanged. Actor's eight goal
slots are current/past, with backward-only velocities and measured robot history.

Initial native evaluation of old81000 with new causal semantics and learner
controlling standing fails at control23, before source, for all four motions.
This is the new unadapted baseline, not the earlier BFM-assisted5.46s result.
Smoke1 exposed oversized optimizer updates. Smoke2 rolls back actor updates
whose analytic sampled-batch KL exceeds.03, with explicit reduced learning rate.
Receiver checks cover streaming/offline features, unseen-future mutation,
packet-loss fault latching, and explicit body/task calibration on rearm.
Smoke CPU/GPU input checks pass. Not readiness.

Pilot_v1:128 GPU environments,64 rollout steps,1000 short causal adaptation
updates, then bounded800 PPO updates/3h. Checkpoints400/800 get complete native
motion and standing evaluation. Pilot waits for initial bootstrap evaluation
before PPO. Create its CONTINUE file only after inspecting that result. STOP
file requests a checkpointed stop at next PPO update boundary.

Primary output: E:/codex-artifacts/sonic23_teleop_resume_20260911/causal_dynamics_v1.
Previous successful prepared feedback is preserved. Independent-clock runtime
integration is implemented in gear_sonic/native/true23_clock.cpp and the local
run_native_clock.py runner. See CURRENT.md for nominal success and negative
disturbance failure. Root owns all code/processes; no new agents started.

Reusable API: gear_sonic.utils.g1_true23_causal_controller.CausalController.
receive() accepts the existing Packet plus three hand/head task poses;
command(qpos30,qvel29,now) returns ControllerCommand(targets23,status).
commit_applied() records the command actually applied once per control tick;
rearm() is explicit. No simulation step or clock advance occurs in this API.
The evaluator uses it; checkpoint400 targets and physics are exactly equal
to the preceding inline evaluator. Initialization validates native target
limits/default pose. Candidate policies remain explicitly unqualified.

Run received-only simulation from PowerShell:

```powershell
& Z:\codex\GR00T-WholeBodyControl-sonic-transfer-23dof\artifacts\teleop_resume_20260911\RUN_CAUSAL_FULL_BODY_SIM.ps1 -Actor E:\codex-artifacts\sonic23_teleop_resume_20260911\causal_dynamics_v1\pilot_v1\actor_00400.onnx
```

Default runs all four full motions with 30s hold. A behavioral failure raises
an error after saving reports. -InitialVelocity accepts a declared X velocity
offset; -FaultControl stops received packets at the chosen control. Source
tracking and generated-stop verdicts remain separately labeled.

Prepared independent-clock benchmark:

```powershell
& Z:\codex\GR00T-WholeBodyControl-sonic-transfer-23dof\artifacts\teleop_resume_20260911\RUN_NATIVE_CLOCK_SIM.ps1
```

Uses existing Ubuntu-22.04 MuJoCo3.2.3 runtime, pinned native model, warmed
standing process, 500Hz plant and 50Hz control. Pauses only this exact owned
training pilot while timing runs, resumes it in finally. Reports misses even
when physics catches up. Requires prepared walk003 gains and simulator root
state. No readiness promise follows from a successful prepared benchmark.
