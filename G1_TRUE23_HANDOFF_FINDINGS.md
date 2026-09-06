# True23 handoff: reproduced defects and next integration boundary

Status, 2026-09-07: **not ready for physical dance or live full-body teleop**.
This work changes the offline qualification check, not the hardware controller.
No robot, DDS, SSH, motor, arming or mode operation was performed. A local
network-adapter check reports robot Ethernet disconnected; it supplies no
current robot/firmware observation. No additional policy training was started.

## What the actual restore code does

The current, pre-existing hardware worktree closes its `rt/lowcmd` publisher,
then selects the captured motion service and requests internal WALKRUN control.
If the observed FSM remains 0 or 1, its normal-return routine calls the same
recovery sequence used after faults: FSM `1 → 4 → 801`.

An offline fixture compiles those exact C++ restore functions and their real
evidence gate. Only RPC responses, LowState input and the clock are substituted.
It does not copy the control algorithm into a separate implementation.

Reproduced outcomes:

- In the injected stand-RPC failure, `SetFsmId(1)` succeeds, `SetFsmId(4)` fails
  with the injected 3104 response, and a second attempt does the same. The
  routine exits unsuccessfully with the mock FSM still 1. The recorded calls
  are `1, 4, 1, 4`, not a completed standing handoff.
- The successful mock recovery has a **6.5 s gap between the restore routine's
  physical-state reads**, even with instantaneous RPCs. This does not mean
  the separate DDS subscriber thread stopped; that thread is not executed by
  this fixture. A persistent injected disable is rejected at final evaluation,
  but two stand requests are made before the routine reads that failure.
- Current enabled-motor validation correctly rejects disabled, stale,
  frozen-tick, bad-CRC, nonfinite and zero-torque final evidence. It can still
  accept FSM 801 with both knees at 1.5 rad: enabled telemetry is not normal
  standing, weight bearing or balance.
- The committed pre-existing restore code from `64f2bc3` has no physical-state
  gate. The same fixture exposes its FSM-only false successes and its failure
  to recover from FSM 0 when the accepted internal-control RPC has no effect.

These are software behaviors under explicit injected conditions. They do
**not** establish which condition occurred in any past physical session, why
the physical bit-30 latch occurred, or whether real firmware permits a given
transition. Four historical positive LowCmd damping tails and later FSM
recovery must not be conflated with that separate latched-driver incident.

## Qualification correction

`qualify_g1_true23_active_lifecycle_no_robot.py` now runs this RPC-path audit,
not just the dependency-light core harness and a binary/string surface check.
The CLI remains offline. Its output schema is now 2; it creates a new adjacent
`<output>.restore_rpc/` evidence directory and returns 2 when the normal-standing,
no-damp contract fails. Direct Python callers must supply `rpc_audit_directory`.
This is not a replacement for the launcher's existing safety gates or physical
qualification, and does not by itself disable an already authorized launcher.

On the same current binaries/source, the old qualifier returns success; the
updated qualifier returns failure. Core and binary/source checks still pass.
The additional failures are explicit damp requests, stand requests after
injected motor disable, and accepting a crouched mock state as normal standing.
The audit's own `experiment_completed=true` is not a qualification pass.

Verification: **26 tests pass against each of the two source versions**, with
no skipped scenario tests. The committed source/header snapshots were checked
against their exact Git blob IDs. Ruff E/F and formatting checks pass. Current
hardware-source bytes and all tested source hashes remain unchanged during
the completed checks. Existing hardware edits are not staged or promoted here.

Evidence root: `artifacts/g1_true23_frozen_lora/restore_rpc_20260907_v1/validated/`.
Key files:

- `verification.json`: `c77e90a6bda8d6e3924da5d496c40b0543901ae60feee58fef7a81c481f8b3b1`
- `qualification_after.json`: `54608a7d43a8defcf49db8718baeb647824c72a6f91315dce2fdb225645d9087`
- `qualification_after.json.restore_rpc/report.json`: `6aa6279a72cfa9975470ffc0f26514b20be756aa12a43e7b8db14d2f323dcbc5`

The initial fixture compile error, an incorrect test expectation at a virtual
time boundary, and a lint-only failed verification start were corrected before
these results. Available earlier failed logs remain separate; no failed run
is counted as a completed qualification.

## A different official full-body transport is relevant, not yet qualified

Unitree's [pinned user-control example](https://github.com/unitreerobotics/unitree_sdk2/blob/30405b31d82f137d48f33cbba095d149749db601/example/g1/high_level/g1_userctrl_dds_example.cpp)
uses `rt/user_lowcmd`, enters through `SwitchToUserCtrl()`, and returns through
`SwitchToInternalCtrl(LAST)` without releasing/reselecting the service. It
commands legs, waist and arms; this is not the upper-body-only `rt/arm_sdk`
approach. Our active runtime uses `rt/lowcmd` and `ReleaseMode()` and never
enters that user-control protocol. The vendored SDK already exposes both RPCs.

Important limit: the example **requires starting in passive FSM 1**, sends an
initial zero-kp packet and returns to LAST. It does not demonstrate standing
801 → user control → standing 801, does not qualify this robot's firmware or
23-axis configuration, and must not be run unchanged as a dance/standing test.
The [official client](https://github.com/unitreerobotics/unitree_sdk2/blob/main/include/unitree/robot/g1/loco/g1_loco_client.hpp)
also maps `Damp()` to FSM 1; its RPC return code alone supplies no physical
restoration evidence.

Inference: the retained-service full-body transport is the next architecture
to evaluate for normal handoff. Merely changing a topic string, removing damp
calls, retaining two competing writers, or returning LAST from a passive start
is not a proven fix. No transport migration or hardware mode call is performed.

Before wiring an executable candidate into the live launcher, establish:

1. Firmware support and observable ownership acknowledgment for 7110/7111,
   including whether entry from standing 801 and return to it are supported.
2. User-topic command semantics for this true23 machine: controlled/absent
   slots, PR mode, motor mode fields, CRC, watchdog and command-age behavior.
3. The supported writer/handoff ordering and behavior on RPC rejection,
   timeout, lost input, operator stop and a raw motor-disable edge. Normal
   completion and fault recovery must remain distinct.
4. Continuous raw motor/voltage/temperature/IMU/command/ownership evidence
   across the transfer, plus observed physical standing. The existing latest-
   sample health probe cannot reconstruct a past fault edge.

Separately, the native23 policy still completes only **2.06/10.7 s** of the
historical-start dance in the constrained simulator and has no successful
full-dance return. Fixing transport will not train that policy, establish
live PICO calibration, or make the six absent axes physically available.
