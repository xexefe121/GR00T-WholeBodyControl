# Teleop on the 23-DoF G1 — state, and what not to try again

> **READ FIRST (2026-09-17): the working controller is BFM-Zero 23-DoF, not SONIC.**
> Plain BFM-Zero (no trained residual), with IMU-based root odometry, completes
> every reference clip — including walk003 and the full 115.6 s PICO capture,
> which every SONIC-derived checkpoint fails — with arm joint RMSE of 0.04–0.13 rad
> against SONIC's 0.51–0.58. It was found and evaluated on 2026-09-10, then
> abandoned when a residual trained on top of it failed. It has now been
> reproduced exactly and wired into a two-process ZMQ teleop simulation with
> sensor-only root. **Start from `artifacts/bfm_teleop_20260917/PROGRESS.md`.**
> Everything below about SONIC remains accurate as history, but SONIC is not the
> path.

Last verified 2026-09-15. Read this before starting work on teleop.

## What runs today

```
wsl.exe -d Ubuntu-22.04 -- bash /root/run_teleop_sim.sh [output-directory]
```

Full-body 23-DoF teleop in simulation: recorded PICO source, real localhost ZMQ
transport, MuJoCo physics. Verified repeatedly, including after every change in
this session:

```
passed                       True
completed_live_transitions   535 / 535
fallback_transition_count    0
maximum_reference_age_ns     25,460,350   (gate 100,000,000)
live_transport_proven        True
```

Transport faults are handled: `timeout`, `gap` and `stale` each degrade to a
stable balance hold rather than ending the session (`/root/fault_tests.sh`).

Copies of both scripts are in this directory.

**Long jobs: keep a `wsl.exe` session attached for the whole run.** The WSL
distro shuts itself down shortly after the last `wsl.exe` session detaches, and
that stops *every* process in it, including `systemd-run` units. `nohup` does not
help either. Run the job in the foreground of a `wsl.exe` call that stays open
for its duration. (An earlier version of this note recommended `systemd-run`;
that was wrong. It only appeared to work because a separate `tail -F` monitor
happened to be holding a session open. Corrected 2026-09-17 after a training
unit was stopped 41 s in, with the journal showing repeated distro boots.)
Enabled services such as `roboticsservice`, and the `E:` fstab mount, do come
back automatically on each boot.

## Do not repeat these

Each was tried, by earlier Codex sessions or this one, and failed.

1. **Another PPO run on the existing recipe.** Four attempts now. The
   2026-09-13 session concluded "simply starting another PPO run isn't a
   credible plan"; this session then did it anyway and stopped at update 400 of
   2000 with `root_error_mean_m` 0.1249 against 0.1037 at update 1 — worse than
   its own starting point after 2.46M transitions.
2. **Arm-only repair of the PICO reference.** The blocking contacts are leg and
   torso geometry: knee/knee 22.657 mm, pelvis/hip-roll 15.365 mm, 53 frames
   that arm motion cannot change. Every arm-only attempt stalled; the best
   fragment hit 20.19 rad/s and still exceeded the speed envelope by 1.668x
   rejoining the source.
3. **The released-SONIC C++ deploy path, for this robot.** `pico_g1_preflight.py`
   accepts only `mode_machine` 2 and 5, both 29-DoF, and explicitly rejects
   mode 4 — which is this robot. It also crashed twice in TensorRT here and
   filled `Z:` to zero bytes doing it.
4. **More evidence/audit machinery.** There is already a great deal. None of it
   improved tracking.
5. **Editing `promotion_enabled` or an approval hash to pass a gate.** That
   manufactures qualification for a humanoid that has already collapsed under
   its own weight at 0.6 policy authority.

6. **Training the frozen-LoRA recipe on feasible references only** (2026-09-17).
   Pre-registered test, verdict KILL: on held-out walk002 the new checkpoint was
   better on all23 RMSE and right hand, worse on left hand, with every difference
   under 3 percent. The reason is visible in both runs' scalars: under
   `native_support_stateful_v2`, `stage_one_actuation_guard` ends episodes after
   about **four control steps (80 ms)**, reward sits near −99 throughout, and the
   policy barely leaves its warm start. No data, learning-rate or iteration change
   gets past that. The next step is diagnosing why the guard fires immediately,
   not another run. See `../teleop_feasible_training_20260917/RESULT.md`.

## The three real problems, separated

They were being conflated. They are independent.

1. **The PICO capture is geometrically infeasible.** 498 self-collision frames,
   maximum penetration 83.320 mm, and 53 frames of fixed-body contacts. The
   three public TWIST2 walks have none. Evidence:
   `artifacts/teleop_six_hour_20260910/REFERENCE_FLOOR_AND_SELF_CONTACT_AUDIT.md`.
2. **The offline evaluation has no locomotion input.** Teleop encoder mode takes
   lower-body joint targets, pelvis-relative VR three-point targets and anchor
   orientation — no root position or world velocity. Locomotion comes from the
   planner, which the saved-clip path never runs. Base-position errors of metres
   against a walking reference are substantially an artifact of this.
3. **Arm tracking is genuinely weak.** On clean TWIST2 references: arm joint
   RMSE 0.51-0.58 rad against leg 0.23-0.28, and hand p95 0.45-0.76 m against
   the 0.055 m a prepared positive control achieves. This one is real and is not
   explained away by 1 or 2.

## Hard external dependencies

No amount of local work removes these.

- **No PICO headset has ever connected.** Three of the eight readiness NO-GO
  reasons depend on it. `pico_manager_thread_server.py` hard-rejects non-XRT
  input, so recordings cannot substitute.
- **The robot is not cabled** (both host Ethernet adapters disconnected).
- **`trained_checkpoint` requires a real training run.** All 30 checkpoints on
  disk were validated and rejected: 28 lack the true23 schema header, 2 are
  initialization-only. No `*.promotion.pt` exists anywhere.

## Environment facts that bite

- **`Z:` has under 400 MB free.** Never build, cache or train there. TensorRT
  writes ~400 MB engines beside the source ONNX.
- **`core.autocrlf=true`** breaks hash-pinned gates: files check out CRLF while
  approvals record LF hashes. `.gitattributes` now pins `eol=lf` for the MuJoCo
  sim2sim approval files; other pinned text files remain exposed.
- **Variable expansion is unreliable** through `wsl.exe -- bash -lc '...'`. Use
  script files and literal paths; `$VAR` silently arrives empty.
- **`E:` was not mounted in WSL** until this session. `/mnt/e_stray` holds
  earlier output written when it was not. Any report citing `E:\codex-artifacts`
  from before 2026-09-15 should be checked against the real drive.

## Full detail

`BRINGUP.md` (this directory) — every blocker found and how it was resolved.
`READINESS_GAP.md` — the exact acceptance contract for each readiness gate.
`../teleop_policy_sweep_20260915/SWEEP.md` — twelve checkpoints ranked, and why
selection cannot close the gap.
