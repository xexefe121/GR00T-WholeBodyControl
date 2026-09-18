# BFM-Zero 23-DoF as the full-body teleop controller (started 2026-09-17)

## Why this direction

Every attempt to make a SONIC-derived tracker work on the 23-DoF G1 has failed,
and the 2026-09-17 feasible-corpus experiment showed why training cannot fix it:
under the frozen-LoRA recipe, episodes end after about four control steps.

Meanwhile a working whole-body controller already existed in this repository.
**BFM-Zero 23-DoF** (`RoboJuDo_Hero/assets/models/g1/BFM0/23dof_260411`, trained
384M steps, public checkpoint `Kennyp-Chen/bfmzero-23dof`), run plainly — no
trained residual — with IMU-based root odometry, was evaluated on 2026-09-10 and
described as "a working simulation candidate"
(`artifacts/teleop_six_hour_20260910/bfm_observable_closed_loop_findings.md`).

The project then moved on to training a residual policy on top of it. That
residual was rejected (`artifacts/g1_true23_bfm_residual_20260911_v1/train3h_v1/REJECTED_OUTCOME.md`),
and the working base controller was not pursued further. This effort picks it up
again.

## Reproduction of the recorded result

`gear_sonic/scripts/evaluate_g1_true23_bfm_observable.py` hard-codes
`DATA = Path("C:/Users/camer/...")` and translates `/mnt/c/` paths to `C:/`: it
was written to run under **Windows Python**, not WSL. Windows Python 3.10 already
has MuJoCo 3.2.3 (the pinned version), torch 2.10.0+cpu, onnxruntime 1.23.2 and
safetensors 0.8.0. The evaluator was run unmodified:

```
PYTHONPATH=Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof \
python -m gear_sonic.scripts.evaluate_g1_true23_bfm_observable \
  --output E:/codex-artifacts/bfm_teleop_20260917/repro_walk002_ideal \
  --clips walk002 --noise ideal
```

| walk002, ideal sensors | Recorded 2026-09-10 | Reproduced 2026-09-17 |
|---|---:|---:|
| Controls completed | 1417 / 1417 | 1417 / 1417, `failure: null` |
| Leg joint RMSE (rad) | 0.182 | 0.18202 |
| Arm joint RMSE (rad) | 0.043 | 0.04273 |
| Root error p95 (m) | 0.367 | 0.36741 |
| Odometry XY error p95 (m) | 0.018 | 0.01844 |

The result reproduces. Wall time was 26 s for 28.3 s of simulated lifecycle on a
single CPU process, so it runs faster than real time.

For context on the same clip, the best SONIC-derived checkpoint measured an arm
joint RMSE of 0.5841 rad. The two evaluators do not score identically — BFM
control counts include entry and return, and its root error is measured against
the controller's request — so the comparison is indicative, not exact. The arm
difference is large enough, at roughly 13.7 times, that the direction is not in
doubt.

## Log

### 2026-09-17 — All recorded cases reproduce

Remaining clips run with the same unmodified evaluator, ideal and fixed
bias/noise sensors (4 min 33 s for all six cases):

| Clip | Sensors | Completed | Leg RMSE | Arm RMSE | Root p95 | Odometry XY p95 |
|---|---|---:|---:|---:|---:|---:|
| walk003 | ideal | 1569 / 1569 | 0.2019 | 0.0508 | 0.597 m | 0.021 m |
| walk003 | bias + noise | 1569 / 1569 | 0.2157 | 0.0510 | 0.590 m | 0.120 m |
| walk008 | ideal | 1114 / 1114 | 0.2885 | 0.1274 | 0.818 m | 0.031 m |
| walk008 | bias + noise | 1114 / 1114 | 0.2890 | 0.1273 | 0.671 m | 0.080 m |
| PICO (115.6 s) | ideal | 6530 / 6530 | 0.1594 | 0.0705 | 0.415 m | 0.188 m |
| PICO (115.6 s) | bias + noise | 6530 / 6530 | 0.1626 | 0.0706 | 0.510 m | 0.278 m |

Every case reports `failure: null` and matches the 2026-09-10 table to its
published precision. Together with walk002, all eight recorded cases reproduce.

Two results matter most:

- **walk003 completes.** Every SONIC-derived checkpoint falls on it.
- **The full PICO capture completes**, with sensor noise, despite that capture's
  498 self-collision frames and 53 fixed-body contact frames. The clean SONIC
  controller fell at transition 189 of 535 on a PICO bundle. BFM tracks what is
  physically reachable rather than failing on the unreachable parts.

Relative landmark p95 values (hands, pelvis-relative, in metres): walk002
0.114 / 0.118, walk003 0.147 / 0.133, walk008 0.276 / 0.323, PICO 0.185 / 0.213.

### 2026-09-17 — Streamed-input fault scenarios pass, with ground-truth root

`gear_sonic/scripts/evaluate_g1_true23_bfmzero_stream.py`, unmodified, Windows
Python, walk002. The stream simulator keeps separate source and simulator clocks,
admits only received samples into the goal, and on a fault switches to a
generated standing reference while the same BFM policy keeps balancing.

| Field | normal | pause | disconnect | resume |
|---|---|---|---|---|
| `scenario_passed` | True | True | True | True |
| Controls | 1824 | 900 | 900 | 1575 |
| `any_fault_latched` | False | True | True | True |
| `physical_failure` | None | None | None | None |
| `standing_return_verified` | True | True | True | True |
| `range_excess_controls` | 0 | 0 | 0 | 0 |
| `effort_ratio_max` | **1.0** | 0.763 | 0.763 | 0.763 |
| `velocity_ratio_max` | 0.387 | 0.364 | 0.364 | 0.364 |
| `simulator_pose_writes_after_initialization` | 0 | 0 | 0 | 0 |
| `root_assistance_forces` | 0 | 0 | 0 | 0 |
| `controller_fallback` | False | False | False | False |
| **`ground_truth_pose_feedback`** | **True** | **True** | **True** | **True** |
| Epochs | 1 | 1 | 1 | 2 |

The passes are substantive: no pose writes, no assist forces, no joint-range
violations, standing verified after every fault, and the same policy balancing
throughout rather than a separate fallback. Faults latch when they should and not
when they should not, and `resume` rearms into a second epoch.

Two qualifications:

1. **Ground-truth root pose is used.** The stream simulator gives BFM the
   simulator's true root pose, which does not exist on hardware. The sensor-only
   `Native23IMUOdometry` estimator that replaces it is proven separately in the
   observable evaluator (XY p95 0.018–0.28 m), but the two have not been combined.
2. **Effort reaches its limit** in the normal run (`effort_ratio_max` 1.0). It is
   clipped rather than exceeded, but it is saturation, and should be watched.

## The remaining integration, all from existing parts

| Piece | Status |
|---|---|
| BFM-Zero 23-DoF inference | Exists, reproduces |
| Sensor-only IMU root odometry | Exists, proven in observable evaluator |
| Stream receiver: admission, latch, standing fallback, rearm | Exists, proven above |
| Real-time pacing (`--paced`, Windows high-resolution clock) | Exists, not yet exercised |
| Real transport between processes (ZMQ) | **Missing** — source is read in-process |
| Live PICO to native23 body frames | Partly present (`build_full_body()`, uncommitted) |

### 2026-09-17 — ZMQ teleop simulation

All 3 of T1/T2/T3 passed their fixed criteria.  Windows Python was used from
the repository root with `PYTHONPATH=Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof`.
For each case, the consumer command was started first in one PowerShell, then
the publisher command was started in another; publisher `--start-delay 3`
allows the ZMQ SUB subscription handshake to complete before source time zero.

| Test | Consumer command | Publisher command |
|---|---|---|
| T1 | `python -m gear_sonic.scripts.run_g1_true23_bfm_teleop_sim consume --endpoint tcp://127.0.0.1:5591 --output E:/codex-artifacts/bfm_teleop_20260917/zmq/T1_walk002_normal --paced` | `python -m gear_sonic.scripts.run_g1_true23_bfm_teleop_sim publish --clip walk002 --endpoint tcp://127.0.0.1:5591 --fault none --start-delay 3` |
| T2 | `python -m gear_sonic.scripts.run_g1_true23_bfm_teleop_sim consume --endpoint tcp://127.0.0.1:5592 --output E:/codex-artifacts/bfm_teleop_20260917/zmq/T2_walk002_pause --paced` | `python -m gear_sonic.scripts.run_g1_true23_bfm_teleop_sim publish --clip walk002 --endpoint tcp://127.0.0.1:5592 --fault pause --fault-at 9.0 --fault-duration 0.5 --start-delay 3` |
| T3 | `python -m gear_sonic.scripts.run_g1_true23_bfm_teleop_sim consume --endpoint tcp://127.0.0.1:5593 --output E:/codex-artifacts/bfm_teleop_20260917/zmq/T3_walk002_disconnect --paced` | `python -m gear_sonic.scripts.run_g1_true23_bfm_teleop_sim publish --clip walk002 --endpoint tcp://127.0.0.1:5593 --fault disconnect --fault-at 9.0 --fault-duration 0.5 --start-delay 3` |

`run_g1_true23_bfm_teleop_sim.py` is thin two-process glue.  Its publisher
sends exactly the six received-reference fields, `sequence`, `source_time`,
`epoch`, and `final` over real ZMQ JSON.  Its consumer gives received packets
to `BFMStreamSimulator.receive()` only, advances the simulator at 500 Hz even
when no packet arrives, and retains the stream module's gate, stale fault,
standing fallback, and rearm logic.  Policy feedback uses simulated pelvis
IMU and encoders plus `Native23IMUOdometry`; true simulator root translation is
only read for the requested tracking score.

| Fixed pass criterion | T1 normal | T2 pause | T3 disconnect |
|---|---:|---:|---:|
| `any_fault_latched True` (T2/T3) | n/a | True | True |
| `physical_failure is None` | True | True | True |
| `standing_return_verified True` | True | True | True |
| `range_excess_controls 0` | True | True | True |
| `ground_truth_pose_feedback False` | True | True | True |
| `simulator_pose_writes_after_initialization 0` (T1) | True | n/a | n/a |
| `root_assistance_forces 0` (T1) | True | n/a | n/a |
| `full_uninterrupted_source_consumed True` (T1) | True | n/a | n/a |
| `missed_control_deadlines` reported (T1) | True (152) | n/a | n/a |
| Result | **PASS** | **PASS** | **PASS** |

| Report field | T1 normal | T2 pause | T3 disconnect |
|---|---:|---:|---:|
| Controls / physics steps | 1573 / 15730 | 604 / 6040 | 603 / 6030 |
| Received samples / dropped or reordered | 1417 / 0 | 450 / 0 | 450 / 0 |
| Late / missed control deadlines | 1573 / 152 | 604 / 7 | 603 / 4 |
| Effort ratio max / velocity ratio max | 0.98075 / 0.31950 | 0.78759 / 0.37489 | 0.71762 / 0.30979 |

T1's source-phase values, calculated with the observable evaluator's joint
and root formulas, are compared below.  Differences are reported rather than
tuned away: transport pacing, received-source buffering, and the stream
lifecycle are different from the uninterrupted ideal evaluator.

| Metric | Observable walk002 ideal | T1 ZMQ paced | Difference (ZMQ − ideal) |
|---|---:|---:|---:|
| Leg RMSE (rad) | 0.18202 | 0.12556 | -0.05646 |
| Arm RMSE (rad) | 0.04273 | 0.12535 | +0.08262 |
| Root p95 (m) | 0.36741 | 0.33673 | -0.03068 |

One existing module changed: `g1_true23_bfmzero_stream.py` gained optional
`sensor_feedback` and `sensor_step` hooks.  When absent, the prior simulator
state feedback path is unchanged; when present, the actor state and received
goal correction consume the supplied sensor observation and root estimate.
The 500 Hz hook receives causal pre-step encoder/orientation state and the
MuJoCo pelvis IMU sample, matching `evaluate_g1_true23_bfm_observable.py`.
After that change,
`evaluate_g1_true23_bfmzero_stream.py --clip walk002 --scenario normal`
still passed unchanged: 1824 controls, no physical failure, full source
consumption, and verified standing return.

### 2026-09-17 — Independent verification of the ZMQ build, and a timing problem

Codex's reported results were rechecked from the report files, and all three
tests meet the pre-set pass criteria. Verified by reading the code, not only the
flag: the policy's goal is built from `feedback()`, which returns
`self.estimate["position_start"]` from `Native23IMUOdometry` fed by simulated IMU
and joint states. `ground_truth_pose_feedback: False` is genuine. The one
idealisation is IMU orientation, which is taken from the simulator's true pelvis
quaternion — equivalent to a perfect orientation estimate.

| Field | T1 normal | T2 pause | T3 disconnect |
|---|---|---|---|
| Received samples, dropped/reordered | 1417, 0 | 450, 0 | 450, 0 |
| `physical_failure` | None | None | None |
| `standing_return_verified` | True | True | True |
| `range_excess_controls` | 0 | 0 | 0 |
| `ground_truth_pose_feedback` | False | False | False |
| Pose writes / assist forces | 0 / 0 | 0 / 0 | 0 / 0 |
| Leg / arm RMSE (rad) | 0.1256 / **0.1254** | 0.0917 / 0.0930 | 0.0911 / 0.0929 |
| **Late / missed deadlines** | **1573 / 152** | 604 / 7 | 603 / 4 |
| **Start lateness p95 / max** | **207.8 / 342.1 ms** | 0.67 / 11.9 ms | 0.67 / 9.6 ms |

**The pass criteria did not cover timing, and T1's timing fails.** Every control
was late and 152 were missed outright, with lateness reaching 342 ms. The short
fault runs stay in time; the full-length run drifts. T1's arm RMSE of 0.125 rad is
about three times the 0.043 rad the observable evaluator achieves on the same clip
with the same estimator run unpaced, consistent with BFM tracking stale goals.

**Cause, measured.** A first hypothesis — single-threaded inference — was tested
and refuted: the in-process paced stream evaluator keeps real time on one thread
(p95 lateness 0.65 ms, zero deadline misses over 36.5 simulated seconds), and four
threads changed nothing. Timing the pieces directly:

| Per 20 ms control | Cost |
|---|---:|
| MuJoCo physics, 10 substeps | 1.40 ms |
| `Native23IMUOdometry.update`, 10 samples at 500 Hz | **5.31 ms** |
| Observable evaluator, everything included (26 s / 1417 controls) | 18.3 ms |

The sensor-only estimator costs 5.3 ms per control. With it, the loop sits at
about 18 ms of a 20 ms budget before ZMQ decoding and pacing are added, so a
full-length paced run tips over and accumulates lag.

### 2026-09-17 — Real-time performance

**0 of P1–P5 passed:** P1 failed its non-negotiable zero-missed-deadline
criterion, so P2–P5 were not run. Per the performance stop condition, this
campaign stops here rather than relaxing a timing criterion or changing policy
semantics.

The consumer now has an opt-in `--profile` mode. It records `perf_counter`
durations only when requested and writes `control_step_profile_ms` in
`report.json`. The profile covered a complete paced ZMQ walk002 run, including
receive/decode, admission, every 500 Hz estimator and physics update, feature
and goal construction, BFM inference, trace work, and pacing. All profile
outputs are in `E:/codex-artifacts/bfm_teleop_20260917/perf/`.

| Per-control part | Before: 1 thread/full trace mean/p95/max (ms) | After: 4 threads/minimal live trace mean/p95/max (ms) |
|---|---:|---:|
| ZMQ receive and decode | 0.168 / 0.218 / 1.827 | 0.182 / 0.254 / 0.906 |
| Stream admission | 0.581 / 0.699 / 1.817 | 0.630 / 0.904 / 1.394 |
| Estimator update, each 500 Hz sample | 0.530 / 0.627 / 1.581 | 0.566 / 0.678 / 7.107 |
| Physics substep, each 500 Hz step | 0.115 / 0.147 / 1.283 | 0.122 / 0.164 / 1.382 |
| Feature construction and goal transform | 0.871 / 1.032 / 2.819 | 0.894 / 1.137 / 3.037 |
| BFM inference | 4.922 / 5.481 / 9.635 | **3.817 / 4.472 / 6.447** |
| Per-step logging and array appending | 0.022 / 0.026 / 0.563 | 0.021 / 0.024 / 0.186 |
| Pacing wait | 6.254 / 7.050 / 7.876 | 6.887 / 7.760 / 8.663 |
| Whole control loop | 13.481 / 15.410 / 23.519 | **12.777 / 15.142 / 23.669** |

The following behaviour-preserving changes were retained:

1. The live ZMQ consumer sets `record_trace=False`. The existing evaluator
   keeps its default full forensic trace. The consumer retains every report
   field, collecting only the source joint errors and root norms needed for its
   scalar tracking report instead of copied 500 Hz arrays and unused
   per-control tensors. Measured full-loop saving at one thread was 0.121 ms
   median and 0.328 ms p95 (`baseline_walk002_profile` versus
   `trace_optimized_walk002_profile`).
2. The consumer uses four Torch CPU threads by default (and retains an explicit
   `--torch-threads` override). BFM inference improved by 1.140 ms/control on
   average, from 4.958 ms to 3.817 ms; whole-loop median improved a further
   0.583 ms. Eight threads were worse (14 missed deadlines) and were rejected.
3. Profiling is behind `--profile`; non-profiled consumer runs do not execute
   its timing calls.

An estimator scratch-buffer candidate was checked over 2,000 recorded 500 Hz
samples. It was bit-identical on `position_start`, `velocity_start`,
`accel_bias_body`, `quaternion_start`, and `support_weights` (maximum absolute
difference **0.0**, threshold `1e-9`), but its timed replay was 0.188
ms/update slower, so it was discarded. The deployed `Native23IMUOdometry` hot
path is therefore unchanged. An attempted removal of `mj_comPos` was also
discarded after the same check detected a 0.404 maximum difference.

Final behaviour regressions passed exactly:

| Check | Result |
|---|---|
| `evaluate_g1_true23_bfm_observable --clips walk002 --noise ideal` | Completed 1417; leg RMSE `0.18202273382760834`; arm RMSE `0.04273216819201739`; root p95 `0.36741054963315306` |
| `evaluate_g1_true23_bfmzero_stream --clip walk002 --scenario normal` | `scenario_passed True`; controls 1824 |

| Test | Status | Fixed criteria / tracking comparison |
|---|---|---|
| P1 walk002 normal | **FAIL** | `physical_failure None`; `standing_return_verified True`; `range_excess_controls 0`; `ground_truth_pose_feedback False`; `simulator_pose_writes_after_initialization 0`; `root_assistance_forces 0`; `full_uninterrupted_source_consumed True`; start lateness p95 **0.559 ms** (<5 ms); **`missed_control_deadlines 1`**. Full loop maximum was **23.5823 ms**, **3.5823 ms over** the 20 ms deadline. Tracking: leg/arm/root p95 = `0.12556 / 0.12535 / 0.33673`, versus unpaced ideal `0.18202 / 0.04273 / 0.36741`. |
| P2 walk003 normal | Not run | Terminal stop after P1 timing failure. |
| P3 pico normal | Not run | Terminal stop after P1 timing failure. |
| P4 walk002 pause | Not run | Terminal stop after P1 timing failure. |
| P5 walk002 disconnect | Not run | Terminal stop after P1 timing failure. |

The remaining cost is not routine loop work: the after profile’s median and
p95 are 12.777 ms and 15.142 ms. The unresolved host-scheduling tail produces
an occasional full control of 23.5823 ms; it exceeds the fixed 20 ms budget by
3.5823 ms. No estimator samples, physics substeps, policy inputs, gates,
thresholds, horizons, or source packets were skipped, merged, or made stale to
mask this result.

### 2026-09-17 — Verification of the performance work, and what the arm error actually is

**Timing is fixed; one miss remains.** After codex's changes, P1 start lateness
p95 fell from 207.8 ms to 0.559 ms, with the whole loop at a median of 12.8 ms and
p95 of 15.1 ms against a 20 ms budget. One control took 23.58 ms. The
pre-registered criterion was zero missed deadlines, so **P1 is recorded as
failed** and the criterion is not relaxed. The single spike coincides with a
7.1 ms estimator outlier on an unqualified shared desktop — host-scheduling
jitter rather than a systematic lag — but that reading is an interpretation, not
a qualification. P1 is being rerun alongside P2–P5 to see whether it recurs.

**Correction: the elevated arm error was never caused by lag.** P1's tracking,
after the lag was removed, is identical to T1's to five significant figures —
leg / arm / root p95 = 0.12556 / 0.12535 / 0.33673 — even though lateness fell by
a factor of about 370. The simulator advances deterministically in simulated
time, so wall-clock lateness never altered what the policy received. An earlier
entry's suggestion that stale goals explained the arm error was wrong.

**Isolating the difference.** The in-process stream evaluator's saved trace
(`stream_walk002_normal/trace.npz`, ground-truth root, no network) was scored with
the consumer's own formula:

| walk002 | Root source | Transport | Leg RMSE | Arm RMSE | Root p95 |
|---|---|---|---:|---:|---:|
| In-process stream | ground truth | none | 0.14021 | **0.12455** | 0.525 m |
| ZMQ consumer | IMU odometry | real ZMQ | 0.12556 | **0.12535** | 0.337 m |
| Observable evaluator | IMU odometry | none, non-causal | 0.18202 | 0.04273 | 0.367 m |

Arm error differs by **0.0008 rad** between ground truth without a network and
sensor-only root over a real socket. **The transport and the sensor-only
estimator together cost essentially no tracking quality.** Leg and root error are
lower with the estimator.

The gap to the observable evaluator's 0.043 rad comes from the stream pipeline
itself and is present even with perfect information. That pipeline admits only
received samples into an eight-frame window, re-anchors each epoch's reference to
the robot's pose (`ReferenceAnchor`), and scores the robot against the *oldest*
frame of that window (`samples[0]`) while the policy steers toward the later
ones. The observable evaluator reads the reference non-causally and scores
against the current time. The two figures are therefore not the same measurement,
and about 0.125 rad (roughly 7 degrees) is the representative figure for causal
teleoperation through this receiver. How much of it is scoring alignment rather
than true tracking lag has not been separated.

### 2026-09-17 — Acceptance over real ZMQ: 2 of 5 pass the strict criteria

Optimised consumer, Windows Python, paced, sensor-only root, real ZMQ between
separate publisher and consumer processes. Script:
`scratchpad/run_p2_p5.sh`. Outputs:
`E:\codex-artifacts\bfm_teleop_20260917\acceptance_p2_p5\`.

| Criterion | P1 walk002 | P2 walk003 | P3 PICO 115.6 s | P4 pause | P5 disconnect |
|---|---|---|---|---|---|
| Controls / received samples | 1573 / 1417 | 1725 / 1569 | 6686 / 6530 | 604 / 450 | 604 / 450 |
| Dropped or reordered | 0 | 0 | 0 | 0 | 0 |
| `physical_failure` | None | None | None | None | None |
| `standing_return_verified` | True | True | True | True | True |
| `ground_truth_pose_feedback` | False | False | False | False | False |
| Pose writes / assist forces | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| `any_fault_latched` | False | False | False | True | True |
| `full_uninterrupted_source_consumed` | True | True | True | — | — |
| `range_excess_controls` | 0 | 0 | **5** | 0 | 0 |
| `missed_control_deadlines` (0 required for P1–P3) | **14** | **9** | **5** | 5 | 1 |
| Start lateness p95 / max (ms) | 0.6 / 6.0 | 0.6 / 4.7 | 0.6 / 5.1 | 0.5 / 8.9 | 0.5 / 3.1 |
| Leg / arm RMSE (rad) | 0.1256 / 0.1253 | 0.1478 / 0.1505 | 0.1641 / 0.2150 | 0.0917 / 0.0930 | 0.0917 / 0.0930 |
| Root p95 (m) | 0.337 | 0.565 | 0.402 | 0.243 | 0.243 |
| **Strict verdict** | **FAIL** | **FAIL** | **FAIL** | **PASS** | **PASS** |

**Functionally, every case works.** All five complete with no physical failure
and verified standing, over a real socket, with sensor-only root and no dropped
or reordered samples. That includes walk003 and the full 115.6 s PICO capture,
both of which every SONIC-derived controller fails. Pause and disconnect latch
the fault and return to verified standing.

**Two things fail the criteria as written, and neither is relaxed here.**

1. **Sporadic missed deadlines on the full-length runs** (14, 9 and 5). Start
   lateness p95 stays at 0.6 ms and nothing accumulates, so this is not the drift
   that existed before optimisation. The count also varies between runs — P1 gave
   one miss in the optimisation run and fourteen here — which points to host
   scheduling on a shared Windows desktop. Timing is not qualified on this
   machine. On hardware the loop would not simulate physics (1.4 ms per control
   here), but it would still carry inference, the 5.3 ms estimator and transport.
2. **Joint-range excess on PICO:** 5 controls, maximum **6.4 mrad (0.37°)**. This
   is the capture with 498 self-collision frames, and it matches the magnitudes the
   rejected-residual report recorded for BFM on PICO (5.37, 3.47, 4.92 mrad). The
   standard remedy is to clamp commanded joint targets a small margin inside the
   model's joint ranges.

Determinism check: P1 tracking again equals T1 and the optimisation run to five
significant figures.

## Status

Full-body teleoperation of the 23-DoF G1 **runs in simulation** with BFM-Zero:
real transport between processes, sensor-only root estimation, every reference
clip completed including the ones that defeat SONIC, and faults handled by a
return to verified standing. It is **not yet qualified**: timing is unqualified on
this machine and the PICO capture produces sub-degree joint-range excess.

Nothing here touches hardware. Live PICO capture remains untested because no
headset has connected.

### 2026-09-17 — Fix 2: joint-range clamp and real-time spikes

**0 of P1–P5 passed in both runs:** the required two complete acceptance runs
were not started because the permitted real-time measures still leave P1–P3
with non-zero missed deadlines; this entry stops at that unchanged hard
criterion rather than relaxing it.

#### A1. Range-excess diagnosis before the controller change

`range_excess` is measured, not commanded: for each 50 Hz control it is the
maximum over its ten post-`mj_step` samples of
`max(jnt_range[1:,0] - data.qpos[7:], data.qpos[7:] - jnt_range[1:,1], 0)`.
`range_excess_controls` counts controls whose measured maximum is non-zero.
It does not inspect PD targets.  Before this fix the final output path already
used `np.clip(target, model.jnt_range[1:,0], model.jnt_range[1:,1])` before
the PD computation; 1,107 PICO target components were hard-clipped somewhere
in the capture.

The five P3 controls were captured over the normal paced ZMQ path with the
pre-fix controller and a forensic trace enabled externally.  `Raw target` is
the policy-derived target before the existing hard clamp; it equals `PD target`
for all five rows.  Therefore the targets are inside the model range, while
the measured joint crosses the lower stop dynamically.

| Control / worst substep | Joint | PD target (rad) | Raw target outside range? | Model range (rad) | Measured qpos (rad) | Excess (mrad) |
|---|---|---:|---|---|---:|---:|
| 4178 / 10 | `left_ankle_roll_joint` | 0.034253 | No | [-0.261800, 0.261800] | -0.263468 | 1.668 |
| 4179 / 10 | `left_ankle_roll_joint` | 0.057484 | No | [-0.261800, 0.261800] | -0.268197 | 6.397 |
| 4180 / 1 | `left_ankle_roll_joint` | 0.081224 | No | [-0.261800, 0.261800] | -0.268193 | 6.393 |
| 4181 / 1 | `left_ankle_roll_joint` | 0.074018 | No | [-0.261800, 0.261800] | -0.266869 | 5.069 |
| 4182 / 1 | `left_ankle_roll_joint` | -0.003313 | No | [-0.261800, 0.261800] | -0.263896 | 2.096 |

The crossing began with the ankle at -0.243636 rad and velocity -0.881365
rad/s at control 4178.  It was still moving outward despite the already
inward target.  The cause is therefore dynamic overshoot, not a commanded
target outside its range.

#### A2. Output safety fix

The final PD-target layer now has an explicit per-joint interior margin: 6.5
mrad for `left_ankle_roll_joint`, zero for the other 22 joints.  6.5 mrad is
the smallest 0.1-mrad-rounded margin that exceeds the measured 6.397 mrad
maximum overshoot (0.103 mrad headroom).  The hard target clamp remains in
place for ordinary commands.  If measured `q + dq * 60 ms` predicts crossing
the corresponding interior limit, only that final PD target is projected to
the opposite interior endpoint to brake it.  This is a controller-output
safety override after policy output and before PD; no simulator state,
range-check, model range, gain, effort/velocity limit, policy input,
estimator rate, physics substep, horizon, stale threshold, or gate changed.

Full paced PICO over ZMQ after the change completed 6,686 controls with
`range_excess_controls 0` and `range_excess_max_rad 0.0`.

#### A3. Clips with no original excess

| Clip | Before leg / arm / root | After leg / arm / root | Change |
|---|---|---|---|
| walk002 P1 | 0.125560 / 0.125350 / 0.336730 | 0.125555 / 0.125346 / 0.336732 | -0.000005 / -0.000004 / +0.000002 |
| walk003 P2 | 0.147800 / 0.150500 / 0.565000 | 0.148471 / 0.150157 / 0.564537 | +0.000671 / -0.000343 / -0.000463 |

Both after runs have `range_excess_controls 0`; every tracking change is below
0.005 rad (or 0.005 m for root), so no no-excess clip was materially changed.

#### B1. Spike attribution

`--profile` now retains, for every control over 18 ms or past its deadline,
the per-control receive/decode, admission, feature/goal, inference, physics,
estimator, and logging durations.  It also installs `gc.callbacks` timestamps
for the timed loop.  The unmitigated P1 profile (automatic GC, normal priority,
no affinity) had one miss and the following complete spike set:

| Control | Full loop (ms) | Missed? | Dominant timed work (ms) | GC correlation |
|---:|---:|---|---|---|
| 0 | 23.362 | Yes | estimator 5.941; inference 3.726; feature/goal 2.946; receive 1.785 | gen-0 GC at 5.428–5.500 ms, only 0.073 ms |
| 72 | 18.073 | No | estimator 7.814; inference 5.082 | None |
| 1424 | 19.123 | No | estimator 5.333; inference 4.112 | None |

GC is not the cause of the observed tail.  The spikes are intermittent host
scheduling/allocation/estimator-and-physics bursts; the profiler records the
evidence in `fix2/b1_walk002_baseline_profile/report.json`.

#### B2. Behaviour-preserving real-time measures and measured effect

| Incremental configuration, all P1 over paced ZMQ | Misses | Full-loop p50 / p95 / max (ms) | Result |
|---|---:|---|---|
| Automatic GC, normal priority, no affinity (profile baseline) | 1 | 12.777 / 15.028 / 23.362 | Baseline |
| Manual GC, normal priority, no affinity | 5 | 12.895 / 15.102 / 26.425 | Retained for bounded collection placement; miss count remains variable |
| Manual GC + HIGH process/main-thread priority, no affinity | 2 | 12.940 / 15.099 / 23.408 | Retained; tail remains |
| Above + four-core separated affinity | 5 | 14.815 / 17.799 / 26.135 | Rejected: slower |
| Above + all-but-publisher-core affinity | 3 | 12.765 / 15.105 / 22.712 | Rejected: still worse than no affinity |
| Disposable full-path pre-warm trial | 3 | 12.952 / 15.218 / 25.763 | Rejected: worsened tail |

Automatic GC is disabled only after warm-up and before the timed loop using
`gc.collect(); gc.freeze(); gc.disable()`.  The prior state is restored and a
collection occurs after the loop.  Full PICO working set fell from
667,181,056 to 654,532,608 bytes across 6,686 controls; it did not grow
without bound.  Three un-timed actor and independent-estimator calls remain
before control start; the discarded full-path warm-up did not alter live state.
The consumer is set to `HIGH_PRIORITY_CLASS` (and its control thread to
`THREAD_PRIORITY_HIGHEST`) for its lifetime.  The pacer already uses a
high-resolution Windows waitable timer, so no `timeBeginPeriod` change was
added.  Affinity is available but defaults off because both tested
role-separated layouts increased misses.

#### B3. Behaviour regressions

| Check | Result |
|---|---|
| Observable evaluator, walk002 ideal | completed `1417`; leg `0.18202273382760834`; arm `0.04273216819201739`; root p95 `0.36741054963315306` |
| Stream evaluator, walk002 normal | `scenario_passed True`; `controls 1824` |
| P1 ZMQ tracking after the clamp | leg / arm / root = `0.12555505979060572 / 0.12534553753068026 / 0.33673229342694455`, matching `0.12556 / 0.12535 / 0.33673` |

#### Acceptance tables

The prescribed harness was copied to
`E:/codex-artifacts/bfm_teleop_20260917/fix2/run_p1_p5.sh`; its only edits are
the output base directory (`fix2/acceptance_run1`).  Per the terminal
instruction below, neither full five-test acceptance pass was started after
the individual P1/P2/P3 evidence showed the non-negotiable deadline criterion
still fails.

| Test | Harness run 1 | Harness run 2 |
|---|---|---|
| P1 walk002 | Not run: individual post-fix P1 has `missed_control_deadlines 2` | Not run |
| P2 walk003 | Not run: individual post-fix P2 has `missed_control_deadlines 1` | Not run |
| P3 PICO | Not run: individual post-fix P3 has `missed_control_deadlines 8` | Not run |
| P4 pause | Not run after terminal timing failure | Not run |
| P5 disconnect | Not run after terminal timing failure | Not run |

**Remaining criterion, exactly:** `missed_control_deadlines 0` is still false
for P1, P2, and P3.  P1 has 2, P2 has 1, and P3 has 8 misses in the final
individual post-fix checks.  All three preserve the other reported safety and
lifecycle criteria, including PICO's now-zero range excess, but they cannot
pass the unchanged acceptance definition.  No acceptance limit, check, or
gate was relaxed, and no hardware, DDS, robot, BFM weight, or training change
was made.

### 2026-09-17 — Review of Fix 2

**Part A verified, with a hardware caveat.** The diagnosis is sound: all five
PICO excess controls were `left_ankle_roll_joint`, with PD targets well inside
range (0.034 to 0.081 rad) while the joint crossed its lower stop at up to
0.88 rad/s. The policy was already pulling inward and body load carried the
joint past the stop anyway, so a target clamp alone could not have worked.

The fix adds a 6.5 mrad interior margin on that joint and, when
`q + dq × 60 ms` predicts a crossing, sets the PD target to the **opposite**
interior endpoint — a jump of about 0.5 rad for the ankle roll joint. Because
the inward target was already insufficient, only a maximal inward command brakes
harder, so the design is mechanically coherent. Its effect on the full PICO run:

| PICO, paced ZMQ | Before | After |
|---|---:|---:|
| `range_excess_controls` | 5 | **0** |
| `range_excess_max_rad` | 0.0064 | 0.0 |
| `effort_ratio_max` | 0.981 | **1.000** |
| `velocity_ratio_max` | 0.663 | 0.519 |

The actuator now reaches its effort limit: the joint is held by saturating
torque in a bang-bang fashion. Normal clips are unaffected (walk002 changed by at
most 5e-6, walk003 by at most 7e-4). **This is acceptable in simulation but must
be replaced with a bounded, rate-limited brake before any hardware use.** A
torque step of that size on a load-bearing ankle is not something to discover on
the robot. It is also only reached on the PICO capture, whose poses include
self-collisions.

**Part B: a host limit, not a code defect.** Every permitted real-time measure was
tried and measured. Garbage collection is not correlated with the spikes (one
0.073 ms gen-0 collection in the profiled set). High process and thread priority
reduced but did not remove misses; both CPU-affinity layouts and full-path
pre-warming made the tail worse. The remaining spikes land in estimator and
physics work intermittently, consistent with host scheduling on a non-real-time
Windows desktop. Final individual checks: P1 2, P2 1, P3 8 missed deadlines.

The zero-miss criterion is **not relaxed and not met on this host**. What it
means in practice is a decision for the operator rather than for this log: a miss
here is a single 50 Hz target update arriving up to about 6 ms late, while the
motor-level PD continues holding the previous target. The options are to qualify
timing on the machine that will actually run the controller, or to set an
explicit, documented tolerance for rare late updates.

### 2026-09-17 — Fix 3: bounded brake and hardware-loop timing

**Part A passed; no Part B runtime configuration passed.** The PICO safety
run has zero range-excess controls and a sub-saturation peak effort ratio, but
none of Windows CPU, WSL CPU, or WSL CUDA met the unchanged zero-miss and
sub-4-ms lowcmd-gap timing requirement. No physical robot, robot `lowcmd`
topic, or robot-facing network interface was opened.

#### A. Bounded ankle brake

The old `left_ankle_roll_joint` endpoint projection was replaced at the final
target layer (after policy output and before PD) by a 0.100-rad/control inward
increment. It retains the 6.5-mrad interior margin and the ordinary hard model
range clamp. On a predicted lower/upper crossing (`q + dq * 60 ms`), the target
is moved inward from the policy target by at most 0.100 rad, clipped to the
opposite interior endpoint, and constrained against the preceding emitted brake
target to a maximum 0.100-rad step. This is one fifth of the former roughly
0.5-rad bang-bang endpoint jump. The selected bound kept PICO at zero excess
while lowering peak effort below saturation; reducing it would provide less
braking margin without a demonstrated safety benefit.

| Paced ZMQ run | Step (rad/control) | Brake engagements | Range-excess controls | Peak effort ratio | Leg / arm / root |
|---|---:|---:|---:|---:|---|
| PICO | 0.100 | 36 | **0** | **0.977141** | 0.161183 / 0.216254 / 0.386952 |
| walk002 | 0.100 | 1 | 0 | 0.980753 | 0.125579 / 0.125358 / 0.336732 |
| walk003 | 0.100 | 2 | 0 | 1.000000 | 0.147200 / 0.150431 / 0.564537 |

The largest emitted brake-created target step was exactly 0.100 rad in all
three runs. walk002 and walk003 remain within 0.005 of their binding reference
metrics. The A3 pass condition applies PICO's effort requirement to PICO; the
walk003 peak effort ratio is reported for completeness and does not change its
tracking comparison.

Final regressions, Windows Python: observable walk002 ideal completed 1417
with leg `0.18202273382760834`, arm `0.04273216819201739`, and root p95
`0.36741054963315306`; stream walk002 normal had `scenario_passed True` and
1824 controls. The stream-brake unit suite also passed: 33 tests.

#### B. Hardware-shaped timing on this PC

`gear_sonic/scripts/time_g1_true23_bfm_hardware_loop.py` contains no MuJoCo
stepping in its timed loop. It preloads the recorded PICO sensor-only replay,
runs `Native23IMUOdometry.update` and a lowcmd send every 2 ms, and admits the
existing PICO ZMQ publisher through `PacketGate` at 50 Hz before building the
BFM features/goal and updating targets through the bounded brake. The PICO
source duration was 115.6 s (57,800 sensor ticks); valid runs admitted 5,780
teleop packets and made 5,773 policy updates.

Windows used a fixed-size local command serialization buffer, not DDS. WSL
constructed real `unitree_hg` `LowCmd` messages, calculated their CRC, and
published only on DDS domain 232/topic `rt/fix3_timing_no_robot_lowcmd` bound
to `lo`; it never used `rt/lowcmd` or `eth0`. Automatic GC was disabled only
during the timed loop. Windows used Fix 2 HIGH process/thread priority; WSL
successfully applied `SCHED_FIFO:10` and `nice:-10`.

CUDA equivalence, before the CUDA timing run: 200 recorded BFM inputs, maximum
CPU/CUDA actor-output absolute difference `3.5762786865234375e-07`, passing the
required `<= 1e-5` check.

| Full PICO run | 500 Hz: misses; start p50/p95/max ms; work p50/p95/max ms | 50 Hz: misses; start p50/p95/max ms; work p50/p95/max ms | Largest lowcmd gap (ms) | Verdict |
|---|---|---|---:|---|
| Windows Python, Torch CPU, HIGH | 22,845; 0.492/4.598/10.890; 0.596/6.130/12.637 | 0; 0.991/1.289/2.655; 5.533/6.676/12.005 | 13.912 | Fail |
| WSL, Torch CPU | 37,880; 2.146/19.358/51.097; 0.969/6.580/10.866 | 432; 0.499/16.846/45.894; 5.581/6.938/9.223 | 11.038 | Fail |
| WSL, Torch CUDA | 57,320; 4395.971/8446.741/9063.242; 1.023/10.630/459.213 | 5,649; 4392.033/8439.519/9053.793; 9.578/11.189/458.343 | 460.307 | Fail |

The CUDA configuration is the least suitable on this WSL host: synchronous
GPU work produced a multi-second accumulated schedule slip. The Windows CPU
configuration is the closest functional result: its 50-Hz path had zero missed
deadlines, but the same-thread 5.5-ms median inference/goal update blocks the
2-ms lowcmd path, yielding 22,845 500-Hz misses and a 13.912-ms maximum send
gap. Consequently no second consecutive full run was started: no first run
passed B3, and the stated zero-miss criterion remains unmet rather than relaxed.

All Fix 3 evidence is under `E:\codex-artifacts\bfm_teleop_20260917\fix3\`.

### 2026-09-18 — Review of Fix 3, and the architecture change it forces

**Part A accepted.** The bounded brake moves the target inward by at most
0.100 rad per control, a fifth of the former endpoint jump. PICO: 36 engagements,
zero range excess, peak effort 0.977 (was 1.000). walk002 and walk003 stay within
0.005 of their reference metrics, both regression checks reproduce exactly, and
33 stream-brake tests pass. CPU and CUDA inference agree to 3.6e-7.

One observation independent of the brake: walk003 reaches effort ratio 1.000 with
only two brake engagements, so BFM uses full actuator effort on demanding walking
on its own. This is a watch item for gantry testing rather than a defect.

**Part B: the failure is structural.** No configuration passed. The decisive
detail is that the Windows CPU configuration's **50 Hz path had zero misses**
(start p95 1.29 ms, work p95 6.68 ms) while its 500 Hz path missed 22,845 of
57,800 deadlines with a 13.9 ms maximum lowcmd gap. The report names the cause:
the 5.5 ms median inference and goal update ran on the same thread as the 2 ms
lowcmd send. A task of that length cannot share a thread with a 2 ms deadline, so
no amount of optimisation within that structure would pass.

WSL CPU did worse (37,880 and 432 misses, 500 Hz start p95 19.4 ms even under
`SCHED_FIFO`). WSL CUDA accumulated a multi-second schedule slip from synchronous
GPU work and is unsuitable on this host.

**Change of architecture for Fix 4** — the standard real-robot structure, and the
one this repository's own C++ deploy stack already uses:

1. A **dedicated 500 Hz loop in native code** whose only work is to receive
   lowstate, buffer the samples, and resend the most recent joint targets in
   lowcmd every 2 ms. It never waits on the policy.
2. **BFM at 50 Hz in a separate process**, which already meets its deadlines on
   Windows CPU.
3. **The IMU estimator updated in a batch at each 50 Hz tick** from the ten
   buffered samples, in order and with their original timestamps. Its estimate is
   only consumed at 50 Hz, by the goal transformation, so the state it produces at
   that point is identical to updating every 2 ms. This must be demonstrated with
   an equivalence check, not assumed. It removes about 5.3 ms of Python work from
   the 2 ms path entirely.

### 2026-09-18 — Fix 4: split control loop

**C1 and C2 passed exactly; no placement passed timing.** C1 is exactly `0.0`
over every estimator output field at all 6,530 50-Hz ticks of the full 130.6-s
replay. C2 is exactly `0.0` over all 6,530 lock-step emitted targets. Neither
placement met the zero-miss and sub-4-ms requirement, so no second run was
started.

`g1_true23_bfm_lowcmd_loop` is a native C++ 500-Hz replay loop using
`clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME)`, with attempted
`SCHED_FIFO:10` and `mlockall`. Its only timed work is a `SampleSource` replay
read, non-blocking ZMQ state enqueue, non-blocking target drain, CRC, and a
LowCmd send. The compiled DDS surface is fixed to domain 232, `lo`, and
`rt/fix4_timing_no_robot_lowcmd`; it has no `rt/lowcmd` or `eth0` option. The
50-Hz process batches the ordered samples through unchanged
`Native23IMUOdometry`, PacketGate, BFM, goal construction, and 0.100-rad brake.

| Check | Result |
|---|---|
| C1 batching | Pass; fields `accel_bias_body`, `position_start`, `quaternion_start`, `support_weights`, `timestamp_s`, `velocity_start`; max `0.0` |
| C2 lock-step | Pass; 6,530 targets; max `0.0`; no differing tick |
| C3 observable walk002 ideal | Pass; `1417`; leg `0.18202273382760834`; arm `0.04273216819201739`; root p95 `0.36741054963315306` |
| C3 stream walk002 normal | Pass; `scenario_passed True`; `1824` controls |

| Placement/run | 500 Hz: misses; start p50/p95/max ms; work p50/p95/max ms | 50 Hz: misses; start p50/p95/max ms; work p50/p95/max ms | Gap ms | IPC RT p50/p95/max ms | Target age p50/p95/max ms | Verdict |
|---|---|---|---:|---|---|---|
| T-A WSL CPU `ta_run2` | 17; 0.015/0.028/3.815; 0.055/0.085/5.376 | 34; 0.070/0.091/50.701; 10.433/12.622/35.346 | 7.675 | 12.015/14.026/39.904 | 1.340/2.031/3.506 | Fail |
| T-B Windows CPU `tb_run5` | 12; 0.015/0.048/13.151; 0.077/0.129/7.878 | 6; 0.326/0.559/27.693; 12.222/14.059/47.683 | 15.175 | 14.020/16.022/26.029 | 0.886/1.366/2.059 | Fail |

T-B admits all 5,780 teleop packets and returns 5,773 targets. Target age uses
four-timestamp NTP mapping between native and policy monotonic clocks, avoiding
the distinct Windows/WSL wall-clock epochs. The closest isolated native result
was `tb_run2` (0 misses, 3.375-ms gap), but its WSL publisher delivered no
teleop packets to Windows, so it is not acceptance evidence. Valid full runs
still show host scheduling tails. No criterion was relaxed.

Evidence: `E:\codex-artifacts\bfm_teleop_20260917\fix4\`.

Reproduce: build with `cmake --build /root/deploy_build --target g1_true23_bfm_lowcmd_loop`; export/check with `python -m gear_sonic.scripts.run_g1_true23_bfm_split_policy {export-replay,check-batching,lockstep}` using the Fix 3 sensor replay; then run `bash gear_sonic/scripts/run_fix4_wsl_timing.sh <fresh-/mnt/e-output> <port>` for T-A or `./gear_sonic/scripts/run_fix4_windows_policy_timing.ps1 -Output <fresh-E-output> -Port <port>` for T-B. Both scripts remain foreground and loopback-only.

### 2026-09-18 — Review of Fix 4: software correct, timing limited by the platform

**The split architecture is verified correct.** Estimator batching is exactly
equal (0.0 on every field at all 6,530 ticks), lock-step emitted targets match
the single-process harness exactly (0.0 over 6,530 targets), and both regression
checks reproduce. The native loop's DDS surface is compiled to domain 232, `lo`
and a test topic only.

**The remaining misses are platform preemption, not software cost.** The native
500 Hz loop's typical behaviour is close to ideal:

| Native 500 Hz loop | p50 | p95 | max |
|---|---:|---:|---:|
| Start lateness, T-A (WSL policy) | 0.015 ms | 0.028 ms | 3.815 ms |
| Start lateness, T-B (Windows policy) | 0.015 ms | 0.048 ms | **13.151 ms** |
| Work time, T-A | 0.055 ms | 0.085 ms | 5.376 ms |
| Work time, T-B | 0.077 ms | 0.129 ms | **7.878 ms** |

Work that takes 0.08 ms at the median reaching 5–8 ms at the maximum can only
mean the process was descheduled part-way through it. Misses were 17 (T-A) and
12 (T-B) of roughly 65,300 ticks, about 0.02–0.03 percent. One isolated run,
`tb_run2`, reached zero misses with a 3.375 ms gap, but no teleop packets reached
its policy, so it is not acceptance evidence; it does show the tail depends on
load. The 50 Hz policy shows the same pattern (6–34 misses, maxima 27–51 ms).
Target age at send is small (p95 1.4–2.0 ms).

This is the behaviour of a process inside a WSL2 virtual machine whose virtual
CPUs are scheduled by Windows. `SCHED_FIFO` inside the guest does not give
real-time guarantees against the host. No further change inside WSL2 is expected
to reach zero misses reliably, so the software effort on this timing criterion
stops here.

Earlier sessions reached the same boundary from a different direction: the
2026-09 gantry report records PICO packet ages of 90–360 ms under host contention
and concludes "the onboard build is the real fix".

**Decision required from the operator.** The binding requirement was zero missed
deadlines on this PC. The evidence shows that is not reliably achievable on this
PC under Windows and WSL2. The options are to move the 500 Hz loop to a real-time
Linux host (the robot's onboard computer, or this PC booted into native Linux),
or to adopt an explicit, documented tolerance for rare late ticks.

### 2026-09-18 — Fix 5: onboard loop preparation

**Offline checks that passed:** the modified x86_64 native loop built in WSL;
the full local TCP functional replay completed; C1 estimator batching and C2
lock-step target parity both remained exactly `0.0`; the LowCmd source-safety
scan passed; both disconnected-robot scripts failed promptly and clearly; and
the two Windows BFM regressions reproduced exactly. No robot connection was
attempted other than those two explicit reachability checks.

#### Network design

`g1_true23_bfm_lowcmd_loop` is now Fix 5. It binds its two ZeroMQ sockets to
operator-supplied **TCP** endpoints: a 500 Hz `PUSH` stream of LowState-shaped
batches from the loop to the policy and a `PULL` stream of 50 Hz joint targets
from the policy to the loop. `run_g1_true23_bfm_split_policy.py` validates that
both endpoints start with `tcp://` and connects to them. On the robot the loop
binds to `tcp://192.168.123.164:5560` and `:5561`; the Windows policy connects
to those addresses. The local functional test used `127.0.0.1` with the loop in
WSL and the policy in Windows.

The loop retains `ZMQ_DONTWAIT` for its state send and target drain. A delayed
or absent policy cannot block its 2 ms schedule: it continues to send the most
recent target. Target age still uses the existing NTP-style four-timestamp
calculation: loop send (`t1`), policy receive (`t2`), policy output (`t3`), and
loop receive (`t4`). The derived offset `((t2-t1)+(t3-t4))/2` maps policy
monotonic time into loop monotonic time; it does not depend on shared wall-clock
epochs and is therefore valid across the two machines.

Replay remains the default `--source replay`. `--source dds` compiles a
read-only `rt/lowstate` subscription and converts the fixed True23 motor slots,
IMU quaternion, gyro, and accelerometer into the same wire sample. The DDS
factory remains loopback-only even in that mode, deliberately preventing this
preparation executable from becoming a robot-facing command program.

#### Output safety

The only `ChannelPublisher<LowCmd>` in the loop is compiled with domain `232`,
interface `lo`, and topic `rt/fix5_timing_no_robot_lowcmd`. There is no command
line option, environment variable, or configuration setting for another topic
or interface. The policy imports no DDS code. The static test
`test_g1_true23_bfm_lowcmd_loop_safety.py` verifies that the sole publisher uses
the fixed topic symbol and rejects both a production command-topic literal and
an Ethernet-interface literal in the loop source. This is still a replay/timing
qualification path, not an actuation path.

#### Offline results

| Check | Result |
|---|---|
| WSL x86_64 build | Passed: `cmake --build /root/deploy_build --target g1_true23_bfm_lowcmd_loop -j2` linked the modified loop. |
| Local TCP functional replay | Passed as a functional check: 115.6 s / 57,800 ticks; 5,780 teleop packets accepted; 5,773 policy updates and 5,773 targets received. It is not timing acceptance: WSL/Windows observed 8 loop misses, 4 policy misses, and a 7.284 ms maximum LowCmd gap. |
| C1 estimator batching | Passed: all six output fields at 6,530 controls; maximum absolute difference `0.0`. |
| C2 lock-step parity | Passed: 6,530 targets; maximum absolute difference `0.0`; no differing tick. |
| LowCmd source scan | Passed: `pytest .../test_g1_true23_bfm_lowcmd_loop_safety.py -q` reported `1 passed`. |
| Disconnected deployment script | Passed expected absence check: exit `69`, `robot not reachable at 192.168.123.164; cable and power the G1 before deployment`. |
| Disconnected timing runner | Passed expected absence check: exit `69`, `robot not reachable at 192.168.123.164; cable and power the G1 before timing qualification`. |
| Observable walk002 ideal | Passed exactly: completed `1417`; leg `0.18202273382760834`; arm `0.04273216819201739`; root p95 `0.36741054963315306`. |
| Stream walk002 normal | Passed: `scenario_passed True`; `1824` controls. |

Evidence is under `E:\codex-artifacts\bfm_teleop_20260917\fix5\`. The
combined local networking report is `local_tcp_full_2/functional_check.json`;
it explicitly marks that run non-acceptance timing evidence.

#### On-robot procedure

1. Power and cable the G1 so the onboard computer is reachable at
   `192.168.123.164`. From this PC, run:

   ```powershell
   wsl.exe -d Ubuntu-22.04 -- bash /mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/gear_sonic/scripts/deploy_bfm_loop_onboard.sh
   ```

   The script first checks reachability, restores the previously inspected
   temporary host address only if safe, and refuses if
   `/home/unitree/bfm_teleop_fix5/` already exists or if the protected checkout
   is not present. It creates only `/home/unitree/bfm_teleop_fix5/`, copies the
   loop source, a minimal CMake file, and the exported replay/initial-command
   files, locates `unitree_sdk2` in
   `/home/unitree/g1_true23_onboard/` read-only, then builds natively in the new
   directory. It prints the resulting binary path and `file` output; the
   expected architecture is `aarch64`. It does not modify the protected checkout
   or start a robot-facing LowCmd publisher.

2. Run the two required full PICO replay qualifications:

   ```powershell
   powershell -NoProfile -ExecutionPolicy Bypass -File Z:\codex\GR00T-WholeBodyControl-sonic-transfer-23dof\gear_sonic\scripts\run_bfm_onboard_timing.ps1 -Output E:\codex-artifacts\bfm_teleop_20260917\fix5\onboard_20260918_run
   ```

   This starts the onboard binary in replay mode, bound to the two robot TCP
   endpoints, starts the Windows BFM policy and local PICO publisher, copies the
   robot loop reports back into the named output directory, and writes per-run
   and overall JSON summaries. It runs two consecutive 115.6 s replays by
   default. The fixed pass condition is unchanged: each run must have zero 500
   Hz misses, zero 50 Hz misses, and maximum LowCmd gap below 4 ms. A nonzero
   PowerShell exit means the pair did not qualify; inspect `summary.json` rather
   than treating that as authorisation to change thresholds.

3. After the loop and runner have stopped, undo this preparation build, if
   required:

   ```powershell
   wsl.exe -d Ubuntu-22.04 -- bash /mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/gear_sonic/scripts/fix5_onboard_ssh.sh run "rm -rf /home/unitree/bfm_teleop_fix5"
   ```

   This removes only the newly created Fix 5 directory on the onboard computer.
   It does not touch `/home/unitree/g1_true23_onboard/`, the Unitree SDK, system
   services, robot modes, or firmware.

### 2026-09-18 — Fix 6: teleop clock alignment

T1, T2, and T3 passed; T4 was not separately completed; neither on-robot run
was valid because the Windows 50 Hz policy missed deadlines, and run 02 also
fell below the unchanged 5,700-target exchange threshold.

#### Diagnosis and fix

`PacketGate.receive()` verifies the canonical source clock against
`sequence * DT`, then calculates
`source_age = now - epoch_started_at - packet.source_time`.  The precise
failing check was `if source_age > self.stale_seconds + 1e-7: raise
PacketRejected("packet_arrived_stale")`.  The exception path increments the
gate's internal `rejected` counter, latches the fault, and returns `False`;
the previous split-policy report counted only transport/decode failures, which
is why this was silent.

`TeleopClockAlignment` now records `offset = policy_now_at_reception -
packet.source_time` for the first valid packet in each gate epoch.  The pending
delivery filter uses `packet.source_time + offset` in policy time.  The same
offset is passed to PacketGate's unchanged age calculation, accounting for the
gate's epoch origin, while the packet itself retains its original canonical
timestamp for the strict 50 Hz sequence-clock check.  `PacketGate.rearm()`
creates a new epoch and the alignment re-anchors on its first packet.  No
staleness bound, gate threshold, BFM input/weight, or brake setting changed.

Both `run_g1_true23_bfm_split_policy.py` and
`run_g1_true23_bfm_teleop_sim.py` use this path.  Their reports now include
`teleop_packets_gate_rejected`, `teleop_gate_rejection_reasons`, and the
per-epoch reception anchors.

#### Test results

| Check | Result |
|---|---|
| T1: synthetic policy/publisher offsets of policy 5 s before, 5 s after, and simultaneous start | Passed: all eight packets accepted in order in all three cases; every computed source age stayed within the unchanged 0.100 s limit. |
| T2: true delayed packet after anchoring | Passed: rejected and latched as `packet_arrived_stale`; the 0.100 s bound is unchanged. |
| Stream/brake protocol suite | Passed: `python -m pytest gear_sonic/tests/test_g1_true23_bfmzero_stream.py -q` — 38 passed. |
| T3 observable `walk002` ideal | Passed: completed `1417`; leg `0.18202273382760834`; arm `0.04273216819201739`; root p95 `0.36741054963315306`. |
| T3 stream `walk002` normal | Passed: `scenario_passed True`; `1824` controls. |
| T4 local two-process delayed-start check | Not completed as a separate local test before qualification. The on-robot two-process run below did exercise two distinct observed alignments, including a 2.140 s policy/publisher offset, but it is not represented as a T4 pass. |

#### On-robot qualification

The first invocation at `fix6/onboard_timing` is retained as failure evidence:
the policy exited with `NameError: teleop_clock is not defined` before producing
a report.  The initialization was moved from the offline lock-step helper to
`policy_process`, the focused suite was rerun (38 passed), and the unchanged
launcher was rerun at `fix6/onboard_timing_retry`.  It used the existing
`/home/unitree/bfm_teleop_fix5/run_loop.sh` deployment unchanged.  No command
was published to `rt/lowcmd` or any robot-facing interface.

| Run | targets received | 500 Hz misses; LowCmd gap p50/p95/max ms | 50 Hz misses; work p50/p95/max ms | Teleop accepted / gate-rejected | Target age p50/p95/max ms | Verdict |
|---|---:|---|---|---|---|---|
| run_01 | 5744 | 0; 2.000/2.022/2.535 | 21; 12.179/14.342/116.491 | 5780 / 0 | 1.187/2.068/73.407 | Invalid: 50 Hz misses. |
| run_02 | 5634 | 0; 2.000/2.022/3.102 | 37; 12.204/14.015/115.792 | 5673 / 0 | 1.150/1.947/82.484 | Invalid: target count below 5700 and 50 Hz misses. |

Run 01's reception anchor was `0.000 s`; run 02's was `2.140 s`.  Both had an
empty gate-rejection-reason map, confirming that the previously silent clock
fault did not recur.  The unchanged qualification criterion remains unmet:
each run must have at least 5,700 targets, zero 500 Hz misses, zero 50 Hz
misses, and a maximum LowCmd gap below 4 ms.

### 2026-09-18 — Review of Fix 6, a contention control, and the decision to benchmark BFM on the Orin

**Clock alignment fixed and verified on the robot.** The failing check was
`PacketGate.receive` raising `packet_arrived_stale` when the policy clock and the
publisher clock started at different times. After anchoring at first reception,
a run with a 2.140 s start offset accepted 5,673 teleop packets with zero gate
rejections; the same situation previously accepted none. T1, T2, T3 and the
38-test suite pass. **T4, the separate local delayed-start test, was not run by
codex**; the on-robot 2.140 s case covers the same condition in practice but is
not recorded as a T4 pass.

**The robot-side 500 Hz loop is qualified in every run that exchanged data.**
Across six full 115.6 s runs with the policy connected over Ethernet — Fix 5
`onboard_timing_2/run_01` and `onboard_timing_3/run_01`, Fix 6 retry runs 1–2 and
quiet runs 1–2 — the loop on the Jetson Orin had **zero missed deadlines** and a
largest LowCmd gap of **3.102 ms**, as an unprivileged process without
`SCHED_FIFO` or memory locking.

**The Windows 50 Hz policy stalls intermittently.** In the Fix 6 retry the
policy's worst step was about 116 ms in both runs, with stale-target episodes at
8.8, 46.9, 79.8 and 112.4 s in run 2 — mid-run, not start-up. Because codex was
itself running on the same PC during those runs, the qualification was repeated
with codex idle:

| Quiet rerun | Targets | Robot 500 Hz misses / gap max | PC 50 Hz misses | PC work max | Stale episodes |
|---|---:|---|---:|---:|---|
| run_01 | 5749 | 0 / 2.312 ms | 17 | 118.45 ms | one, at 22.0 s |
| run_02 | 5651 | 0 / 2.544 ms | 6 | 36.79 ms | one, at 27.3 s |

Contention made it worse (four episodes per run became one), but stalls of up to
about 118 ms persist on this Windows desktop without it. During walking, a target
held for 100 ms is a real risk even though the robot loop keeps sending.

**Validity threshold note.** run_02 falls below the 5,700-target validity check
only because clock anchoring legitimately starts it about 2 s later; the exchange
itself was complete. That fixed threshold should scale with the measured start
offset. It is recorded here rather than silently changed.

**Operator decision (2026-09-18): benchmark BFM on the Jetson Orin.** If the
Orin can run the estimator, admission, goal construction and BFM inference within
the 20 ms budget, the whole control path runs on the robot under native Linux and
this PC only streams teleop input. The Orin has internet access over wlan0, is in
MAXN power mode, and has Python 3.8 only, so the benchmark environment will be
installed in an isolated, removable directory.

### 2026-09-18 — Orin BFM benchmark: correct results, far too slow

**C1 passed; the feasibility criterion is not close to being met, and the Orin
CPU is ruled out as the host for the 50 Hz policy.**

The isolated environment was installed under `/home/unitree/bfm_orin_bench/`
without sudo and without touching system Python, the power mode, the governor,
`/home/unitree/g1_true23_onboard/` or `/home/unitree/bfm_teleop_fix5/`.

**C1, numerical equivalence:** BFM actor outputs on the Orin match the Windows
reference to a maximum absolute difference of **8.64e-7** (limit 1e-5), and every
estimator field differs by less than 1e-6. The Orin computes the same control.

**B1, cost at one Torch thread, full PICO replay:**

| Quantity | p99 on Orin |
|---|---:|
| BFM inference | **97.22 ms** |
| Whole 50 Hz step | **166.33 ms** |

The budget is 20 ms, and the feasibility bar was a p99 under 15 ms. The single
component that dominates, BFM inference, is on its own about five times the whole
budget. Thread counts 2, 4 and 6, and both paced B2 runs, were not completed, but
no plausible threading gain — 2 to 4 times at best on 8 ARM cores — closes a gap
of this size. For comparison, the same inference costs 3.8 ms on this PC's CPU.

**Conclusion: the Orin's ARM CPU cannot host this policy at 50 Hz.** A GPU path
on the Orin is not ruled out by this measurement — the BFM ONNX export and
TensorRT exist on the robot — but that is a separate piece of work, and JetPack's
CUDA PyTorch wheels are built for Python 3.8 while this environment needs 3.10.

**Interruption.** The benchmark stopped part-way because the robot Ethernet link
went down: both host adapters report `Media disconnected` and `192.168.123.164`
no longer answers. This is a physical disconnection, not the WSL mirrored-network
glitch. No network configuration was changed. A benchmark process may still be
running on the Orin under `nohup`.

**Cleanup, when the robot is next reachable:**
`ssh unitree@192.168.123.164 'pkill -f bfm_orin_bench; rm -rf /home/unitree/bfm_orin_bench'`
The separate, working Fix 5 loop deployment at `/home/unitree/bfm_teleop_fix5/`
should be kept.
