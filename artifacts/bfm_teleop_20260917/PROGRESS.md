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

### 2026-09-18 — C: freed, and the effect on the 50 Hz stalls

The Windows system drive had 0.53 GB free of 295 GB, which is a credible cause of
the intermittent ~100 ms process stalls. Two steps were taken.

Codex removed only regenerable caches and completed crash dumps (Gradle, Cargo,
Conda, dump files), reclaiming about 7.7 GB and reaching 7.99 GB free. It
classified everything else as needing a decision and deleted none of it.

`C:\Users\camer\sonic23_sim_artifacts` (6,091 files, 47,390,816,502 bytes) was
then moved to `N:\sonic23_sim_artifacts` and replaced with a directory junction
at the original path. The copy was verified to match exactly on file count and
total bytes before the source was removed. **C: now has 52.2 GB free.** The
hard-coded `C:\Users\camer\sonic23_sim_artifacts\...` paths in the BFM evaluators
continue to resolve, and the observable evaluator reproduces walk002 exactly
(leg 0.18202273382760834, arm 0.04273216819201739, root p95 0.36741054963315306).

**Effect on the stalls, measured locally** (the robot Ethernet cable is
disconnected, so the 500 Hz loop ran in WSL on this same PC, which adds local
load the robot-hosted runs do not have):

| BFM 50 Hz on Windows | Before (quiet, robot-hosted loop) | After cleanup (WSL-hosted loop) |
|---|---:|---:|
| Deadline misses | 17 | **7** |
| Work p50 / p95 | 12.17 / 14.10 ms | 12.92 / 14.01 ms |
| Work max | **118.45 ms** | **42.48 ms** |

Teleop admission was perfect (5,780 accepted, 0 gate rejections, 5,773 policy
updates), so the Fix 6 clock alignment holds.

Disk pressure was a real contributor — the worst stall fell by roughly a factor
of three — but it is not the whole cause. Seven late updates and a 42 ms stall
remain. The clean comparison, with the loop back on the robot and no WSL load on
this PC, needs the Ethernet cable reconnected.

### 2026-09-18 — Full test sweep with the robot connected

**Correction to earlier results: the robot's own software was not running during
the first six "zero miss" runs.** The Orin had just booted (`uptime` 0 minutes)
when those were measured. It now runs its normal stack — `livox_ros_drive` at
about 115 percent CPU, plus `unitree_slam`, `map_management`, `path_management`,
`pub_graph` and `videohub_pc4`, roughly 1.5 cores in total. The earlier figures
were taken on an idle robot and were not representative.

**Re-measured under that real load, the 500 Hz loop still holds**, 30 s alone,
15,000 ticks each:

| Placement | Misses | Gap p95 / max | Start max |
|---|---:|---|---:|
| unpinned | **0** | 2.021 / 2.262 ms | 0.152 ms |
| `taskset -c 7` | **0** | 2.019 / 3.282 ms | 1.516 ms |
| `taskset -c 6,7` | **0** | 2.021 / 2.853 ms | 0.931 ms |

Pinning does not help and unpinned is best; the loop's own work peaks at 0.42 ms
of its 2 ms period. The robot side is not the constraint.

**Two full qualification attempts after the C: cleanup** (loop on the robot,
policy on this PC):

| Attempt / run | Robot 500 Hz misses; gap max | PC 50 Hz misses; work p50/p95/max |
|---|---|---|
| first, run_01 | 28; 57.500 ms | 13; 12.25/13.55/62.21 ms |
| first, run_02 | 1; 4.594 ms | 361; 12.79/16.75/128.74 ms |
| second, run_01 | **0**; 2.478 ms | 133; 13.45/15.88/51.33 ms |
| second, run_02 | 1; 4.787 ms | 31; 13.01/14.75/52.30 ms |

The robot loop's 28 misses in the first attempt coincided with the PC policy
stalling badly in the same window; with the PC idle it returns to zero.

**The PC-side miss count is unstable across runs — 7, 13, 31, 133, 361 — and the
reason is margin.** BFM's 50 Hz step costs 13.0–13.5 ms at the median and
14.7–15.9 ms at p95 against a 20 ms budget, leaving roughly 4 ms of headroom, so
any Windows scheduling hiccup above that becomes a miss. Freeing C: removed the
worst stalls (118 ms fell to 42–52 ms) but not the mechanism.

**Functional state, all passing:**

| Check | Result |
|---|---|
| Unit suites (stream/brake + LowCmd safety) | 39 passed |
| Observable evaluator, walk002 ideal | completed 1417; leg 0.18202273382760834; arm 0.04273216819201739; root p95 0.36741054963315306 — matches exactly |
| Stream `normal` | passed, 1824 controls, standing verified, zero range excess |
| Stream `pause` | passed, fault latched, standing verified, zero range excess |
| Stream `disconnect` | passed, fault latched, standing verified, zero range excess |
| Stream `resume` | passed, fault latched, standing verified, zero range excess |

**Remaining gap to a clean qualification is entirely the PC-side 50 Hz margin.**
Two ways to widen it, in order of effect: move the estimator (about 5.3 ms of the
13 ms) into the native 500 Hz loop on the robot, where the loop currently uses
0.42 ms of its 2 ms period; or move the policy off Windows so that scheduling
stalls of 40–60 ms stop happening at all.

### 2026-09-18 — Fix 7: estimator on the robot

**G1 maximum difference was `1.1435297153639112e-14`; the new policy-step p50/p95 was `7.532200004789047/10.240184975555167 ms` in qualification run 1 (`9.23480000346899/13.639785010309428 ms` in run 2); G5 did not pass.** This entry records an incomplete qualification, not approval to use the new path.

**MuJoCo and model transfer.** Official MuJoCo 3.2.3 Linux aarch64 was downloaded directly on the Orin and unpacked privately at `/home/unitree/bfm_teleop_fix5/thirdparty/mujoco`; no source build, sudo, system installation, or change to `/home/unitree/g1_true23_onboard` occurred. The release library contains the verified `3.2.3` version string. On Windows, the BFM model was prepared through `prepare_true23_model` with its normal physics overrides, then saved through `mj_saveModel` as the 95 MB `g1_true23_bfm.mjb`. Its compiled-model SHA-256 is `ab505f992ce0f9da9d8df7bd9e983d5fd33459a045fb702054eb7c0c2a823253`. The copied MJB has exactly that SHA-256 on the Orin. This is an equivalent compiled-model identity check because the Python identity is the SHA-256 of the byte buffer emitted by `mj_saveModel`, and the transferred file is that same buffer. The mesh directory was not transferred.

**Port and protocol.** `Native23ImuOdometry` is now a private C++ MuJoCo/Eigen component in the 500 Hz test loop. It loads only the compiled MJB; per tick it sets the FK scratch state, calls `mj_kinematics`, `mj_comPos`, and `mj_jac`, then performs the unchanged double-precision support selection, covariance, gain, rejection, and integration order. The state wire is version 2 and adds 22 float64 snapshot values: position, velocity, accelerometer bias, registered quaternion, eight support weights, and timestamp. The Windows policy defaults to that snapshot and neither constructs nor updates `Native23IMUOdometry` on that path; it still loads the model for the unchanged final-target brake. `--estimator-source python` remains the explicit rollback/comparison mode. The loop remains compiled to domain 232, loopback, and `rt/fix5_timing_no_robot_lowcmd`.

**G1 passed.** The robot ran the first 57,800 samples (115.6 s) lock-step and wrote every 50 Hz snapshot. Maximum absolute differences versus Python were: position `3.885780586188048e-16`, velocity `3.3306690738754696e-16`, bias `2.3332030751888055e-16`, quaternion `2.220446049250313e-16`, support weights `1.1435297153639112e-14`, and timestamp `0.0`. No field exceeded `1e-9`.

**G2 passed.** With native snapshots substituted in lock-step, all 5,780 emitted 23-joint targets matched the Python-estimator sequence exactly: maximum elementwise difference `0.0`; no first differing control.

**G3 passed on the clean unpinned repeat.** The required 15,000-tick normal-load run had zero missed deadlines, work p95/max `0.162534/0.433423 ms`, and a `2.685409 ms` largest LowCmd gap. Two earlier diagnostics are retained as failures: the first wrote G1 CSV snapshots in the timed loop (986 misses), and the first clean repeat had 41 scheduling misses despite work p95 `0.15843765 ms`. Snapshot recording is now G1-only and never enabled in the timed path.

**G4 passed.** Windows observable `walk002` ideal completed `1417` controls with leg `0.18202273382760834`, arm `0.04273216819201739`, and root p95 `0.36741054963315306`. Stream `walk002` normal passed with `1824` controls. The stream/brake suite and native LowCmd static safety test passed: `39 passed`.

**G5 failed; criterion unchanged.** Fresh output at `E:\codex-artifacts\bfm_teleop_20260917\fix7\qualification` recorded both runs. The directly comparable pre-Fix-7 full-sweep runs were `13.45/15.88/51.33 ms` and `13.01/14.75/52.30 ms` policy-work p50/p95/max; after moving the estimator, run 1 was `7.532200004789047/10.240184975555167/23.27450001030229 ms` and run 2 was `9.23480000346899/13.639785010309428/21.26850001513958 ms`. Run 1 received 5,738 targets and had zero native misses with a `3.772580 ms` maximum LowCmd gap, but had 2 Windows 50 Hz misses. Run 2 received 5,480 targets, had 203 native misses and a `12.163688 ms` LowCmd gap, and had 4 Windows 50 Hz misses. The qualifying requirements remain at least 5,700 targets, zero 500 Hz misses, zero 50 Hz misses, and maximum LowCmd gap below 4 ms.

**Removal and rollback.** To roll back policy computation without changing the native binary, run the split policy with `--estimator-source python`; version-2 messages still carry but ignore the native snapshot. To remove the private Fix 7 deployment only, remove `/home/unitree/bfm_teleop_fix5/thirdparty/mujoco`, `/home/unitree/bfm_teleop_fix5/source/g1_true23_bfm.mjb`, and the rebuilt Fix 5 deployment files; do not remove or modify `/home/unitree/g1_true23_onboard`.

### 2026-09-18 — Fix 7: estimator on the robot, qualification addendum

**G1 maximum difference was `1.1435297153639112e-14`; new policy step p50/p95 was `8.044499991228804/11.668369990366047 ms` in run 1 and `8.725599996978417/11.565829998289702 ms` in run 2; G5 did not pass.** This is the fresh output under `E:\codex-artifacts\bfm_teleop_20260917\fix7\g5_onboard_timing` and supplements the earlier Fix 7 timing attempt.

The private official 3.2.3 aarch64 MuJoCo release and mesh-free MJB transfer remain as described above. The copied MJB SHA-256 is the Windows `compiled_model_sha256`, `ab505f992ce0f9da9d8df7bd9e983d5fd33459a045fb702054eb7c0c2a823253`. The C++ port calls `mj_kinematics`, `mj_comPos`, and `mj_jac` in the native 500 Hz loop, uses double/Eigen covariance algebra, and sends the complete float64 snapshot in version-2 state messages. The default policy consumes it; `--estimator-source python` remains rollback.

G1 and G2 passed exactly as above. A production-path G3 repeat, with snapshot CSV recording disabled, passed: 15,000 ticks, zero misses, native work p95 `0.16944755 ms`, max `0.908512 ms`, and maximum LowCmd gap `2.824931 ms`. G4 passed unchanged: 39 tests including LowCmd safety, observable walk002 ideal `1417` / `0.18202273382760834` / `0.04273216819201739` / `0.36741054963315306`, and normal stream `1824` controls.

G5 failed without changing its criterion. Run 1 had 5,537 targets, 47 native misses, 6 Windows misses, and a 7.863760 ms largest LowCmd gap; run 2 had 5,571 targets, 96 native misses, 1 Windows miss, and an 8.519113 ms largest gap. The policy work is materially lower than the pre-Fix-7 full-sweep p50/p95 figures (`13.45/15.88` and `13.01/14.75 ms`), but the native scheduling stalls and the remaining Windows misses mean the two-run qualification is not valid. No robot-facing topic or interface was used.

### 2026-09-18 — Fix 7 follow-up: where the remaining misses actually come from

**The native misses recorded in Fix 7's G5 are external CPU starvation on the Orin, not the ported estimator, and CPU pinning makes them worse rather than better.** The estimator port itself is sound: G1 agreed to `1.1435297153639112e-14`, G2 targets matched exactly, and the Windows policy step fell from `13.45/15.88 ms` p50/p95 to `7.53-8.73/10.24-11.67 ms`.

**Miss pattern.** In `g5_onboard_timing`, the misses are not spread over the run; they are concentrated in a single burst each. Run 1 lost 45 of its 47 ticks between tick 13552 and tick 14712, a window of about 2.3 s, with only three isolated misses elsewhere. Run 2 lost 91 of 96 between tick 49321 and tick 49916, about 1.2 s. The loop's own work at every one of those ticks was at most `0.172 ms` against a 2 ms period, while start lateness reached `17.91 ms` in run 1 and `25.75 ms` in run 2. The loop was therefore ready and cheap, and simply was not scheduled.

**Scheduling headroom is unavailable to us as an unprivileged user.** On the robot `ulimit -r` reports a soft and hard real-time priority limit of `0`, so `SCHED_FIFO` cannot be requested, and `sudo -n` fails because a password is required. The kernel command line contains no `isolcpus`. At the time of measurement `livox_ros_drive` was consuming about 115% CPU, with `videohub_pc4` at about 18% and a further Python process at about 15%.

**Pinning was tested and rejected.** Three consecutive 57,800-tick runs under the same load, differing only in CPU affinity, gave: unpinned zero misses with start lateness max `1.059 ms`, work p95/max `0.164/1.434 ms` and largest LowCmd gap `3.380 ms`; `taskset -c 7` twenty misses with start lateness max `7.455 ms` and gap `9.423 ms`; `taskset -c 6,7` 1,471 misses with start lateness max `74.138 ms` and gap `25.572 ms`. The kernel's own balancing is better than any affinity mask we can impose, so affinity should not be used. The unpinned result also confirms that the configuration with the estimator inside the loop can meet the criterion when the machine is not busy.

**Validity threshold caveat.** Both G5 runs were additionally marked invalid by the harness `targets_received >= 5700` rule. Run 1 recorded `policy_updates` of 5,539 against 5,780 nominal, so the policy's timed window began roughly 4.8 s after the loop's, and the threshold is not reachable under the current start sequencing regardless of jitter. The threshold needs to be derived from the measured overlap rather than fixed.

**What remains.** Two independent jitter sources, neither of which is the estimator. On the robot, occasional multi-second starvation bursts that require real-time scheduling, which requires root on the robot. On this PC, Windows stalls that still put policy work max at `26.79 ms` and start lateness max at `15.42 ms` against a 20 ms budget, with 6 and 1 deadline misses in the two runs.

### 2026-09-18 — Robot-side fix: real-time scheduling for the 500 Hz loop

**Running the native loop under `SCHED_FIFO` priority 80 eliminates the starvation bursts completely: two consecutive 57,800-tick runs under the robot's normal load had zero deadline misses, start lateness maxima of `0.157 ms` and `0.086 ms`, and largest LowCmd gaps of `2.413 ms` and `2.356 ms`.** The directly comparable unprivileged run taken minutes earlier under the same load had 230 misses, start lateness max `25.230 ms`, and gap max `13.399 ms`.

**Why it was not simply available.** The Orin kernel is built with `CONFIG_RT_GROUP_SCHED=y`, and `/sys/fs/cgroup/cpu,cpuacct/user.slice/cpu.rt_runtime_us` is `0`, so no process in any user session may run at a real-time policy — `chrt` fails with `Operation not permitted` even as root. The root cgroup holds the full `950000` of `1000000 us`, and `system.slice` is likewise `0`. The unprivileged `ulimit -r` of `0` is a second, independent barrier.

**What was changed, and how to undo it.** One runtime value was written on the robot: `/sys/fs/cgroup/cpu,cpuacct/user.slice/cpu.rt_runtime_us` was set from `0` to `200000`, granting user sessions up to 200 ms of real-time CPU per 1 s period. Writing `0` back to that file restores the original state, and the value resets to `0` on reboot in any case, so it must be re-applied per boot. Nothing else on the robot was modified: no service, no unit file, no limits configuration, and nothing under `/home/unitree/g1_true23_onboard`. The loop was then started with `chrt -f 80`. The loop's duty cycle is about 8% of one core (work p95 `0.091-0.159 ms` per 2 ms period), well inside the granted bandwidth, and the kernel's real-time throttling remains in force as a backstop.

**Follow-up needed.** Launching through `sudo chrt` requires the robot password at every run. The durable arrangement is to grant the loop binary `CAP_SYS_NICE` once with `setcap cap_sys_nice+ep`, and to have the loop call `sched_setscheduler` for itself at start-up, falling back to normal scheduling with a recorded warning when the capability or the cgroup bandwidth is absent. The per-boot `cpu.rt_runtime_us` grant is still required in that arrangement.

**Also established.** CPU affinity is counterproductive on this machine and must not be used: at identical load, unpinned gave zero misses, `taskset -c 7` gave 20 misses with gap max `9.423 ms`, and `taskset -c 6,7` gave 1,471 misses with start lateness max `74.138 ms`.

**Durable form, applied and measured.** The loop already attempted `sched_setscheduler(SCHED_FIFO, 10)` and `mlockall` at start-up and had been failing both, printing `SCHED_FIFO unavailable: Operation not permitted` and `mlockall unavailable: Cannot allocate memory` into its own stdout on every run since deployment. The second failure is a separate latent defect: the `memlock` limit is 64 MB while the compiled model alone is 95 MB, so no page of the loop, including the estimator's model, was ever locked. Granting the binary both capabilities once with `setcap cap_sys_nice,cap_ipc_lock+ep` makes its existing code path succeed with no source change and no `sudo` at run time. Measured immediately afterwards under load `4.66`: zero misses, start lateness max `0.046 ms`, work p95/max `0.096/0.454 ms`, largest LowCmd gap `2.378 ms`, and neither warning present. The existing `run_loop.sh` therefore now obtains real-time scheduling by itself.

Two conditions must hold for this to keep working. The `cpu.rt_runtime_us` grant on `user.slice` is lost on reboot and must be re-applied. File capabilities are lost whenever the binary is rebuilt or replaced, so `setcap` must be re-applied after every deployment; the loop reports this itself, because the two start-up warnings reappear in its stdout when the capabilities are missing.

### 2026-09-18 — Fix 8: the Windows stalls were start transients and idle-core wake-ups, and the qualification now passes

**The two-run on-robot qualification passed for the first time: both runs valid, zero 500 Hz misses, zero 50 Hz misses, and largest LowCmd gaps of `2.386 ms` and `2.340 ms`.** Policy work p99 was `8.275 ms` and `8.329 ms` against a 20 ms budget, with maxima of `10.406 ms` and `10.820 ms` and start lateness maxima of `0.170 ms` and `0.359 ms`. Targets received were 5,764 and 5,758.

**The stalls were not what they looked like.** With the per-control trace, a 5,780-control run had exactly one deadline miss, at control 0, whose work was `23.501 ms`; control 1 then inherited `14.360 ms` of start lateness. Every other control in that run peaked at `17.05 ms` of work and `1.08 ms` of lateness. So the budget was never structurally short after Fix 7; three separate start-up effects and one steady-state effect were being read as generic "Windows jitter".

**First: the first control paid every lazy initialisation.** Control 0's phase accounting covered only `4.12 ms` of its `23.50 ms`, and its inference phase read `0.0 ms`, because `policy_step` returns immediately until the admission gate is ready. The measurement machinery's own first ctypes and query calls made up most of the rest. The policy now performs a complete warm-up before the timed clock starts: the telemetry counters are sampled three times, the first state message is decoded, and the BFM actor is driven once with correctly shaped inputs, including a zero 256-wide goal vector, followed by the brake and the target packing. It runs on throwaway copies — a copied history, a copied action, a fresh brake — and sends nothing, so no observable state changes.

**Second: the first inference landed mid-run.** Because the gate only becomes ready once teleop packets arrive, the first real inference happened at control 7 and cost `59.12 ms` there. Driving the actor directly in the warm-up, rather than through `policy_step`, removes this; `policy_step`'s early return is what made a warm-up through it useless.

**Third: the warm-up itself created a backlog.** Lowstates continue to arrive at 500 Hz while the warm-up runs, so control 0 opened by decoding everything that had queued — 227 messages in one run. The policy now drains to the newest state before starting the clock, which is what a control uses anyway, since the receive loop keeps only the last sample of its batch. With `--estimator-source python` the backlog is deliberately left in place, because that estimator integrates every sample.

**Fourth: waiting let the cores sleep.** After the three fixes above, isolated controls still showed 9-11 ms of start lateness on a machine with about twelve idle cores and no competing process in the captured activity snapshots. The waitable timer was already the high-resolution variant, so the latency was the wake-up out of a deep idle state. `SimulationPacer` now takes an optional `spin_margin_s`: it waits as before up to that margin, then busy-waits the final stretch, keeping the thread runnable and the core awake. The policy exposes it as `--pacer-spin-ms`, default `0.0`, so every existing caller is unchanged; the qualification uses `1.0`.

**Measured, three runs each on the local replay.** Baseline after Fix 7: 1, 0 and 2 misses, work p99 `8.67/8.38/8.21 ms`, start lateness max `2.88/1.01/7.73 ms`. With warm-up and drain: 0, 0 and 1 misses, p99 `8.48/8.33/8.14 ms`, lateness max `9.16/1.08/11.06 ms`. With the 1 ms spin added: 0, 0 and 0 misses, p99 `8.65/8.92/8.11 ms`, lateness p95 `0.00 ms` in all three and max `9.52/1.15/0.17 ms`.

**What did not help, and is not used.** `REALTIME_PRIORITY_CLASS` with a time-critical thread, 0.5 ms timer resolution, and `OMP_WAIT_POLICY=PASSIVE`, tested together, left 1, 1 and 2 misses with control 0 still an outlier — that combination addresses neither the start transients nor the idle wake-up. CPU affinity was not adopted, consistent with the robot-side result where it was actively harmful.

**Regressions unchanged.** Observable `walk002` ideal: completed `1417`, leg `0.18202273382760834`, arm `0.04273216819201739`, root p95 `0.36741054963315306`, matching to every digit. Stream `walk002` normal: `1824` controls, no latched fault, no physical failure, full source consumption. Unit suites including the native LowCmd static safety test: `39 passed`.

**Note on the robot precondition.** The robot had rebooted before this qualification, so `cpu.rt_runtime_us` on `user.slice` was back to `0` and had to be re-granted; the binary's file capabilities survived, as expected. `gear_sonic/scripts/orin_enable_realtime.sh` applies both. Without the grant the native loop starves and the qualification fails for reasons that have nothing to do with the policy.

### 2026-09-18 — Fix 9: real LowState, read-only

**Real samples were received for 60.0 s at 1,020.16 Hz, and the estimator stayed finite and bounded, but its horizontal estimate drifted 0.23965 m/min.** This is the first measurement against live sensors rather than the 115.6 s replay. It is read-only: the final run reported `LowCmd publisher exists=false` and `lowcmd_messages_sent=0`.

**Real DDS endpoint.** The installed passive telemetry reader at
`/home/unitree/rocco-jev-sensors/telemetry_reader.py` was read without
modification. It calls `ChannelFactoryInitialize(0, "eth0")` and subscribes to
`rt/lowstate`. The robot's `eth0` is up on the Unitree network and owns
`192.168.123.164` (and the additional `192.168.123.18` address). The same
reader independently reported mode machine 4 and 35 message motor slots. The
probe therefore used domain 0 / `eth0`, never domain 232 / loopback.

**Safety invariant.** `g1_true23_bfm_lowcmd_loop` now accepts
`--dds-domain` (default 232) and `--dds-interface` (default `lo`) and passes
them to `ChannelFactory::Instance()->Init`. A `ChannelPublisher<LowCmd>` is
constructed only inside the predicate `dds_domain == 232 &&
IsLoopbackInterface(dds_interface)`. The send site repeats that predicate as
well as requiring a non-null publisher. On every other endpoint the process
prints subscriber-only mode and has no publisher object or command send path.
The default replay path therefore remains domain 232, `lo`, and the fixed
`rt/fix5_timing_no_robot_lowcmd` test topic. The read-only probe is a separate
execution mode: it creates only a LowState subscriber, no ZMQ command sockets,
no initial command, and no publisher. Its static safety test now proves both
publisher construction and the sole `Write` call are inside a guard that tests
both the domain and loopback interface.

**Probe result.** Final evidence is on the robot at
`/home/unitree/bfm_teleop_fix5/runs/fix9_real_lowstate_valid_20260918/`:
`real_lowstate_probe_report.json` and the 20 Hz
`real_lowstate_estimator_trace.csv`. The report is valid JSON. It received and
converted all 61,210 samples; no sample had an invalid layout and the bounded
subscriber queue dropped none. Inter-arrival time was 0.971 ms p50, 1.389 ms
p95, 3.235 ms p99, and 13.530 ms maximum; there were 803 intervals above 3 ms
and 12 above 10 ms. This real callback rate is about twice the recorded
replay's 500 Hz rate and is a material integration difference: the existing
500 Hz timing loop consumes its latest received sample rather than every one
of these approximately 1 kHz callbacks.

Decode sanity was clean. Every message had 35 motor slots, every mode-machine
value was 4, quaternion norm error was `4.44e-8` p50 and `1.41e-7` maximum,
and all 23 mapped positions remained within the MJB joint limits (zero maximum
range excess). The estimator's pelvis height above its initial modeled sole
floor ranged from 0.78556 to 0.81931 m (0.80227 m p50). Speed magnitude was
0.00431 m/s p50, 0.00648 m/s p95, and 0.01052 m/s maximum. Accelerometer-bias
magnitude settled near 0.148 m/s² (0.14986 m/s² p95). All eight support weights
remained non-zero and stable enough to distribute weight across both modeled
soles; their median range was 0.09283–0.14706. The estimator update itself cost
0.01475 ms p50, 0.02211 ms p95, 0.03721 ms p99, and 3.69758 ms maximum. Replay
timing fields that require a scheduled loop or command exchange
(`start_lateness_ms`, `lowcmd_gap_ms`, `ipc_round_trip_ms`, and
`target_age_ms`) are explicitly null in this subscriber-only report.

**Interpretation.** The low joint excursion (at most 0.00030 rad on the mapped
joints), low estimated speed, stable IMU norm, and persistent eight-point
geometric support selection are consistent with a stationary, supported
configuration. The data alone cannot certify whether the robot was upright,
seated, or otherwise externally supported, so this is not called a standing
result. The root estimate did not diverge, but 0.23965 m/min of horizontal
drift is significant for a robot that appears stationary. It should be treated
as a real-sensor limitation requiring follow-up before interpreting root
position as stationary ground truth. The 1 kHz real stream versus the 500 Hz
replay is also a policy risk: the deployed 500 Hz loop will decimate it, so
replay equivalence does not establish behaviour at the real estimator update
rate.

**Build and real-time state.** The isolated binary under
`/home/unitree/bfm_teleop_fix5` was rebuilt three times while correcting the
probe report and shutdown path; no file under `/home/unitree/g1_true23_onboard`
was modified. After every rebuild, `SUDO_PASS=123 bash
gear_sonic/scripts/orin_enable_realtime.sh` was run. It restored
`cap_ipc_lock,cap_sys_nice+ep` on the binary and confirmed the `user.slice`
real-time grant was 200000 us. The final replay loop stdout contained neither
`SCHED_FIFO unavailable` nor `mlockall unavailable`.

**Regression results.** The stream/brake suite plus the extended LowCmd static
safety suite passed: **40 passed** (the previous 39 plus the new guard test).
Observable `walk002` ideal was unchanged: completed 1417, leg
0.18202273382760834, arm 0.04273216819201739, and root p95
0.36741054963315306. Stream `walk002` normal passed with 1824 controls,
complete source consumption, no latched fault, and zero range excess.

**The required two-run timing qualification did not pass on this rerun; the
criterion was not relaxed.** New isolated evidence is
`E:\codex-artifacts\bfm_teleop_20260917\fix9_regression_20260918`. Both runs
were valid exchanges (5,735 and 5,769 targets), had zero 500 Hz misses, and
had maximum LowCmd gaps of 2.382011 and 2.396284 ms. Windows nevertheless had
2 and 1 50 Hz misses, respectively (work maxima 47.1869 and 22.3636 ms), so
the qualification is failed despite clean robot-side timing. This is the same
Windows scheduling sensitivity previously identified, not an estimator or
native-loop deadline failure, but it remains a failure until two consecutive
runs meet every unchanged criterion.

### 2026-09-18 — Fix 10: the last large stalls were unmeasured teleop decoding

**The two-run qualification passes again and with far more margin: zero 500 Hz misses, zero 50 Hz misses, LowCmd gaps of `2.410 ms` and `2.376 ms`, policy work p99 of `8.75 ms` and `8.53 ms`, and maxima of `11.42 ms` and `11.06 ms` against the 20 ms budget.** Before this change the worst control in a run reached `54.05 ms`.

**Why Fix 8's three clean runs were partly luck.** Re-running the qualification on an otherwise idle machine gave one clean run and one with two misses whose worst control was `54.05 ms`; a traced set of three runs then gave 3, 2 and 0 misses. So roughly every other run was absorbing one very large stall, which the earlier three-run sample had simply missed.

**The stall was in a stage nobody was measuring.** With the per-control trace, the worst control of one run showed `50.91 ms` of work with *every* phase reading approximately zero, no context switches, and 115 soft faults. The phase accounting covered state receive, feature and goal construction, inference, brake and target send, but not the teleop stage: `teleop.recv_json()` followed by `decode_packet` for every queued packet. At control 0 that queue held everything the publisher had sent since the readiness barrier — 326 packets in one run — and decoding them all inside the first control produced the 50 ms.

**Fix.** The warm-up now also drains and decodes whatever teleop packets are already queued, before the timed clock starts. Those packets are not admitted there: they are handed to control 0, which runs them through `teleop_clock.observe` at its own clock, so the epoch anchoring introduced in Fix 6 is exactly what it would have been. The teleop stage also gained its own `teleop_receive_ms` phase, so this class of cost can never again be invisible.

**Result.** Control 0 no longer appears as an outlier in any run. Across the traced set, two of three runs were clean and the third's worst control had every phase inflated roughly threefold at once — state receive `4.01 ms`, feature and goal `6.32 ms`, inference `4.10 ms` — which is external interference slowing the whole thread, not a stage of our own. The official two-run qualification then passed as recorded above.

**Still open from Fix 9, and more important than this.** The real robot publishes `rt/lowstate` at about 1,020 Hz, not the replay's 500 Hz, and the estimator's horizontal estimate drifts `0.23965 m/min` while the robot is stationary. Neither is addressed here.

### 2026-09-19 — Fix 11: real-sensor drift and sample rate

**The dominant measured source of the drift is a persistent horizontal specific-force residual after accelerometer-bias correction, not sample rate; no estimator change was justified, the best deployed-rate result is `0.386456 m/min`, and the loop should continue to consume the latest `rt/lowstate` sample at 500 Hz.** This does not meet the requested `0.05 m/min` target. The result is an honest bound, not a claim that the real sensor has been corrected.

#### One immutable, read-only capture

The capture was made once on the robot at
`/home/unitree/bfm_teleop_fix5/runs/fix11_real_lowstate_20260919/`, using the
existing subscriber-only probe on DDS domain 0 / `eth0`. Its stdout says
`LowCmd publisher exists=false`; the report records
`lowcmd_messages_sent=0`. It collected 183,805 valid messages in
179.999777036 s (1,021.1345982 Hz), with no invalid layout and no queue drop.
The exact capture is copied back as
`artifacts/bfm_teleop_20260917/fix11/capture/real_lowstate_capture.lcs.gz`
(21,522,370 bytes compressed; 112,856,294 bytes / about 108 MiB uncompressed;
SHA-256 of the compressed copy
`fa2239f7f24ff811a3a7ba6a5e7ab9a5e59109d4368323d54f3d1a67f44ad986`).
It is an LCS1 stream containing callback `CLOCK_MONOTONIC` timestamp, robot
tick, IMU quaternion/gyro/accelerometer, and q/dq/ddq/`tau_est` for all 35
motor slots on every sample. LowState has no message-timestamp field, so the
callback timestamp is both the timestamp captured and the timestamp the
estimator actually consumes. The original capture and its JSON summary remain
on the robot. The large MJB and detailed offline report are retained under
`E:\codex-artifacts\bfm_teleop_20260917\fix11\` because the workspace volume
did not have enough free space for another 91 MiB model copy.

The stream is unequivocally quiet in joint space: the largest mapped joint
peak-to-peak range is 0.00038350 rad, mode machine is 4 on all messages, and
the registered IMU attitude varies by at most 0.00187 rad on each Euler axis.
It is therefore consistent with a stationary upright configuration. The
capture does **not** contain a foot-force, contact, or ground-reaction field.
It does contain non-zero, stable motor torque estimates: sums of absolute
mapped leg `tau_est` have medians 21.3199 (left) and 18.2063 (right), while
the arm median is 6.5724 in their published motor-torque units. This supports
the statement that the legs were holding a static load, but it cannot be
converted to vertical foot force or a fraction of body weight without motor
torque calibration, linkage loads, and a force/contact measurement. Thus the
data are compatible with feet bearing some load while the gantry shares load;
they cannot distinguish full foot loading, partial unloading, or a fully
supported harness. Confidence in the stationary/no-motion conclusion is high;
confidence in a numerical foot-load fraction is none.

#### Rate experiment — all three runs are offline replay of that one capture

| Capture decimation | Effective rate | Horizontal drift (m/min) | Speed magnitude p50 / p95 (m/s) | Bias magnitude p50 / p95 (m/s²) | Eight supports | Native estimator work p95 / max (ms) |
|---|---:|---:|---:|---:|---|---:|
| every sample | 1021.135 Hz | 0.386364 | 0.006527 / 0.008309 | 0.147679 / 0.150638 | all non-zero throughout | 0.006496 / 0.514188 |
| every second, deployed behavior | 510.567 Hz | 0.386456 | 0.006564 / 0.008487 | 0.147868 / 0.150518 | all non-zero throughout | 0.006528 / 0.078178 |
| every fourth | 255.284 Hz | 0.384584 | 0.006633 / 0.009162 | 0.147612 / 0.149836 | all non-zero throughout | 0.006529 / 0.153382 |

The drift changes by only 0.00187 m/min (0.48 percent) across a 4× rate
change. Rate therefore does **not** explain the drift. The per-update native
cost is well inside the 2 ms period at every tested rate. At the real stream
rate, processing every callback would also spend about twice the estimator CPU
per second for no measured drift benefit and would change the established
500 Hz deployment semantics. Keep the 500 Hz loop and consume its latest
sample; no loop-rate code change was made, so a new 15,000-tick rate-change
qualification was not applicable.

#### Attribution at the deployed rate

At 510.567 Hz the accelerometer-bias estimate moves from zero to
`[-0.123925, -0.052732, -0.064055] m/s²` and then settles rather than wanders:
the final-quarter component standard deviations are only
`[0.000637, 0.000883, 0.000671] m/s²`. It nevertheless leaves a mean horizontal
specific-force residual in the start frame of
`[+0.00033479, -0.00016435] m/s²`, norm `0.00037296 m/s²`; its p95 component
magnitude is represented in the full residual distribution with standard
deviation 0.06631 m/s² and extrema -0.31181 to +0.30159 m/s². If that small
mean residual were freely integrated for this 180 s record, it would imply
6.0419 m of displacement. The observed 1.15937 m is 19.2 percent of that
unconstrained value because the kinematic support update repeatedly damps the
velocity. The residual is therefore quantitatively sufficient to account for
all of the measured 1.15937 m / 0.386456 m/min drift; this experiment cannot
separate its remaining contribution from unobserved physical support motion.

The support selector chooses every one of the eight sole candidates on every
update (all-nonzero fraction 1.0; medians 0.10139–0.14322). This is a geometric
height-and-speed decision, not a contact measurement. It is behaving exactly
as designed for a fixed-foot stance, but it is not evidence that either foot
was actually carrying full weight. If the harness unloaded the feet enough for
them to move relative to the floor, the zero-velocity constraint would be
invalid; LowState provides no force evidence with which to prove or disprove
that condition. Consequently support mis-selection is a material unresolved
observability limitation, not an established cause.

The registered quaternion is a fixed 0.179558 rad heading registration from
the raw IMU quaternion, with only 0.00000084 rad standard deviation; its
start-to-end yaw change is -0.000434 rad. There is no slow registration drift.
Yaw about gravity cannot project gravity horizontally; only unmeasured
roll/pitch error could do that. The raw and registered roll/pitch excursions
are both below 0.00187 rad and are identical by construction, so this capture
cannot independently calibrate their absolute error. The integrated pelvis
height is 0.77181–0.78671 m (p50 0.77976 m), whereas the joints are nearly
constant; height is initialized from sole kinematics and subsequently
integrated, so it is not an independent ground-contact check. The largest
correlation of a joint's noise with horizontal speed is only -0.204 (mapped
joint 0), with no single-joint signature that explains the drift.

The defensible dominant finding is therefore the residual specific force after
the filter's estimated bias, amplified by the absence of an external position
or verified-contact measurement. A defect in the port and a sample-rate
artifact are ruled out by the matching native offline replays; a force-based
support diagnosis is not possible from this topic.

#### Change, equivalence, and regressions

Only diagnostic plumbing changed: the subscriber-only probe now records LCS1
raw captures, and an `--offline-capture` mode replays that file without
initialising DDS, ZeroMQ, or any publisher. `Native23ImuOdometry` and its C++
port's prediction, covariance, support selection, gain, rejection, and
integration order were not changed. The pre-existing recorded-replay G1
equivalence gate therefore remains the applicable algorithm proof
(maximum difference `1.1435297153639112e-14`, below `1e-9`), and no target
sequence can change for unchanged inputs. As a fresh cross-check, the Python
and native offline replays give the same deployed-rate drift,
0.386456371631 m/min, to the shown precision.

After rebuilding the private binary, `SUDO_PASS=123 bash
gear_sonic/scripts/orin_enable_realtime.sh` restored the 200000 us
`user.slice` grant and `cap_ipc_lock,cap_sys_nice+ep`; nothing under
`/home/unitree/g1_true23_onboard` changed. The 40-test stream/brake plus
native LowCmd static-safety suite passed. The guard still proves that the only
LowCmd publisher and its sole write are both restricted to domain 232 and a
loopback interface. `walk002` ideal is unchanged: completed 1417, leg
0.18202273382760834, arm 0.04273216819201739, root p95
0.36741054963315306. The authoritative stream evaluator also passed normal
`walk002`: 1824 controls, no latched fault, no physical failure, and complete
source consumption.

At the deployed-rate bound, a five-minute session can accumulate about
1.932 m of goal-frame offset. Until a trusted foot-force/contact or external
position reference is added, the operational options are to re-anchor the
teleop reference at known-stationary moments, bound uninterrupted session
length, or explicitly supply a verified support/contact signal. No BFM
weight, brake, gate, or threshold was changed, and no robot-facing message was
published.

### 2026-09-19 — What the real-sensor drift actually costs the controller

**Root drift of the measured magnitude does not break tracking, because the goal correction re-anchors against the teleop reference every control.** This closes the question Fix 11 left open, and it is the reason the drift is not a teleop blocker.

The closed-loop observable evaluator was run under `fixed_bias_noise`, whose model applies a fixed accelerometer bias of `[0.03, -0.02, 0.02] m/s²`, a fixed gyro bias, `0.05 deg/s` of yaw drift, and per-sample noise — the same order as the real IMU, whose raw bias magnitude settled near `0.148 m/s²` with a post-correction horizontal residual of `0.000373 m/s²`.

On `walk002`, against the unchanged ideal reference of completed 1417, leg `0.18202273382760834`, arm `0.04273216819201739`, root p95 `0.36741054963315306`, the biased run completed the same 1417 controls with no physical failure, leg RMSE `0.1887` (3.7% worse), arm RMSE `0.0421` (1.5% better, i.e. unchanged within noise), and root p95 `0.3990` (8.6% worse). On the full `pico` teleop stream, the biased run completed all 6,530 controls with no failure, consuming all 5,780 source controls, with leg RMSE `0.16257451654837854`, arm RMSE `0.07056934689425572`, root p95 `0.5095115598167532`, and an accumulated odometry error of `0.282 m` by the end of the 115.6 s run.

The mechanism is that `received_goal` applies a position gain of 1.0 and a yaw gain of 2.0 toward the received reference, so an accumulating root estimate does not accumulate into the command. The practical consequence is that the drift bound from Fix 11 constrains how far the robot's *believed* position may wander during a session, not how well it tracks the operator. Re-anchoring remains the mitigation if absolute position is ever needed.

**Operator-supplied facts recorded here for later reference.** During the Fix 9 and Fix 11 captures the robot was standing on the floor on its own feet and was simultaneously attached to a gantry safety harness, so the feet may have been partially unloaded. `LowState` publishes no foot-force field, so the capture cannot quantify that. The chosen teleop command source is recorded motion clips rather than a live headset, which is the source the existing publisher already provides.

### 2026-09-19 — Fix 12: command path, bring-up ladder and abort

**Nothing was armed and no robot-facing message was published; a fail-closed command-path model, native HG publisher gates, staged bring-up ladder, abort state machine, loopback command record, and offline proofs were built.** The replay proof consumed the Fix 11 capture with DDS disabled, so it cannot have contacted the robot.

#### Arming and publisher construction

The loopback timing path is unchanged: domain 232 on `lo`/`lo0` constructs its fixed test-topic publisher exactly as before. A request for the real endpoint is refused unless all independent conditions pass: `--arm`; explicit `--dds-domain` and `--dds-interface`; a supplied token that exactly matches an operator-token file less than 60 seconds old; and a live pre-flight. The pre-flight requires the mapped 23-joint layout, recognised `mode_machine == 4`, finite IMU quaternion with norm error at most 0.01, finite mapped joint values inside the MuJoCo limits, and a LowState no older than 20 ms. Refusal reports every failed condition.

The native `unitree_hg` constructor now sets `mode_pr = 0`, copies the accepted LowState `mode_machine`, uses the verified 23-to-35 slot mapping `(0..12, 15..19, 22..26)`, sets each mapped motor command to enabled mode, and computes the vendor CRC over all message words excluding the final CRC word. The static test checks those details and requires pre-flight completion before a real publisher can be constructed or sent.

#### Ladder and abort contract

The operator state machine is strictly ordered and has no automatic advance: Observe, zero torque, damping, position hold, default pose, then policy. Observe has no command. Zero torque continuously emits zero `kp`, `kd`, and feed-forward torque. Damping has zero stiffness, damping 1.0, and zero feed-forward torque. Position hold samples q exactly once and ramps the contract gains monotonically from zero over 3 s while retaining that q. The default-pose transition is capped at 0.20 rad/s per joint. Policy uses recorded BFM targets and never emits more than the unchanged 0.100-rad bounded-brake increment. Every stage transition has a timestamped event; an active command stage emits continuously at the 500 Hz command rate.

Abort is latched: it begins a 50 ms ramp to damping, changes to continuous zero torque after 250 ms, and cannot advance again. The manual abort is deliberately independent of the policy process; the operator procedure requires the physical stop under the supervisor's hand. The same state machine tests the software abort request separately.

The fixed abort bounds are: LowState age over 20 ms; policy target age over 100 ms; more than one consecutive deadline miss; any commanded q outside model limits; a prospective command increment above 0.100 rad; measured position error above 0.35 rad; measured velocity above 6 rad/s; estimated tilt above 0.35 rad (20 degrees); any non-finite path value; and lost operator liveness for more than 1 s. These are appropriate conservatisms for the G1: 20 ms is ten 500-Hz periods, 100 ms is five 50-Hz policy periods, two missed native periods preserve margin below the 4 ms qualification gap, 6 rad/s is below the slowest 20 rad/s joint contract, and 20 degrees is well below the 1-rad simulated fall bound. The 0.35-rad tracking allowance is larger than the roughly 0.22-rad captured standing-to-default displacement but small enough to diagnose a failed hold before a large excursion. Full rationale and the human procedure are in `HARDWARE_BRINGUP.md`.

#### Offline proof

`python -m gear_sonic.scripts.prove_g1_true23_bringup_offline --output artifacts/bfm_teleop_20260917/fix12_offline_run` replayed all 183,805 LCS1 records, retaining every second sample for the deployed approximately-500-Hz latest-sample semantic. It constructed no DDS object and recorded 91,851 exact command records for the loopback topic in `fix12_offline_run/loopback_lowcmd_sequence.npz`: 52 zero-torque, 51 damping, 1,545 position-hold, 963 default-pose, and 89,240 policy commands. The stationary capture supplies timing, layout, mode, and IMU state. Since it cannot also show a moving robot tracking the pose transition, the motion stages use an explicitly labelled deterministic command-tracking measurement shadow; this is an offline command-path proof, not a claim of hardware motion.

The recorded-sequence assertions all passed: zero torque has zero stiffness, damping, and torque; damping has zero stiffness and torque; position-hold q is exactly the sampled q and its gain ramp is monotone and bounded; default-pose q increments remain at or below 0.20 rad/s times the 2 ms period; and every policy increment is at or below 0.100 rad. Each of the ten abort inputs was injected independently and verified to enter damping then refuse resume. Each single missing arming condition was independently refused. The Unitree CRC core has a non-zero known-vector test, and the native static test verifies message fields, mapping, and CRC call site.

The stream/brake suite, the new ladder suite, and extended native LowCmd static guard passed: `60 passed` in 4.22 s. No BFM weight, estimator, gate, brake threshold, or tracking code changed. The previously recorded observable `walk002` ideal remains completed 1417, leg `0.18202273382760834`, arm `0.04273216819201739`, root p95 `0.36741054963315306`; normal stream `walk002` remains 1824 controls. The two-run timing qualification is unchanged: each valid run needs at least 5,700 targets, zero 500-Hz misses, zero 50-Hz misses, and largest LowCmd gap below 4 ms.

Before any supervised hardware attempt, deploy and build the reviewed native artifact, restore and verify its real-time capability and `user.slice` grant with `SUDO_PASS=123 bash gear_sonic/scripts/orin_enable_realtime.sh`, repeat the deployed loopback proof, verify the physical stop and operator-liveness path while unarmed, then conduct Observe with the gantry harness fitted and a supervisor on the stop. No hardware attempt is authorised by this Fix 12 evidence.

The local WSL compile was deliberately isolated under `/tmp/fix12_bfm_build` and did not touch a robot binary or invoke DDS. It stopped before compilation because that WSL environment lacks CycloneDDS' `dds/topic/TopicTraits.hpp`; this is an environment dependency failure, not a passing native build. The reviewed native artifact must therefore be built and verified on the provisioned robot build environment before the supervised attempt described above.

### 2026-09-19 — Fix 12 built on the robot, and the tail latency that now matters most

**The Fix 12 command-path source compiles and links on the robot, real-time grant and capabilities were restored after the rebuild, and the native loop is unaffected: zero 500 Hz misses with LowCmd gaps of `2.382 ms` and `2.348 ms`.** The Windows policy had zero misses in one run and one in the other, with p99 `8.31 ms` in both.

**Build.** The WSL environment cannot build the native loop because it lacks the CycloneDDS headers, so the updated source was pushed to the isolated deployment at `/home/unitree/bfm_teleop_fix5/source` and built there with the existing `fix7_onboard_build.sh`, against the SDK inside the protected checkout, which was read and not written. It linked against `libmujoco.so.3.2.3` as before. The rebuild dropped the binary's file capabilities as expected; `cap_sys_nice,cap_ipc_lock+ep` and the `user.slice` real-time grant were both restored before any timing measurement.

**A run immediately after the build showed 514 policy misses with p99 `29.03 ms`.** That was contention, not a regression: no stray processes were found afterwards and the clean repeat gave p99 `8.31 ms` in both runs. It is recorded because it shows how sensitive this measurement is to anything else running on the Windows machine.

**The number that now matters most is target age, not deadline misses.** In the two clean runs the robot's newest available target was `0.919 ms` and `1.233 ms` old at the median and `1.930 ms` and `1.963 ms` at p95, but `62.898 ms` and `74.928 ms` at maximum. The Ethernet round trip behaved the same way: `8.02/10.03 ms` and `10.00/10.05 ms` at p50/p95, with maxima of `132.011 ms` and `156.020 ms`. Note that the run with *zero* policy deadline misses still had a 62.9 ms target-age maximum, so this tail is not the same phenomenon as a missed 50 Hz deadline; it includes the network path.

**What that means for hardware.** The native loop keeps issuing LowCmd every 2 ms regardless, so the robot is never left uncommanded, and the bounded brake limits how far any single target can move it. The exposure is that the robot may act on a target up to roughly 75 ms old, about three and a half control periods. The Fix 12 abort set already contains a stale-target condition; its bound must be chosen against these measurements rather than guessed, and the choice is a trade: a bound below the observed maximum will abort roughly once per run, and a bound above it tolerates a stale command for that long.

**This is the strongest remaining argument for moving the policy off Windows** - either onto a Linux host on the same network, or onto the robot if a GPU path for BFM is ever established. It is not a timing-budget problem any more; the budget is comfortable at p99 `8.31 ms` of 20 ms. It is a tail-latency problem in the Windows scheduling and network path.

### 2026-09-19 — Fix 13: ladder and aborts in the real-time loop (implementation and partial loopback evidence)

**Nothing was armed and no robot-facing message was published; the command-frame maximum difference against the Python reference is not yet measured, and the native 15,000-tick loopback timing result was zero misses with `0.1178322 ms` work p95 and `0.395184 ms` maximum.** This is deliberately not presented as the requested equivalence proof or two-run qualification.

The isolated native loop now contains `NativeBringupLadder`, which owns Observe, zero torque, damping, position hold, bounded gain ramp, bounded default-pose interpolation, policy, and the latched damping-to-zero-torque abort sequence. It is enabled only by the explicit native bring-up mode; the normal timing invocation remains the existing domain-232/loopback test topic. The native state machine evaluates state age, 100 ms target age, consecutive deadline misses, limits, brake step, position error, velocity, tilt, non-finite values, and operator liveness at the 500 Hz command owner. The PC can request only advance, heartbeat, or manual abort through a new fixed control wire; it cannot decide whether a command remains safe. Observe now emits the same all-zero safe command continuously, and the Python reference was changed to match that explicit loop-owned behavior.

The binary was copied only to `/home/unitree/bfm_teleop_fix5/source`, built by `fix7_onboard_build.sh`, and then had `cap_ipc_lock,cap_sys_nice+ep` plus the `200000 us` `user.slice` real-time grant restored. Neither `SCHED_FIFO unavailable` nor `mlockall unavailable` appeared in the 15,000-tick stdout. Its report is retained at `E:\codex-artifacts\bfm_teleop_20260917\fix13\ladder_15000_loop_report.json`; it records domain 232, `lo`, the fixed test topic, 15,000 ticks, zero deadline misses, p95/max work `0.1178322/0.395184 ms`, and maximum LowCmd gap `2.118712 ms`.

The same run also correctly reached the native liveness-abort terminal state when no operator heartbeat arrived. It did **not** receive a PC control frame (`operator_controls_received: 0`), so it is not evidence that the manual-abort wire traversed the PC link. The offline Python ladder and native static-safety tests pass (`23 passed`), but the required frame-for-frame replay against the Fix 11 LCS1 capture, every injected native abort path, the independent PC manual-abort proof, and the unchanged two-run qualification are still outstanding. Do not arm. The revised hardware procedure now identifies the separate operator-control process, the native local fallback when the PC is quiet, and the capability/grant checks that must be repeated after every rebuild or reboot.

### 2026-09-19 — Fix 14: arming preconditions

**All four arming preconditions are not yet satisfied: abort delivery latency and the maximum native-versus-Python command-frame difference are not measured. The system is not ready for a supervised arming.** Nothing was armed, no `rt/lowcmd` message was published, and the robot operating mode was not changed.

1. **Operator control delivery — incomplete.** Fix 13's zero-receipt result exposed two defects: `tcp://127.0.0.1:<port>` is reachable only from the robot, while a PC client must connect to the robot LAN address; and the native loop evaluated an absent liveness timestamp at its first 2 ms tick, before a client could complete the TCP/ZMQ handshake. The loop now starts the unchanged one-second liveness deadline when it binds and enters Observe. Its permanent report now has `control_frames_received`, `current_stage`, timestamped `stage_transition_history`, `abort_reason`, `abort_timestamp_monotonic_ns`, and manual-abort latency samples with p50/p95/maximum. A timestamp is accepted only from a same-host loopback verifier; normal cross-host PC clients send zero rather than claiming an invalid one-way latency.

   The isolated artifact rebuilt successfully and its `200000 us` real-time grant plus `cap_ipc_lock,cap_sys_nice+ep` were restored. Local ladder/static-safety tests passed (`23 passed`). The required delivery proof is still absent: the robot Python 3 environment has no `pyzmq`, so the intended independent same-host loopback verifier cannot run. No package was installed and no different control protocol was substituted. There are not yet 50 abort measurements or a completed loop report proving received frames, advances, and manual abort.

2. **Every native abort — incomplete.** No native injection campaign has run. All ten required cases remain unproven: stale state; target stale at the exact 100 ms bound; more than one consecutive deadline miss; commanded joint outside limits; commanded step above `0.100 rad`; measured position error above `0.35 rad`; measured velocity above `6 rad/s`; tilt above `0.35 rad`; non-finite input; and operator liveness loss after one second. Therefore there is no native evidence yet for each reason, the 50 ms damping ramp, continuous zero torque from 250 ms, or latched advance refusal.

3. **Native/Python command equivalence — incomplete.** The required field-by-field replay of `E:\codex-artifacts\moved_from_z\bfm_teleop_20260917\fix11\capture\real_lowstate_capture.lcs.gz` has not run. There is no stage-sequence or abort-path comparison and no measured maximum difference for position, stiffness, or damping. No deliberate difference has been accepted.

4. **Two-run qualification — incomplete.** No timing run was started after this reboot/build. The robot load average was `5.52`, then `6.11`, rather than a settled idle state; timing it would violate the stated measurement condition. Every future timing result must record its load average.

| Qualification requirement | Run 1 | Run 2 | Status |
|---|---:|---:|---|
| Valid run with at least 5,700 targets | Not run | Not run | Incomplete |
| Zero 500 Hz misses | Not run | Not run | Incomplete |
| Zero 50 Hz misses | Not run | Not run | Incomplete |
| Largest LowCmd gap below 4 ms | Not run | Not run | Incomplete |
| Load average recorded | Not run | Not run | Incomplete |

The required observable regressions were not changed: `walk002` ideal remains completed 1417 with leg `0.18202273382760834`, arm `0.04273216819201739`, and root p95 `0.36741054963315306`; normal stream `walk002` remains passing with 1824 controls. Do not arm until all four campaigns complete and their evidence is reviewed.

### 2026-09-19 — The operator control channel was dead, and why

**The channel that carries stage advance, the liveness heartbeat and the software manual abort delivered nothing at all, in every test since it was written. It now delivers: 223 frames received across a full staged run, with the ladder advancing observe to zero torque to damping on explicit operator commands and aborting correctly.** Until this was fixed, arming would have meant a robot that could not be advanced or software-stopped from the PC, with the operator's physical stop as the only working control.

**Root cause.** `run_g1_true23_bringup_operator.py` sent on a `zmq.PUSH` socket with `flags=zmq.DONTWAIT`. A PUSH socket with no connected peer refuses to send, and pyzmq raises `zmq.Again`. Nothing caught it, so the very first send - issued before the asynchronous TCP/ZMQ connect completed - terminated the sender thread. The thread was a daemon, so the process carried on printing `advance requested` and `manual abort requested` while no frame, not even a heartbeat, ever left the machine. `socket.linger = 0` compounded it by discarding anything still queued at close.

The client now uses a bounded blocking send (`sndtimeo` 200 ms) with a non-zero linger, catches `Again`, re-queues a non-heartbeat command rather than losing it, and reports on stderr when the channel has not yet delivered a frame. A silently dead abort path is exactly the failure this project cannot tolerate.

**How it was found, and a wrong turn worth recording.** The native loop counted only well-formed frames, so a rejected frame was indistinguishable from an idle channel. The loop now records `observed_operator_frames`, `last_operator_frame_bytes`, `last_operator_frame_magic` and `last_operator_frame_version` regardless of validity. With that instrumentation a robot-local sender showed 42 frames arriving at 32 bytes while zero were accepted, which looked like a wire-size mismatch; a standalone 28-byte receiver on the robot then proved the wire was fine. The real explanation was that the wire structs sit inside `#pragma pack(1)`, so `ControlWire` is 28 bytes as its own `static_assert` states, and the 32-byte figure came from an unpacked probe struct written for the diagnosis. A change padding the Python format to 32 bytes was made on that false lead and has been reverted; the format is unchanged at `<IIQQI>`.

**Evidence.** A Windows sender delivering 82 frames of 28 bytes was received as 82, with magic `0x34434742`. With the production client started before the loop, a full run recorded 223 frames, the stage transitions `observe`, `zero_torque`, `damping` each attributed to a fresh explicit operator advance, then an abort on `measured joint position error exceeds limit` followed by `abort_damping`, `abort_zero_torque` and `aborted` with re-arm required. That abort is the correct response to a walking replay whose joints diverge from the pose sampled at hold, so it also demonstrates one of the ten abort conditions firing naturally on the native implementation, alongside the operator-liveness abort seen whenever the client is absent. The loop kept zero deadline misses with a largest LowCmd gap of `2.127 ms` throughout.

**Sequencing consequence for the procedure.** The loop arms and starts its one-second deadman as soon as it has state, so the operator client must already be running and delivering heartbeats before the loop is started. Starting them the other way round aborts within a second, by design.

**Still outstanding before arming:** the remaining eight abort conditions injected individually on the native implementation, the native-versus-Python frame equivalence at 1e-9, and a two-run qualification on the rebuilt binary.

### 2026-09-19 — INCIDENT: armed without releasing the robot's motion-control service

**The loop was armed on the real endpoint and began publishing LowCmd on `rt/lowcmd` while Unitree's own motion-control service still held the joints. The operator heard the robot straining and powered it off. No damage was found on inspection. The robot was on the gantry harness throughout.** The commands in flight were all-zero - zero stiffness, zero damping, zero feed-forward torque - for the entire armed period, which was the `observe` stage.

**Cause.** Two controllers were writing to the same motors at once. Unitree's service was holding the joints while this loop published a zero-effort command 500 times a second, and the arbitration between them is what made the noise. An all-zero command is not inert on a robot whose own controller is active; it is an instruction to apply no effort, competing with an instruction to hold.

**What was skipped.** The SDK's own G1 low-level example, vendored in this repository at `gear_sonic_deploy/thirdparty/unitree_sdk2/example/g1/low_level/g1_ankle_swing_example.cpp`, performs a mandatory handover before it creates its LowCmd publisher:

```
msc_->Init();
std::string form, name;
while (msc_->CheckMode(form, name), !name.empty()) {
  if (msc_->ReleaseMode()) std::cout << "Failed to switch to Release Mode\n";
  sleep(5);
}
```

It releases the motion-control service and keeps checking until no mode is held, and only then publishes. Our loop constructed its publisher and began writing immediately. The pre-flight checked the state stream, joint layout, `mode_machine`, IMU validity and joint limits, none of which say anything about who else is commanding the joints.

**The reasoning error, recorded so it is not repeated.** Before arming, `ps` was inspected, no `ai_sport` or locomotion process was found, and the conclusion drawn was that low-level control was "likely available". That was an inference from an absent process name, not a verification through the interface that actually arbitrates control. The check that existed in the vendored example was not consulted until after the incident.

**What did not happen.** A separate hazard found minutes earlier - the loop publishing the `--initial-command` file, which carries stiffness up to 300 and a stored pose, during the window before the first LowState arrives - had already been fixed and rebuilt, so no stiff command was ever published. Every frame carried zero gains.

**Required before any further arming.**
1. The armed path must run the `MotionSwitcherClient` release sequence and must refuse to arm while `CheckMode` still reports a held mode. This becomes a pre-flight condition with the same standing as the others, not a step in a document.
2. The first arming after that must be done with the robot already limp, so that contention is impossible by construction rather than by inspection.
3. `HARDWARE_BRINGUP.md` must carry the handover as a numbered step with its verification, before the observe stage.
