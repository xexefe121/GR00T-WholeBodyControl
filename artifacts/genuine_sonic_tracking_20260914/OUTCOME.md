# Genuine SONIC tracking: first localized result

The braking review's recommendation stands: preserve strict rejection handling,
leave sustained braking disabled, and do not promote either experimental filter.
This run used the existing paired breadth25 SONIC encoder/decoder and existing
native23 live-packet controller. No braking filter, training, reward change,
new controller, or robot connection was added.

## What ran

Two diagnostic replays of the existing walk002 segment completed all 656 controls
(13.12 seconds of physics) without fallback. The second adds a local knee-command
probe; both nominal traces are byte-identical. This is neither a new full source
plus standing qualification nor a live/timing test.

The diagnostic records received and admitted references, original29 hand/head
goals, encoder267, history930, decoder994, raw23, transformed targets and physical
state. Original source goals are aligned by an exact match of every retained
joint across all 667 source frames; the lifecycle offset is 361.

## First divergence

**Before inference:** the saved received packets already differ from the original
source hand/head goals. Pelvis-local position differences at the first packet are
8.11 cm left hand, 9.49 cm right hand and 2.19 cm head. Over the replay their p95
differences are 12.63, 14.32 and 7.36 cm. Retargeting changes these saved packet
values by at most 7.78e-16: the loss predates this runtime retarget call.
These are reference representation differences, not measured physical tracking
errors. This confirms an earlier known loss; it is not a newly discovered fix.

The retained joint mapping and all past lower-body position samples match the
source motion. That check does not certify every training observation convention.

**First sustained leg tracking error:** control index 5, source control frame 15,
post-control time 0.12 s. Localization uses leg RMSE above 0.15 rad for five
consecutive 20 ms samples, without replacing the original acceptance criteria.

| Joint | Reference next pose | Measured before | SONIC target | Measured after |
|---|---:|---:|---:|---:|
| Left knee | -0.087267 | 0.200855 | 0.324068 | 0.233649 |
| Right knee | -0.087267 | 0.214958 | 0.318036 | 0.244623 |

All values are radians. Neither knee saturates torque during these ten physics
steps. The actor/wrapper requests further flexion while the reference asks for
extension; physics moves in the commanded direction. This interval is not
explained by knee torque saturation. Contact coupling and the actor's balancing
objectives can still matter; target-position error alone is not a proof that a
dynamic control action is wrong.

## Controlled local tests

All probes copy complete MjData from the same pre-command boundary, preserve
solver memory and apply ten 2 ms physics steps. The nominal fork reproduces both
position and velocity exactly. They are causal one-command probes, not proposed
controllers or recovery demonstrations.

| Command/input at that boundary | Post-control leg RMSE |
|---|---:|
| Nominal SONIC | 0.178325 rad |
| Original hand/head goals, same policy and robot history | 0.175804 rad |
| Nominal targets except knees hold their measured pre-command angles | 0.172721 rad |

Restoring original goals changes actor output but does not remove the immediate
leg divergence. Removing the extra requested knee flexion reduces it slightly.
The full reference-target branch is unavailable: the reference knee angle
(-0.087267) is below the existing safe transform's target lower bound
(+0.07308635). No reference projection or limit relaxation was performed. This
target restriction does not itself prove that tracking within the permitted
error tolerance is impossible; PD targets and physical joint positions differ.

## Limit on the conclusion

Walk002 is not an already-qualified feasible reference. Its import says so; the
saved conditional support screen found solutions within effort limits for only
6 of 667 frames, under that screen's stated contact/derivative assumptions. This
does not prove infeasibility under every possible motion realization.

The successful prepared walk003 expert used waist Kp 300, whereas this SONIC
runtime uses 40.17923847367. Its success cannot establish feasibility under an
identical controller execution contract without a matched-plant comparison.

Therefore this run does **not** establish that SONIC needs more training. The
next experiment must bind the successful walk003 reference, original task goals,
prepared plant and actuator contract explicitly, then compare the existing SONIC
path against that feasibility witness. A change required to match the plant must
be identified as an experimental runtime difference, not silently called parity.
Do not start another fit, filter sweep or architecture on the strength of this
walk002 result.

## Verification and reproducibility

- Nominal one-command continuation: exact qpos and qvel from copied MjData.
- Two fresh nominal traces: identical compressed trace SHA256
  `5f8f124f5ec59517fa4fdb4e5c47ca32153d74c515611f0dfd1f3f8bdbe32f45`.
- Historical saved baseline: not bit-exact. First position difference is
  5.55e-17 at control 1; maximum stays below 1e-12 through control 157, then grows.
  Whole-run maximum position/velocity differences are 0.068917 and 3.992069.
  This run does not renew historical exact-parity claims; the local diagnosis at
  control 5 differs from that archive by at most 1.67e-16 in qpos.
- MuJoCo 3.2.3, NumPy 1.26.4, ONNX Runtime 1.23.2. Full report retains model hash,
  pair hashes, gains, effort limits and raw counterfactual values.
- Restored missing balance ONNX from the other existing checkout only after
  matching its pinned SHA256. It did not activate in this replay.
- Restored native23 XML's pinned LF bytes after proving the mismatch was solely
  CRLF conversion. Preserved original bytes under the output directory and added
  a path-specific LF rule to `.gitattributes`; model semantics did not change.

Script: `gear_sonic/scripts/diagnose_g1_true23_sonic_tracking_boundary.py`

Latest report and trace:
`E:/codex-artifacts/genuine_sonic_tracking_diagnosis_20260914/walk002_v2/`

Existing WSL runtime launcher:

```bash
/mnt/e/codex_sonic_runtime/mjbatch323_20260910/venv/bin/python \
  /mnt/e/codex-artifacts/genuine_sonic_tracking_diagnosis_20260914/run_existing_runtime.py \
  --repository-root /mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof \
  --recording-directory /mnt/c/Users/camer/sonic23_sim_artifacts/internet_pico_20260909_v1/walk002 \
  --output-directory /mnt/e/codex-artifacts/genuine_sonic_tracking_diagnosis_20260914/new_run
```

Use a new output directory; existing evidence is never overwritten. The runtime
bootstrap imports the existing pinned simulator before adding the existing
ONNX dependency directory. No new dependency environment was installed.
