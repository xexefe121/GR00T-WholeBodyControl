Exact native clipped-PD target canonicalization succeeded, with limited reduction.

For each joint and 20 ms control, every saved 2 ms torque is first recomputed
from the original target. If any substep is unsaturated, the original target is
preserved. If all ten substeps saturate, positive saturation supplies a target
lower bound `q + (effort + kd*dq)/kp`; negative saturation supplies an upper
bound `q + (-effort + kd*dq)/kp`. Their intersection with native joint target
bounds admits equivalent targets on the saved trajectory. The target closest
to the previous canonical target is selected, then all ten torques are
recomputed. Any floating-point difference reverts that joint component to the
original target. No tolerances replace exact torque equality.

The initial previous target is the already received v4 joint-reference sample:
frame 10 for the complete walk003 lifecycle, frame 3750 for the standalone
restoration segment. This makes the sequential target-choice rule explicit;
neither initial target is written into physical state.

Walk003 source results:

- 45 of 18,837 target components changed, across 42 of 819 source controls.
- 51 components were fully saturated; six boundary candidates reverted.
- Target step RMS decreased 0.356668 to 0.354910 rad, about 0.49%.
- Maximum target step decreased 4.35266 to 3.95322 rad.
- Maximum normalized action stayed 6.65252; components outside ±5 fell 31 to 30.

One complete native MuJoCo 3.2.3 replay of all 1,569 lifecycle controls followed.
Every qpos, qvel, and applied torque matched the original at all 15,690 physics
steps exactly. Engine warning counts and lastinfo stayed zero. Cumulative clock
error relative to ideal multiplication was 6.51e-12 seconds. No physical state
writes occurred after the original frame-10 initialization.

Restoration65 results:

- 10 of 1,495 components changed; one of 11 eligible candidates reverted.
- Target step RMS decreased 0.531854 to 0.524588 rad, about 1.37%.
- Maximum normalized action stayed 8.47993; components outside ±5 fell 139 to 138.
- All 650 saved-state torque calculations matched exactly. No additional
  restoration physical replay was needed or claimed.

Original and canonical targets are separate arrays in each `targets.npz`.
The walk003 native replay is saved separately. Existing controller, model,
gains, caps, and historical traces were not changed.

This result preserves a recorded physical trajectory. Canonical targets change
the normalized previous-action history consumed by a policy, so online policy
equivalence is not claimed. The small reduction also does not establish that
label range alone explains or fixes student closed-loop failures.
