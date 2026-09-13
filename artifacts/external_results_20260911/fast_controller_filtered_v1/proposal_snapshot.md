# Fixed private-preview controller experiment

Prepared 2026-09-11 05:59 UTC. Conditional on the separate preallocated-preview parity result. No connected rollout authorized by this document alone.

The ordinary final 65,000 head fails from a correctly reproduced BFM entry state. Its command error grows through measured joint velocity before clipping; afterward, storing preclip student actions also differs from the applied-target convention in the expert moving-phase labels. The existing planner already checks an imminent target privately before applying it. The proposed next controller keeps the current learned weights and makes command admission and action history explicit.

## Proposed fixed behavior

Keep canonical initialization, native model, original BFM and residual references, all 1,569 lifecycle controls, the original 819 source controls and the separate continuous 250-control hold. Preserve original BFM standing controls 0–249 exactly. At the first learned call, verify the same actual state, complete history, base output and 1,069 features as the qualified query250 example before considering any candidate.

During acquisition, source, return and terminal standing, consider these four already bounded position targets in this exact order:

1. The ordinary frozen controller proposal: student plus BFM during controls 250–1268; original BFM yaw4 during terminal standing.
2. The original BFM proposal with the mode's existing goal convention, clipped to native target bounds.
3. The preceding actual applied position target.
4. The current measured joint pose clipped to native target bounds, producing native PD damping.

For each candidate, predict exactly 100 ms of holding that target using a reusable private native MuJoCo state. Check every 2 ms against the unchanged strict position, velocity, effort, fall, warning and repeated-clock rules. Select the first candidate whose whole private segment passes. Repeated candidates may be checked once only if exact numerical equality and the skipped identity are recorded. Do not use future actual plant states, recorded expert commands, control-index-dependent weights or candidate-cost selection.

Apply the selected target for exactly 20 ms, then obtain the next actual state and repeat admission at 50 Hz. The 100 ms horizon is private prediction only; it does not become a five-control open-loop commitment. Generate the policy proposal once per actual control, staging its history/prior updates without committing them while candidates are tested. Commit the measured history, chosen action and control count once after admission. Rejecting every candidate must leave the actual plant and precontrol controller history/prior/count unchanged.

If no candidate passes, retain the complete partial trace and declare the lifecycle failed/incomplete before applying a rejected command. This is an offline simulation gate; it is not an implemented physical emergency stop. A passing 100 ms prediction is not an invariant safety set or proof of recursive feasibility.

For learned controls 250–1268, store the normalized actual applied target as the next action input, matching expert moving-phase labels. This applies even when the selected primary proposal was clipped. Preserve all actual measured history. Initial BFM standing retains its original raw action convention. In terminal BFM mode, retain raw BFM action when its proposal is applied unchanged; if a fallback replaces it, record the action corresponding to the applied target. Record both proposed and applied actions explicitly.

## Evidence required before one full trial

The private preview must first reproduce all 232 predetermined saved-state forecasts, including each native state, command torque, actual force, clock, warning count/last-info and first failure, against the independent oracle. Every caller state must remain unchanged. Its timing must include restoration, predicates and output recording; timing under concurrent work is diagnostic.

Freeze source, model, references, head, candidate order, 100 ms horizon and action-history rules. Review the adapter and actual launch command. One fresh canonical lifecycle and, if it completes, one continuous hold are allowed after explicit root selection. Do not pick among multiple horizon, blending or checkpoint trials.

Record every considered candidate's verdict, failure witness, selected action, solver time and full control-loop time. The initial 250-control BFM prefix and query250 input must match exactly. Separately assess complete source intent, both quiet windows, every native step and timing; a source-tracking failure remains a failure even if all physical limits pass.

This candidate combines an explicit command-admission rule with a correction to moving-phase action-history semantics. It is a new controller experiment, not an isolated claim that either change alone fixes the original failure. No new learner fitting or expert query is proposed here.
