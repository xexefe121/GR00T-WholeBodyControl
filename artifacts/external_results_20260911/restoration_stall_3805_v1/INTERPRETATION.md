# Control 3805 restoration stall

The identical fixed-merit solve generated 63 alpha candidates. All 63 failed the every-2ms native feasibility predicate, so none reached the independent actual-state oracle. No feasible candidate was discarded solely because its merit was worse.

The original shifted seed crosses the right ankle roll upper bound at horizon control 28, substep 5 (570 ms): q = 0.2618427054683092 rad versus upper bound 0.2618, excess 0.0000427054683092 rad. The initial-state merit is zero. All nonzero state merit is at knots 20–30, so an unchangeable initial-state penalty did not cause the stall.

All dynamics and cost derivatives are finite. Every backward sweep succeeds, at regularization 1, 10, 100, 1000, 10000, 100000 and 1000000. All corresponding line searches reject. The solver stops when regularization rises to 10000000, exceeding its 1000000 ceiling; the relative cost-drop tolerance is never the stop cause. Initial and final merit remain 4.292815755335639 and returned targets equal the initial seed exactly.

Dynamics derivatives are large: maximum |A| = 89471.3084 and |B| = 85961.6323. The final returned K has maximum absolute entry 5120.6349. Even the smallest alpha (1/256) causes target changes exceeding 4 radians in six of seven sweeps. The closest generated merit is 4.3340764909 at regularization 1000 and alpha 1/256; it still crosses the right ankle roll upper bound at 572 ms, excess 0.0006932133 rad. This supports investigating feedback amplification during the private restoration line search; it does not establish that a feedforward-only candidate will succeed.

All generated target sequences are retained in all_generated.npz. No physical control was executed. No source trajectory was skipped or altered.

Reporting correction: the original report's first_failure joint_index/name/q/direction fields were generated for all failure types. Those additional fields are meaningful only when reasons includes joint_range; for speed-only failures, disregard these extra range-specific fields. The recorded reasons, scalar strict metrics, failure timing and feasibility result remain valid. The diagnostic source is corrected for future runs; the original report and frozen executed source remain preserved.
