# Direct absolute-target feature feasibility

The saved-data check found **zero exact reduced-input duplicates and zero absolute-target conflicts** across all 146,733 required rows. Adding the already qualified PICO and walk002 moving rows gives 153,580 distinct reduced inputs, again with zero conflicts. This supports investigating the proposed input/output contract; it does not establish a stable policy or sufficient state.

| Saved population | Rows | Absolute targets outside native bounds |
|---|---:|---:|
| Three walk003 nominal branches | 3,057 | 0 |
| Existing velocity probes | 140,622 | 0 |
| Physical one-control branches | 3,054 | 0 |
| PICO moving nominal | 5,980 | 0 |
| walk002 moving nominal | 867 | 0 |

Every source float32 input and float64 **absolute** target is preserved. Residual labels are never used as the target. Group keys retain original bytes and signed zeros. A separately labeled numerical-zero equivalence check gives the same result; no retained feature or absolute target in these populations contains negative zero. SHA256 matches require actual retained-byte equality. Eight synthetic tests cover conflicts, removed and retained columns, signed zeros, dtype rejection, nonfinite inputs and forced hash collisions. All 31 initial scan source/input pins matched again at completion. The final completion receipt additionally pins both physical source-frame metadata arrays, this note and its source.

The saved hash/group index identifies every row without writing a new feature or label dataset. No rows were removed, averaged or reweighted. Existing broader-label wrapper exit remains historically unknown; its completed producer outputs and independent saved-array/BFM authenticity audits provide the qualified-data evidence.

## The 1,000 retained features

| Original 1,069-vector slice | Kept meaning |
|---|---|
| 0:23 | Current native joint position minus default |
| 23:46 | Current native joint velocity |
| 46:49 | Native root angular velocity slice |
| 49:52 | Gravity projected into measured body coordinates |
| 75:78 | Root linear velocity in measured-heading coordinates |
| 78:79 | Root height |
| 79:1023 | Eight 118-value received-goal blocks at offsets 0, 1, 2, 4, 8, 16, 24 and 37 |

Removed 52:75 is previous target minus default. Removed 1023:1046 is unclipped BFM base minus default; removed 1046:1069 is raw previous action. The first 52 retained values follow **GoalFeatures ordering**, which differs from the BFM state52 ordering and scaling.

Source inspection of the pinned `student_linear_runtime.py` and `g1_true23_mpc_student.py` shows that retained features require measured qpos/qvel, contract/default values, prepared native/original29 goal data and rotation math. They need zero BFM actor or backward calls and no BFM history. Previous-target arithmetic is confined to the removed slice. A finite dummy previous target would leave all retained feature bits unchanged in the existing builder. The existing runtime still invokes BFM before that builder; eliminating those calls requires a separately selected feature/runtime change. None was made here.

The received-goal contract still requires the 38-sample span and preceding prepared task sample used for velocity/angular-velocity differences, with the existing EOF clamp. This is no new raw-pose processing or causal streaming qualification. The input carries no explicit clip ID, frame ID, absolute clock, planner state or planner gains.

## Output representability and limits

All saved absolute targets lie inside the same native 23-joint box. A 23-value affine position output has no residual-base reachability restriction; a fixed per-joint midpoint/half-span mapping with final native clamping can cover the closed box. This is a possible parameterization, not an implemented architecture. The BFM actor's ±5 action mapping must not silently be inherited. A finite tanh output without an endpoint convention cannot exactly reach closed-box endpoints; many teacher components are exactly at native bounds.

Float32 output cannot reproduce these float64 targets bit for bit: the largest observed cast-and-return error is 1.1920500542217383e-7 rad. Casting PICO targets alone makes 12 components slightly exceed their exact float64 native bound, so a future direct-output adapter would still need an explicit final clamp against the unchanged float64 limits. Per-joint minima/maxima, at-bound counts and cast statistics are retained in `results/report.json`.

Zero exact conflicts is a limited finite-data result: almost identical inputs with different targets, hidden committed-plan state, and multi-step departure/recovery remain untested. Velocity labels describe the existing clipped committed feedback map; physical labels cover one executed command followed by that same-clock map. This check does not attribute the prior control251 error to action history and does not authorize fitting, normalization changes or a controller trial.

All work was pure saved-array/hash processing: **zero model calls, native steps, optimizer updates, new labels or downloads**. No frozen controller or training files changed.
