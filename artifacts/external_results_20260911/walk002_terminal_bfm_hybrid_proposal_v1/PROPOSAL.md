# One walk002 terminal BFM hybrid trial — preparation only

Original exact-controller walk002 completed all1417 controls and all667 source frames. Independent native replay is bit-exact and every source metric passes. Original terminal quiet fails root speed (.09404m/s > .05) and joint speed (p95 1.09577rad/s > .5; maximum5.57707 >2). The prepared same-MPC250 hold will not execute.

Proposed single experiment starts a new native MjData at unchanged v4 frame10, physically reexecutes the original applied MPC targets0..1116, then switches at the original returned-standing boundary1117 to the already qualified walk003 terminal BFM method. Its input reference is **walk002 native_original.npz**. BFM horizon8, position gain1, yaw gain4, raw actor scaling5, target mapping, history and goal expressions remain unchanged. No new MPC optimizer, model fit, gain sweep, reference edit or state injection.

| Segment | Global controls | Native steps | Controller |
|---|---:|---:|---|
| Canonical prefix, including all source and return | 0..1116 | 11,170 | Original saved actual MPC targets, fresh manual-PD physics |
| Original terminal standing and proof margin | 1117..1416 | 3,000 | Actual BFM, original goal frames1128..1427 |
| Separate5s hold | 1417..1666 | 2,500 | Same actual BFM, final goal frame1427 |

Before the first BFM inference, require every reexecuted prefix qpos/qvel, requested/applied torque, actual actuator force, time and warning sample to match the qualified main trace; all1117 actual targets, source indices, precontrol histories and prior actions must also match. Direct algebra already confirms all1118 previous actions and numerically equal histories; frozen observation expressions reproduce every history byte. The vectorized independent gravity expression creates eight signed-zero differences at initial history only; this is disclosed in `proposal.json`, with no tolerance introduced.

At1117 transfer the live measured BFM history and the prior raw action derived from actual MPC target1116. Afterward history stores BFM raw actor output×5 before native target clipping. Continue the same MjData and independently accumulated clock. Record complete291 integration state, ctrl, qacc_warmstart, warnings, named history and prior action at1117 and1417. There is no independent full291 snapshot at1117 yet; the saved boundary file is comparison-only and is forbidden as an initialization source. Root can reconstruct the boundary independently during the subsequent full hybrid audit. The existing root endpoint1417 belongs to the failed-quiet MPC lifecycle and cannot initialize this branch.

The archived runner needs bounded adapter work before launch: explicit walk002 timeline/hash assertions; current root-report schema; full prefix parity instead of q/dq/ctrl only; repeated binary64 clock; first-substep strict range threshold1e-6 instead of its old .01 abort; full291 and history continuity receipts; canonical trace aliases and empty/partial failure shapes. Preserve actor/backward and goal math byte-for-byte. Snapshot all dependencies, freeze the final command and run focused source/stub tests before independent review. Use one hidden durable launcher with output-absence checks, start/exit receipts and pre/post hashes.

Run separate250 only after complete lifecycle and strict physical pass. Report original300 and separate250 quiet windows independently; a later hold pass never changes an original quiet failure. Root must independently audit every new native step, original source alignment and both quiet windows. Successful evidence would qualify one **offline recorded-MPC-prefix plus actual-BFM-terminal hybrid**; it would not establish fresh online MPC, received-Pico causality, realtime execution or hardware readiness.

`proposal.json` binds37 source/input assets and16 passing read-only checks. `saved_boundary1117_comparison_only.npz` has state/history values for checks, never physics initialization. No policy inference, optimizer or dynamics executed during preparation. A concrete CLI contract is in `proposed_command.txt`; its runner remains an implementation target and is deliberately not executable yet.
