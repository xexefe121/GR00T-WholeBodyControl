The 75000 controller failed during acquisition, before any of the 819 source controls. At control 319, substep 3, the left ankle roll reached 0.2665440468 rad against the unchanged upper bound 0.2618. Root independently reproduced all 3,193 actual physics steps exactly. The failure is part of a growing multijoint deviation, not an isolated reporting error or a reason to loosen the limit.

The saved-outcome audit passed 4,474 checks over all 320 issued commands, including the partial last control. Actual measured state, full integration slices, features, named/flat history, raw prior/action, unclipped proposal, applied target and applied-action diagnostic are exact. Original BFM controls 0–249 and the control-250 state/input/own-head-witness parity are exact. All 70 learned controls 250–319 were retained; no inference, optimizer, physics or replanning was performed by either diagnostic.

The first input departure from the matching query250 expert is control 251. At 250, the exact initial input already yields 0.048884 rad target RMSE against that expert. At 251, measured joint-velocity deviation is 0.324213 rad/s RMS, BFM-base change is 0.010231 rad RMS and residual-head change is 0.054261 rad RMS against saved final-head predictions on the matching expert input. These saved predictions were produced by Windows Torch; their difference from the WSL first-activation witness at the identical input is only 2.54e-7 rad RMS, explicitly diagnostic rather than a byte-equivalence claim across batching/runtime.

The first learned target clip is control 264, at the left ankle pitch. It creates a raw-versus-applied action difference of 0.3236924, which enters direct previous-action features at 265 and lagged action history at 266. No such raw/applied difference occurs in learned controls 250–263. Named actor history starts differing from the expert at 252 because measured states and previously issued commands already differ. Thus clipped-command feedback cannot explain the initial amplification, though it can affect later behavior. Of the 70 learned commands, 47 are clipped. The first coordinate-range excursion outside the complete training feature envelope (3,057 centers, 140,622 velocity probes and 3,054 physical branches) occurs at 265 through raw prior; direct velocity leaves its range at 273. Range membership alone is not a stability or coverage proof.

All 70 nominal saved-plan targets reconstruct byte-exactly before evaluating those same committed feedback maps on actual student states. The table uses radians for target/pose errors and radians/second for velocity errors.

| Control | Actual target vs nominal expert RMS | Actual target vs fixed map RMS | Joint pose drift RMS | Joint velocity drift RMS |
|---:|---:|---:|---:|---:|
| 250 | 0.048884 | 0.048884 | 0 | 0 |
| 251 | 0.066735 | 0.066119 | 0.004989 | 0.324213 |
| 260 | 0.235322 | 0.265603 | 0.014957 | 0.340704 |
| 264 | 0.138645 | 0.168601 | 0.017107 | 1.071166 |
| 270 | 0.431882 | 0.416476 | 0.050496 | 1.174690 |
| 290 | 0.405098 | 0.393523 | 0.158800 | 2.968078 |
| 319 | 0.797203 | 0.800800 | 0.391809 | 5.293559 |

The fixed map's change from its nominal target remains bounded by the original correction clips: at 319 its RMS change is 0.095656 rad while the student's map error is 0.800800. The feedback correction hits its ±0.1 clip on 69 of 70 actual states, and final native clipping is active on 12 rows. These are matching-clock **committed** maps, not fresh MPC solutions, safety witnesses, or expert truth on the diverged states. They ignore actor history/prior and do not establish feasibility or recoverability.

At precontrol 319 the failing ankle is already at 0.242384 rad with +7.556338 rad/s velocity. Both the student and the fixed map request the same upper-bound target 0.2618; the recorded plant still crosses the bound three substeps later. Merely matching this stale map's ankle target at that point would not distinguish a safe control. The broader pose/velocity error and loss of measured-state coverage must remain visible.

A narrowly controlled change to retain inverse-applied-target feedback only on actually clipped learned components would test the later unsent-command feedback hypothesis. The exact original prefix through control 264 and state 265 should match before the first changed prior. It would not test or fix the initial control-251 discrepancy, and its behavioral result is unknown. This note selects no new controller or fit; the original raw-history failure is preserved.

Evidence: `student_physical_response_saved_outcome_v1/report.json` SHA f268678450a143ff9e8249d91edc00554591cfaac2a1fb134d1faac5d9dd57a3; this directory's `report.json` SHA 1377e85f7321848931bd990c1582518c438db411117f599c4f7aeda8b6d03e12. Each request pins its source and actual inputs. The original 70000 diagnostic sources remain unchanged. The first direct WSL invocation failed before Python because the E: mount was absent; that receipt is preserved. The successful audits used the existing pinned bootstrap and made zero model/native calls.
