# Additional quiet-standing diagnostic

Declared 2026-09-10, approximately18:47 UTC, before observing the new
walk003 terminal-BFM hybrid result. The original15 recorded-source gates
in SIM_ACCEPTANCE.md remain unchanged.

Score the final3 seconds of the original complete lifecycle, using every
2 ms physical state. A separately requested extension cannot replace that
window. The old MPC standing segment uses the same comparator definition.

- All original native position, speed and effort checks and warning checks pass.
- Root XY error p95 relative to declared standing goal <=0.05 m.
- Original heading absolute error p95 <=5 degrees.
- Root linear speed p95 <=0.05 m/s.
- Maximum-over-joints absolute joint speed p95 <=0.5 rad/s.
- Maximum joint speed anywhere in the window <=2 rad/s.
- Root tilt anywhere in the window <=0.15 rad.

Report each predicate and its raw measured quantity. These are engineering
settling diagnostics for this nominal simulation, not hardware certification
or a replacement for packet-loss, perturbation and live timing validation.
