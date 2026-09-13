**The nominally successful teacher already uses abrupt position targets. Near the PICO fixture, normalized action/history excursions become much more frequent. These observations do not establish why a student or BFM handoff fails.** All figures below come from saved actual commands and measured states; no policy inference, optimization or dynamics were run.

| Source-only interval | Controls | Target jump p95 / max, rad per 20 ms | Target slew p95 / max, rad/s | Measured joint average speed p95 / max, rad/s | Normalized action p95 / max | Controls with any action outside ±5 |
|---|---:|---:|---:|---:|---:|---:|
| Passing walk003, complete source 0–16.38 s | 819 | 0.791 / 4.353 | 39.55 / 217.63 | 3.05 / 21.25 | 3.461 / 6.653 | 3.66% |
| Original PICO source before fixture, 0–67.80 s | 3,390 | 0.820 / 4.329 | 40.98 / 216.45 | 3.10 / 26.44 | 3.160 / 9.618 | 15.66% |
| Original PICO final 65 controls before fixture, 66.50–67.80 s | 65 | 0.756 / 3.479 | 37.81 / 173.95 | 4.02 / 7.74 | 5.140 / 6.776 | 81.54% |
| New restoration continuation, 67.80–69.10 s | 65 | 1.174 / 4.061 | 58.70 / 203.07 | 3.09 / 8.96 | 5.708 / 8.480 | 89.23% |

Percentiles pool absolute joint components over the indicated controls. Target slew is `(actual_target[c] - actual_target[c-1]) / .02`; each interval includes the real preceding command at its first boundary. Measured average speed is `(q[c+1]-q[c])/.02`. These target-rate values are **not motor-speed violations**: native every-2-ms measured speed ratio maxima are respectively 0.808, 0.860, 0.445 and 0.526. No target-rate limit is substituted for the existing actual physical gates.

The largest passing walk003 target jump is the right shoulder pitch at source 2.26 s: target changes from -2.41191 to +1.94076 rad in one control, while measured joint position changes only -0.20778 to -0.17162 rad. A leg example is right hip yaw at 7.66 s: target changes 3.95322 rad while the actual joint changes 0.05162 rad. In restoration at source 68.58 s, left hip yaw target changes +1.98182 to -2.07952 rad; measured joint changes -0.76729 to -0.86075 rad. These commands exploit the declared PD/effort-limited plant rather than requesting that the joint instantaneously reach the command position.

The command-to-precontrol-joint absolute gap p95/max is 0.743/2.772 rad for passing walk003, 0.860/2.199 rad for the preceding PICO 65 controls, and 0.992/2.812 rad for restoration. The 12-leg target slew p95 rises from 46.95 rad/s in passing walk003 to 52.19 in preceding PICO and 64.86 in restoration; arm values are 28.84, 18.34 and 48.22 rad/s. Thus the abrupt commands are not confined to wrists or a single joint group.

Normalization exactly follows recorded-control BFM history construction, preserving arithmetic order and float32 conversion:

`a[c] = float32((target[c] - default_q) * kp / (0.25 * training_effort))`

No clipping to ±5 is applied. `previous_action[c]` must equal `a[c-1]`; the four saved precontrol action-history rows equal `a[c-2], a[c-3], a[c-4], a[c-5]`. All previous-action and four-history-row reconstructions match the saved arrays **bit for bit** over every analyzed control, including the restoration boundary. The ±5 figure is a declared reference band, not a measured bound on the complete original BFM training distribution; exceeding it alone does not prove the input is out of distribution.

At the exact original control-3740 fixture (source 67.80 s), previous action maximum is 6.34475: left hip pitch is -6.31457 and right knee +6.34475. Two of 23 previous-action components exceed ±5. Nine of 92 action-history components exceed the band; history maximum is 5.94342. Actual instantaneous joint speed maximum is only 0.1443 of its native limit. Fixture q/dq and initial continuation q/dq match the original producer exactly; the saved fixture's **future private BFM proposal targets are excluded** from actual-command statistics.

Restoration's maximum normalized action 8.47993 occurs in waist yaw. Legs reach 7.24677, arms 4.43666. Every control with an excursion has a leg excursion; 26.15% also have a waist excursion. Component-wise outside-band frequency rises from 5.75% in the preceding PICO interval to 9.30% in restoration. The four-action history p95/max rises from 5.070/6.776 to 5.648/8.480, so a BFM handoff receives persistent recent command excursions, not only a single unusual current action.

Implication for a fast student: pose proximity alone does not specify these rapidly changing, often distant PD targets. Low average imitation error at teacher states therefore need not preserve the teacher's exact contact/effort sequence. Implication for handoff: actual MPC-derived action/history can differ materially from what the frozen actor itself would have generated. Both are plausible difficulties supported by these measurements; neither is causal proof. This audit proposes no target smoothing, action clipping, controller change or relaxed threshold.

The restoration record contains **65/100 requested controls**, 650 actual physics steps, then `no_feasible_seed` at global control 3805. It is an incomplete continuation, not a successful two-second recovery. Original PICO prefix statistics likewise do not imply its full-source qualification. Passing walk003 is the already independently qualified recorded-source hybrid; only its unchanged MPC source controls 350–1168 are included here.

`report.json` contains per-group p50/p95/p99/max/RMS, command-to-joint gaps, history values, largest-jump joint/time witnesses, exact source ranges, and SHA256 input provenance. `audit_saved_commands.py` uses NumPy and saved arrays only. Existing sources and traces are unchanged.
