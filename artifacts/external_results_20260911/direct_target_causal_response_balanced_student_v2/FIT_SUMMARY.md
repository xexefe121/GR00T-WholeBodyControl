# Ordinary71000 saved-fit result

The fixed3000-update warm continuation completed, preserving the ordinary endpoint and all diagnostics. AdamW advanced3000→6000; training consumed exactly9000 forwards /44,058,000 rows. Initial outputs were byte-exact to causal68000 across all367570 rows.

| Metric | Initial GPU32 | Final GPU32 | Change | Final ORT64 |
|---|---:|---:|---:|---:|
| nominal_objective | 0.000325353705557 | 0.000313900699742 | -3.5202% | 0.000313900663362 |
| full_state_objective | 0.000207214816804 | 0.000202364100327 | -2.3409% | 0.000202364105564 |
| balanced_full_state_objective | 0.000281882735628 | 0.00025665183966 | -8.9508% | 0.000256651854832 |
| physical_objective | 8.4152042598e-05 | 8.02511313294e-05 | -4.6356% | 8.02510774126e-05 |
| weighted_objective | 0.000786392364108 | 0.000762215863069 | -3.0744% | 0.000762215782298 |
| balanced_weighted_objective | 0.000922199926906 | 0.000860955531787 | -6.6411% | 0.000860955469087 |

Final original-response /zero-response baseline: 1.049253. Balanced-response /weighted-zero-response baseline: 1.330734. Values above1 mean worse than zero response under that metric.

Same-weight CPU64/GPU64/ORT64 maximum preclamp difference: 1.02728607843e-08 rad; fixed gate1e-5 passed. Public input/output remain float32.

This is one engineering continuation, not a matched comparison against extra epochs or warm optimization. No physical trial is qualified by these losses or export checks. The earlier v1 metadata failure remains preserved with zero forwards/updates.

Source report SHA256: f918f1dfe675aae01e25307d302ce6c54f599fb7f0f5d1fa3fd943a179b5aded
