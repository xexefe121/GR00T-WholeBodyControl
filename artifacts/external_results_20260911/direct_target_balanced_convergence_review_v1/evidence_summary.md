Saved71000 fit improves overall errors, but leaves a large directional-control mismatch. This diagnosis used saved ledgers and predictions only; no new model or simulation calls.

InitialGPU32→finalGPU32 nominal loss improves3.52%, original full-state response2.34%, balanced response8.95%, physical response4.64%. All15 nominal and9 physical cells improve;49/54 response cells improve. Balanced response remains1.330734× the zero-response baseline; original response remains1.049253×. Directional odd error is98.14% of balanced error. A zero-response baseline is a diagnostic, not a proposed stabilizing controller.

| Tangent group | Response change | Final error / zero response |
|---|---:|---:|
| root_position | -16.45% | 1.608× |
| root_rotation | -0.58% | 1.041× |
| joint_position | -13.14% | 1.544× |
| root_linear_velocity | +0.14% | 0.977× |
| root_angular_velocity | -0.67% | 0.969× |
| joint_velocity | -10.81% | 1.845× |

Acquisition nominal loss improves8.38%; middle phase only0.97%; terminal5.93%. PICO middle remains the largest nominal cell, MSE0.00140924. These equal-cell averages do not show a missing phase in the sampler. Widening phase weights alone would trade existing cells against each other without directly repairing local response representation.

The exact query250/control250 saved nominal entry gets worse:0.0473776621→0.0479952678rad RMSE on GPU32. FinalORT64 is0.0479951952rad; no target components clip there. Left hip pitch remains the largest error,0.153246rad. Query250 first24 acquisition rows improve0.055837→0.053535rad, so even this small regional average hides the worsened first action. This is nominal evidence, not a prediction of the next actual failure time.

Native-target clipping decreases modestly: nominal5011→4946/9904 rows; full-state176448→173615/354612; physical1522→1500/3054. Teacher full-state native-clipped flags cover19298 rows; feedback-clipped flags64677. Student and teacher mask semantics are retained separately. Saturation remains widespread, but the first-entry error exists before saturation.

Warm LR restart produces a nominal-loss peak at update5,6.74× the initial batch loss, before recovery. Total gradient norm peaks0.555806, below clip10; clipping never limits optimization. After update250, norms remain around0.0012–0.0014. Late nominal and physical losses still decline slowly as LR approaches1e-6. Changing sampled response batches prevents reading the last two response-window means as a fixed-corpus plateau. There is no evidence here for raising the clipping ceiling or using scalar loss magnitude to select another coefficient.

The saved evidence establishes supervised underfit of directional responses. It cannot distinguish insufficient capacity from objective interference or insufficient optimization. Correct causal-history alignment and the prior matched context improvement weaken a pure bookkeeping explanation. Wider response coverage alone also has not solved the problem. Actual off-trajectory states may exceed these local single-axis neighborhoods; no capacity change certifies stability there.

If the71000 canonical fails, prepare one function-preserving width512 capacity experiment, keeping causal1323 inputs, all qualified labels, normalization, balanced coefficients and strict acceptance. Preserve old256 neuron blocks and optimizer moments; add seeded nonsymmetric incoming units with zero outgoing paths. This gives a concrete representational change while retaining the old controller at initialization. It is an engineering trial, not proof that capacity was the unique cause. Do not simultaneously reweight phases, retune derivatives, change PD gains, or select an intermediate rollout winner.

Use a meaningful fixed optimization proposal rather than another tiny low-rate continuation:10000 updates, retain warm old moments,250-step LR ramp1e-6→1e-5 followed by9750-step cosine1e-5→1e-6. The ramp responds to the observed warm-restart transient; fixed endpoint and independent saved metrics still govern evaluation. These optimizer details are a proposal requiring separate source/request review, not a selected fit. Synthetic expansion tests must prove original functions, all six optimizer states and the new gradient path before any actual checkpoint use.

Current1323→256→256→23 uses410112 dense MACs; width512 uses951296,2.32×. Prior68000's53 learned inference timings were median0.687ms,p951.218ms,max4.943ms. Those are inference measurements, not full controller deadlines, and linear extrapolation does not qualify the wider model. Require measured WSLbatch1 feature construction plus head+target processing below20ms before deployment, unchanged FP64/publicFP32 1e-5rad export checks, then original1569+conditional250 native acceptance. No hardware or connected clock qualification follows from this diagnosis.

Exact per-cell values, clipping counts, window means, entry vectors, input hashes and limitations: [report.json](report.json).
