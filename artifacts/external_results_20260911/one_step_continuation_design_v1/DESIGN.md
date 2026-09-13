This is one proposed continuation design, not a selected fit. No optimizer update, network evaluation, physics, new label, checkpoint selection or controller change was performed for this assessment. The forthcoming branch collection and its independent audit must finish before a concrete training request can bind its actual outputs and validity mask.

Keep the ordinary 70,000 head, its 1,069 inputs, 256/256 ELU architecture, residual output, original feature mean/std, joint spans and raw combined-action memory convention. Add one physical finite-response term to the existing nominal plus velocity objective. Use all valid physical endpoints every update, paired with the same dataset's nominal successor at c+1. Keep all 3,057 nominal anchors and the existing velocity-chord sampler and loss unchanged. Proposed budget: one fixed 5,000-update continuation, ordinary final 75,000 only.

The current 70,000 controller failed its canonical simulation at control 261. This proposal targets one-step state and memory departures; it does not establish recovery or a qualified controller. The captured physical branches are driven by frozen 70,000 commands. A new head will visit a different distribution, so even perfect training errors cannot replace a fresh canonical simulation.

Notation and exact algebra

For any saved row x, let b(x) be the original **unclipped** BFM base target in radians, t(x) the native-bounded fixed teacher target, s the unchanged 23 joint spans, and f_theta(x) the head's normalized residual output after the unchanged feature normalization. Its residual is d_theta(x)=s*f_theta(x), and its unclipped total proposal is u_theta(x)=b(x)+d_theta(x). Applied-target clipping belongs to the unchanged runtime and diagnostics; it is not inserted into the training loss.

The residual label is r(x)=t(x)-b(x). Thus normalized residual error f_theta(x)-r(x)/s equals normalized total-proposal error (u_theta(x)-t(x))/s mathematically. Preserve the original nominal float32 residual-label arithmetic exactly rather than replacing it with an algebraically equivalent operation that rounds differently.

For a physical endpoint p reached after control c, pair with nominal row n=(same dataset,c+1), never the start row c. Use the branch's newly evaluated BFM base b(p) and its full committed-map target t(p). Define:

    e_physical(p) = ((b(p)+s*f_theta(p))
                    -(b(n)+s*f_theta(n))
                    -(t(p)-t(n))) / s

The equivalent residual expression is f_theta(p)-f_theta(n)-(r(p)-r(n))/s. It is useful as an independent algebra check, but implement the explicit total-target form with the same dtype/order conventions as the existing velocity loss. Both predictions remain differentiable. Reuse f_theta(n) from the one full nominal forward pass, without detaching it or making a second center forward pass.

There is no division by displacement, velocity delta, previous-action delta, feature distance or perturbation amplitude. This is a finite response, not a derivative estimate. Root pose, joint state, raw previous action and their BFM consequences change together; the loss matches the total response across that coupled change. Omitting b(p)-b(n), substituting the nominal BFM base, or comparing only a head-output difference with the teacher's total-target difference would give the wrong label.

One objective and explicit weights

    L = L_nominal + L_velocity + L_physical

All three scalar coefficients are fixed at 1. No coefficient search, adaptive gradient balancing, error-based row mining or data-dependent phase weight is proposed.

L_nominal remains the original mean of nine dataset-by-phase mean squared normalized residual errors over all 3,057 rows. Dataset order is old, query1, query250. Phase intervals are acquisition 250..349, source 350..1168, return 1169..1268. Each dataset has counts 100,819,100; each cell gets coefficient 1/9. A row/joint coefficient is 1/(9*N_cell*23).

L_velocity remains byte-preserved source logic: each update draws 64 center-axis pairs in each of the nine cells using the restored global Torch RNG, then evaluates both signs, for 576 pairs and 1,152 probe rows. Keep its original teacher labels, BFM bases, live reused center predictions and normalized total-target finite-response arithmetic. All 140,622 velocity probes remain in the sampling population and full initial/final diagnostics. Sampling remains dataset then phase, exactly nine torch.randint calls per update; physical full-batch loss consumes no RNG draws.

Physical branch starts are 250..1267; their endpoints are 251..1268. Assign physical phase by endpoint/successor clock. The nine requested cells are:

| Dataset | Acquisition endpoints | Source endpoints | Return endpoints | Total |
| --- | --- | --- | --- | --- |
| old | 251..349: 99 | 350..1168: 819 | 1169..1268: 100 | 1,018 |
| query1 | 251..349: 99 | 350..1168: 819 | 1169..1268: 100 | 1,018 |
| query250 | 251..349: 99 | 350..1168: 819 | 1169..1268: 100 | 1,018 |

For requested physical cell C with N_C=99,819,100, let V_C contain its independently qualified complete endpoints. Use:

    L_physical = sum_C [ sum_(p in V_C) sum_j e_physical(p,j)^2
                        / (9*N_C*23) ]

The denominator uses the requested cell count, not the surviving count. A feasible row/joint coefficient is 1/(9*99*23), 1/(9*819*23), or 1/(9*100*23), repeated for each dataset. Publish valid, strict-failed and requested counts and the effective mass |V_C|/(9*N_C) for every cell. Empty cells contribute zero and are explicitly marked empty; their weight is not redistributed. This defines conditional supervised data only, not a zero error or success for failed branches. There is no training target at an incomplete physical endpoint. No valid branches would mean this proposal has no new physical training signal and requires a new root decision before fitting.

Use the collector's original 3,054-row identity table and immutable status masks. Index qualified rows before tensor construction; never multiply NaN placeholder rows by zero. Require every requested row to have its final nominal/physical status and bind the first strict failure plus partial native evidence for all failures. Any collection exception, unattempted row, missing endpoint audit, nominal mismatch or missing call accounting prevents preparation of a qualified fit request. Native-clipped commands, teacher-clipped responses, zero gains and replan-boundary cases remain included when physically valid, with group counts and separate diagnostics. No validity threshold or outlier cutoff is introduced here.

Why this choice over direct absolute branch loss

The simpler alternative is L_nominal+L_velocity+mean((u_theta(p)-t(p))/s)^2 on valid endpoints. Its target is correct when the new BFM base is included, and it directly anchors each endpoint. However, it mainly adds more absolute residual examples and does not explicitly separate nominal bias from the response to a coupled physical departure.

The selected finite-response term instead trains e_p-e_n, where e_p and e_n are normalized total-target errors. It matches the existing velocity objective and reuses the live nominal predictions at no extra network-row cost. The full nominal term anchors e_n. This is not equivalent to absolute branch MSE: the cross-term can trade nominal bias against response error. Therefore publish both absolute branch error and centered response error, before and after the fixed update budget. Retaining the old objective/data does not guarantee each old error remains unchanged. There is no stop-gradient trick, second absolute branch term or frozen-70,000 response distillation in this proposal.

Restoration and one update protocol

1. Bind ordinary-70,000 checkpoint, ONNX, completed fit/export/root reviews; original centers and complete velocity data/source; the final physical collection manifest, request, call ledger, all nominal replay proof, actual251 witness, full-state capture evidence and independent branch audit. Pin every source/runtime/input and the exact 3,054 identities, successor mapping, validity mask and nine-cell counts. No placeholder output paths/hashes may satisfy a launch gate.
2. Load the checkpoint on the same Windows CPU Torch runtime. Restore actor tensors, AdamW full state, six parameter step counters=70,000, Torch and NumPy RNG, feature mean/std and joint spans exactly. Check complete trees/bytes before changing the learning-rate schedule. No reseeding, optimizer reset, re-normalization, feature clipping, model change or output clamp.
3. Preserve original restoration checks: recreate byte-identical 70,000 ONNX; reproduce original all-3,057-row Torch predictions and nominal objective, full velocity diagnostics, and original fixed seven diagnostic inputs with their original batch conventions. Confirm diagnostics/export did not advance RNG. New branch restoration comparisons against WSL collection outputs use numerical export parity, never a false byte-parity claim across Windows batch256 and WSL batch1.
4. Proposed updates are exactly 70,001..75,000. Reuse the prior stage's fixed cosine schedule, lr(k)=3e-7+0.5*(3e-6-3e-7)*(1+cos(pi*k/4999)), k=0..4999. This intentionally resets only the schedule's starting lr from the restored final 3e-7 to 3e-6 after restoration proof; it does not reset AdamW moments or RNG. Keep weight_decay=1e-5, global gradient clip norm=10 with nonfinite rejection, CPU threads=1, and original architecture/dtypes.
5. Each update: sample original 576 velocity pairs, evaluate all 3,057 nominal rows once, evaluate 1,152 sampled velocity rows once, evaluate all V qualified physical rows once in fixed dataset/control order. Compute the three losses, make one backward pass and one AdamW step. Preserve all loss components per cell, sample identities/axes, LR, attempted/returned row counters, optimizer-call state and six step counters. No additional BFM, teacher, native simulation or ONNX inference occurs in optimization.
6. On any exception/nonfinite state, preserve current inputs, returned predictions, attempted step, committed losses, RNG/model/optimizer state and counters; label a possibly partial optimizer step explicitly. Do not retry or automatically resume. Preserve ordinary final 75,000 once all 5,000 updates complete; no early stopping, best checkpoint, additional epoch or alternative seed.
7. Export the ordinary final and repeat fixed numerical diagnostics. Saved-input errors are training diagnostics. Final numerical/source review and a separately selected strict fresh canonical simulation remain necessary; this design does not itself authorize either fit or rollout.

Fixed cost and diagnostic scope

Training network rows are exactly 5,000*(3,057+1,152+V), where 0<=V<=3,054. Maximum 36,315,000 rows versus the previous 21,045,000. This is 1.725588 times the previous row budget. Scaling the parent-reported approximately 485 s prior fit gives approximately 837 s (14 min), before added serialization/review overhead. This is a rough CPU estimate, not a benchmark or wall-time guarantee. Keep one CPU thread and the same runtime; no new performance run.

Proposed initial/final diagnostics preserve all 143,679 nominal+velocity rows and the same seven actual inputs; add all V qualified physical endpoints and their paired nominal successors using already computed predictions. Torch diagnostic rows are 2*(143,679+7+V), maximum 293,480. If original 563 ORT calls per stage are preserved and physical endpoints use separate deterministic batches of at most256, total head ORT calls are 1,126+2*ceil(V/256), maximum1,150. Fix the exact count after the final validity mask exists. BFM/teacher/native diagnostic calls remain zero. Existing 14 analytical sensitivity evaluations can remain separately counted and unchanged; do not silently add probes.

Report all nine nominal and physical cells, full velocity metrics, first24 phase rows, actual query250/251 witness-related saved-input predictions, preclip/applied target error, each joint, clipping groups, replan boundaries, zero gains and failed-row coverage. Compare retained losses to restored70,000 without claiming monotonic improvement or selecting a checkpoint. For future deployment parity, Windows batched checks do not replace a separately selected WSL batch1 activation witness for the new final head.

Known limits remain: one-control branches do not populate multi-step history drift, failed-row recovery, all source clips, terminal transitions, fault handling or live received-input timing. Physical validity of a sampled endpoint is not validity of its frozen K teacher beyond that local response. All three datasets are walk003; broader PICO/walk002 labels and held-out walk008 are excluded from this fixed proposal. No improved training metric changes the native 2 ms speed/range/effort/warning/clock gates, original full-source intent checks or separate continuous quiet hold.
