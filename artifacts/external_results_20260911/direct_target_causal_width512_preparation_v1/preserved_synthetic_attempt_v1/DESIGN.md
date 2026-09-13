# Width512 preparation only

No actual checkpoint load, task model evaluation, training, export validation, witness, or native run is authorized by this package. It contains an initializer and synthetic tests for one possible follow-up if the ordinary71000 canonical fails. The completed71000 endpoint remains unchanged.

The candidate retains all1323 causal inputs, existing normalization, qualified nominal/physical/full58 labels,54-cell energy balance, coefficient1.8188207859141674, target cast/clamp contract and strict1569+conditional250 acceptance. It widens1323→256→256→23 to1323→512→512→23. Six parameter tensors remain; this is not an ensemble or a different controller clock. No phase reweight, gain change, label change, extra causal input or checkpoint selection is included.

## Exact old function at initialization

Copy the old first-layer256 rows, second-layer top-left256×256 block, first256 second-layer biases, final first256 columns and all23 final biases. Initialize new first-layer rows/biases and new second-layer bottom rows/biases with independent uniform draws from a local CPU generator seeded20260912, using bounds1/sqrt(fan-in). Zero the second-layer top-right block and final-layer new256 columns. Added units are nonsymmetric, while their paths to the old outputs initially have zero weights.

GPU32 execution keeps the old256 contractions exactly shaped and contiguous. The old first layer still performs the reviewed1000-feature contraction plus323-context contraction. Old second-layer and final contractions keep their256 dimensions. New contributions are added after those contractions. The stored tensors are still whole512 layers; differentiable contiguous copies preserve gradients into their slices.

Ordinary IEEE addition of +0 can change the sign bit of an old -0 output. `zero_preserving_add` therefore returns the old value exactly when the added value is zero; otherwise it returns their sum. Its custom backward returns the incoming gradient to both equal-shaped inputs, the exact derivative of addition over real numbers. This preserves all old output bits without disconnecting initially zero paths. Synthetic signed-zero, gradient and second-gradient tests qualify this small numerical primitive. It does not change the mathematical network, labels or loss. Final monolithicFP64 evaluation/export is still mandatory, with the original1e-5rad preclamp cross-backend gate; no bit-equivalence is claimed between FP32 and FP64.

At the first training backward, new final columns and second-layer top-right columns can receive gradients from their nonzero activations. New incoming rows can have zero task gradient on that first backward because outgoing paths are zero. After one synthetic update opens those paths, their gradients must be nonzero. This is delayed opening, not an all-zero dead expansion.

## Warm optimizer and RNG

Require a causal ordinary71000 source with optimizer step6000, ordered six actor tensors, finite float32 parameters/moments, nonnegative squared moments and one qualified AdamW group. Preserve parameter identities, all existing moment blocks and the whole parameter-group dictionary, including saved LR1e-6, before any explicit schedule change. Fill only newly added moment slots with zeros; all six scalar float32 step counters remain6000. New entries share the existing tensor's bias-correction age. This is an explicit warm-state choice, not a fresh per-neuron Adam optimizer, and may influence adaptation speed.

Use deep copies, so the source capsule is not mutated or aliased. Preserve source Python/NumPy/CPU/CUDA RNG payload exactly for later restoration. The expansion's local generator does not consume any global RNG. Save its seed and final local generator state separately. Construction of the synthetic module preserves global CPU RNG as well. Future actual request must bind the source71000 checkpoint hash395aabf879c0a40b5187a4265ae850de49cfb62583e144145c1986d65402143d, completed source reviews and actual restoration evidence before task use.

## Fixed optimization proposal, not selection

Propose10000 updates, ordinary81000 and optimizer16000, retaining nominal9904 + physical3054 +1728 full-state endpoint rows each update. Use the already qualified10000×864 schedule from the full-state study; verify its first3000 rows equal the current schedule. That gives146,860,000 training rows and30000 training forwards, with no new sampling draws or calibration. Initial/final diagnostics retain the five existing367570-row corpus/backend passes and1437 partitions each. No intermediate rollout or best-checkpoint export.

The saved71000 run showed a6.74× initial nominal-loss spike after an immediate1e-6→1e-5 warm restart, while no gradient was clipped. Propose a250-update inclusive linear ramp1e-6→1e-5, then9750 updates with inclusive cosine1e-5→1e-6. Weight decay1e-5 and clip10 remain. This schedule is a reasoned proposal addressing observed optimizer transients, not a proven better schedule; changing width and schedule prevents a clean capacity-only causal inference. Fixed final endpoint and strict physical acceptance remain necessary. No actual fit is selected here.

## Deployment and inference limits

Dense arithmetic rises410112→951296 MACs per example,2.32×. Split execution also adds kernel/dispatch overhead. Prior causal68000's saved53 learned inference samples had median0.687ms,p951.218ms,max4.943ms; they do not establish end-to-end deadlines or timing for this wider model. Require a separately selected real WSLbatch1 measurement including feature assembly, head and target reconstruction under20ms, then the unchanged full native pipeline. CPU synthetic tests do not qualify CUDA accumulation, finalFP64 parity, contacts or stability.

Observed supervised directional mismatch remains the rationale: balanced response1.330734× zero-response baseline,98.14% odd error, first-entry target RMSE0.047995rad despite aggregate gains. Width may help representation, but saved scalar losses cannot separate capacity, objective interference or optimization; local committed-map labels also do not prove correctness on larger coupled actual departures. See the source-bound saved diagnosis in `../direct_target_balanced_convergence_review_v1`.
