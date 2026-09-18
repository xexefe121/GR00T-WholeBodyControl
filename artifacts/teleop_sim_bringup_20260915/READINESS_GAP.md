# True23/PICO readiness gap — 2026-09-15

## Scope and result

This is a read-only evidence audit. No headset or robot connection was attempted, no robot-facing process was started, and no evidence file was created, altered, backdated, or stubbed.

The only readiness improvement available from existing state is already present: `shadow_binary` passes for `gear_sonic_deploy/target/release/g1_true23_shadow_gate`. The four checks investigated here cannot be made to pass from current legitimate inputs. The final readiness result remains `NO-GO`.

Commands below ran under WSL Ubuntu 22.04 with `/root/venvs/teleop23` activated. Large command outputs were written to `/mnt/e/codex-artifacts/`.

## 1. `trained_checkpoint`

### Exact acceptance contract

`gear_sonic/scripts/true23_pico_readiness.py` calls `_check_checkpoint`, which loads a safe true23 checkpoint and requires all of the following:

- a safe true23 schema header and `g1_23dof_metadata` matching the current native true23 contract;
- `checkpoint_stage == "trained"`; `checkpoint_initialization` is explicitly rejected;
- a positive extracted global step, identical to `g1_23dof_training_evidence.global_step`;
- policy-state SHA-256 recomputed from the checkpoint and identical to the evidence value;
- exact training-evidence keys: `schema_version`, `kind`, `producer`, `robot_model`, `global_step`, `history_length`, `observation_layout`, `decoder_input_dim`, `decoder_output_dim`, `reference_profile`, `reference_contract`, `source_family`, `source_revision`, `source_checkpoint_sha256`, `initial_policy_state_sha256`, `training_start_global_step`, `training_updates`, `minimum_training_updates`, `policy_state_sha256`, `motion_dataset`, `training_material`, and `weights_only_initialization`;
- a pinned approved warm-start lineage, validated motion-dataset and training-material provenance, changed policy weights, and `training_updates == global_step - training_start_global_step` with at least 50 updates.

The producer is the real trainer/checkpoint writer: `gear_sonic.trl.callbacks.ModelSaveCallback` (training entry points include `gear_sonic/train_agent_trl.py`; the exact checkpoint record validator is `gear_sonic/utils/g1_23dof_artifact.py:validate_training_checkpoint_records`). The simulation/export documentation requires the trainer-produced weights-only `*.promotion.pt` sidecar, not a full resume checkpoint.

### Existing artifacts checked

The default checkpoint is real but initialization-only:

`/mnt/z/codex/GR00T-WholeBodyControl/sonic_release/g1_23dof_rev_1_0_init.pt`

- SHA-256: `7d014520894c6ae7d5fe242b5b30c53cecc94b8b76699fd77e9a9e99214e542d`
- policy-state SHA-256: `c247e5cf8bf06bc954db314013cba5ed8b56b6fe4c9a952c19f053583714f0bc`
- result: `checkpoint_stage=checkpoint_initialization`; rejected as requiring genuine retraining.

The alternative release initialization checkpoint is also rejected:

`/mnt/z/codex/GR00T-WholeBodyControl/sonic_release/g1_23dof_rev_1_0_low_latency_init.pt`

- SHA-256: `5e5be23982f15eaf2eb1d52b2433d081b5f43c260ecb5055457edf222b77c9bb`
- result: initialization-only.

A ZIP-metadata scan found 30 other repository checkpoints carrying true23-related metadata. The readiness validator was run against every one. All 28 diagnostic checkpoints failed `checkpoint lacks the safe true23 schema header`; the two release checkpoints were the initialization-only files above. Per-file results are in:

`/mnt/e/codex-artifacts/true23_checkpoint_candidate_validation_20260915.jsonl`

`/mnt/e/codex-artifacts/true23_checkpoint_candidate_validation_part2_20260915.jsonl`

The full repository and `/mnt/e/codex-artifacts` filename scan found no `*.promotion.pt` file. The E: policy-sweep result is explicitly non-deployment diagnostic evidence, not an accepted checkpoint:

`/mnt/e/codex-artifacts/teleop_policy_sweep_20260915/paired_encoder_20260905_v2__original_breadth25__model_25/report.json`

It states `deployment_ready: false`, `hardware_authorized: false`, and `recorded_simulation_only: true`.

### Legitimate next requirement

This stays `NO-GO` until a genuine native true23 training run starts from an approved warm start, performs at least 50 policy updates, and emits a safe, weights-only trained promotion checkpoint with the exact current evidence/provenance records. Existing diagnostic or initialization weights cannot be converted into this evidence without falsifying lineage.

## 2. `simulation_evidence`

### Exact acceptance contract

The input is one strict JSON report, `kind == "g1_23dof_sim_validation"`, schema version matching `SIM_VALIDATION_SCHEMA_VERSION`, and no unrecognized or omitted top-level keys. Required top-level keys are:

`schema_version`, `kind`, `robot_model`, `checkpoint_sha256`, `producer`, `reference_profile`, `reference_contract`, `observation_layout`, `history_length`, `decoder_input_dim`, `decoder_output_dim`, `decoder_output_layout`, `runtime_config`, `simulator`, `material_provenance`, `trace_manifest_sha256`, and `runs`.

The report must bind the exact supplied checkpoint SHA-256, its reference profile and reference contract, and pass all current asset/config/runtime hashes. Its producer must exactly equal configured approved producer `kind`, `version`, and runner SHA-256. It must contain raw JSONL trace records whose file SHA-256 and canonical-payload SHA-256 match the report. The validator replays/recomputes all per-run metrics; it does not trust a boolean pass flag.

Required campaign coverage is exact: scenarios `nominal`, `disturbance_50`, and `disturbance_100`; seeds `1729`, `2718`, and `3141`; 22 episodes per seed; 250 steps (5.0 s at 50 Hz) per episode. This is 198 episodes and 49,500 simulation steps. Each trace must show valid 23-action behavior and zero termination, nonfinite, and soft-limit counts. The report also requires recomputed phantom-slot, recovery, saturation, and MPJPE metrics within hard thresholds, plus a canonical trace-manifest hash.

Producer script: `python -m gear_sonic.scripts.run_g1_23dof_sim_validation --checkpoint /path/to/trained_true23.promotion.pt --output /path/to/sim-report.json`. `--validate-only` validates an existing report; it cannot create evidence.

### Current blocking condition

No accepted trained checkpoint exists, therefore no report can bind to one. More importantly, current checked-in approval config is itself closed:

`gear_sonic/config/sim_validation/g1_23dof_rev_1_0.json`

contains `producer.promotion_enabled: false`. `validate_simulation_report` fails any report while that value is false, even if syntactically complete. Changing it without completed, reviewed real campaign evidence would be faking promotion and was not done.

The documented MuJoCo sim-to-sim workflow is useful candidate qualification, but this readiness check calls `validate_simulation_report` and presently accepts only the approved IsaacLab report schema/producer above. It cannot substitute for this check without a separately reviewed readiness-gate change.

### Existing artifacts checked

No file with the required `g1_23dof_sim_validation` report kind was found in repository artifacts or in `/mnt/e/codex-artifacts`. Old simulation/diagnostic reports do not contain the required report kind, raw-trace hash chain, and trained-checkpoint binding.

No producer run was attempted: the producer correctly rejects an initialization checkpoint before it can generate promotable evidence, and any output would still be rejected while promotion is disabled.

### Legitimate next requirement

First produce the trained checkpoint above. Then run the real fixed IsaacLab disturbance campaign at the stated 198-episode/49,500-step scale. After a review that authorizes the current producer/config hash, enable promotion through the approved governance path; only then can the report pass this gate. This requires training and simulator access, not robot hardware.

## 3. `paired_onnx`

### Exact acceptance contract

Three separate files are required: static encoder ONNX, static decoder ONNX, and a strict JSON metadata sidecar. The sidecar must have `artifact_kind == "g1_23dof_validated_teleop_encoder_decoder_onnx_pair"`; name the exact supplied three basenames; and contain a valid `metadata_payload_sha256`, computed as SHA-256 of canonical JSON after removing that field.

It must satisfy the full native true23 deployment contract: trained/deployment-ready metadata, fixed H10 canonical IL29 observations, 267-float encoder input, 64-float token, 994-float decoder input, and 23-float action output. The sidecar's `hashes` bind checkpoint, policy state, encoder state, decoder state, both ONNX byte streams, raw simulation-report bytes, canonical simulation-report payload, local robot/config/contract files, and embedded ONNX metadata for both roles.

The verifier also requires:

- sidecar `training_evidence` to pass the same checkpoint-record validation as check 1;
- sidecar `simulation_evidence` to agree exactly with a complete recomputation from the supplied report/traces;
- matching training/simulation material provenance;
- each ONNX graph's structure and embedded metadata to equal the sidecar contract;
- CPU onnxruntime chained dry run: zero `[1,267]` input produces finite encoder `[1,64]`, then finite decoder `[1,23]` output.

Producer script: `python -m gear_sonic.scripts.export_g1_23dof_onnx --checkpoint /path/to/trained_true23.promotion.pt --simulation-report /path/to/sim-report.json --output /path/to/pair`. Its implementation is `export_validated_true23_artifact` in `gear_sonic/utils/g1_23dof_artifact.py`. It atomically refuses overwriting an existing triplet.

### Existing artifacts checked

Many old `*.diagnostic.encoder.onnx` / `*.diagnostic.decoder.onnx` files exist under `artifacts/g1_true23_frozen_lora/`; they have no validated-pair sidecar and are not deployment evidence. The concrete E: policy-sweep report above binds one such diagnostic pair and labels it non-deployment-ready. No required validated-pair metadata sidecar or matching report/checkpoint chain was found.

No export was run. The exporter first validates a trained checkpoint and checkpoint-bound simulation report, both unavailable. Running it against the initialization checkpoint would only produce a truthful rejection, not an artifact.

### Legitimate next requirement

After checks 1 and 2 have real passing inputs, run the existing exporter. No retraining beyond check 1 and no hardware are required for this final offline export, but it is impossible beforehand because its required hash chain does not exist.

## 4. `integrated_live_shadow`

### Exact acceptance contract

This input is a strict JSON object of schema version 2 and kind `g1_true23_integrated_live_shadow_evidence`. Required root keys are:

`schema_version`, `kind`, `captured_at_utc`, `producer`, `artifact_hashes`, `source_contract`, `window`, `pico_samples`, `lowstate_samples`, `inference_samples`, `summary`, and `authorization`.

It must be no more than 30 seconds old at validation. It must bind SHA-256 hashes for exactly five files: `checkpoint`, `simulation_report`, `encoder_onnx`, `decoder_onnx`, and `metadata`. It also binds the materialized `g1_true23_live_shadow` ELF, its pinned source SHA-256, and approved source/core/audit/approval-manifest hashes.

At least 25 advancing samples are required from every source, with 40–60 Hz effective rate, gaps no greater than 100 ms, and one common window no longer than 15 seconds. Every PICO sample must have valid calibrated body/tracker/controller semantics and advancing source timestamps/frame indices. Every LowState sample must be CRC-valid, advancing, `mode_machine == 4`, and use the exact 23 active hardware slots. Every inference sample is independently reconstructed through 267 -> 64 -> 994 -> 23 ONNX replay and semantic-reference/H10 checks. Outputs must be finite with zero physical target-limit and slew violations. The authorization object must prove all command/publisher/motion-switcher/robot-mutation authority is false.

Producer executable and source: `gear_sonic_deploy/target/release/g1_true23_live_shadow`, implemented by `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_true23_live_shadow.cpp`. Validator: `gear_sonic/utils/g1_23dof_live_shadow.py:validate_live_shadow_evidence`.

### Existing artifacts checked

Old JSON/JSONL files under `artifacts/g1_true23_frozen_lora/physical_dance_v1/` contain a `live_shadow_evidence` field, but they are historical diagnostic/actuation artifacts, not fresh schema-v2 integrated evidence bound to the five current true23 files. They cannot pass the 30-second freshness rule, and their diagnostic model chain is not a validated pair.

No live producer run was attempted. The verified context has no PICO headset and no robot Ethernet connection; attempting it would not create authentic common-window PICO/LowState evidence.

### Legitimate next requirement

This remains hardware-blocked. Obtain/configure an approved PICO headset and networked G1 robot, retain the complete passing trained/simulation/ONNX chain, then run the read-only integrated shadow producer during a real live common window. This proof does not authorize motor commands or gantry testing.

## Search and validation commands

```bash
# Candidate inventory across repo artifacts, sonic_release, and E:.
find /mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts \
     /mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/sonic_release \
     /mnt/e/codex-artifacts -type f \
     \( -name '*.promotion.pt' -o -name '*23dof*.pt' -o -name '*true23*.pt' \
        -o -name '*.onnx' \) -printf '%s %p\n'

# Read-only readiness validation of every metadata-bearing candidate.
source /root/venvs/teleop23/bin/activate
python .codex_tmp_validate_checkpoint_candidates.py

# Final best-legitimate readiness invocation. No valid optional artifact path exists.
source /root/venvs/teleop23/bin/activate
python gear_sonic/scripts/true23_pico_readiness.py --json-only
```

The temporary candidate-validation helper was removed after recording its E: outputs; it was a thin loop around the repository's own `_check_checkpoint` function.

## Final readiness JSON

Command exit status: `1` (expected for `NO-GO`). Full machine-readable record:

`/mnt/e/codex-artifacts/true23_readiness_final_20260915.json`

Relevant final result:

```json
{
  "shadow_readiness": "NO-GO",
  "checks": {
    "trained_checkpoint": "FAIL: valid native true23 initialization checkpoint only; genuine retraining required",
    "simulation_evidence": "FAIL: simulation report missing: None",
    "paired_onnx": "FAIL: encoder, decoder, or metadata sidecar missing",
    "shadow_binary": "PASS: g1_true23_shadow_gate ELF, sha256 3a9994255a287cb6ab7d067fae831ecde7fd144faadbc836155864398018d1d9",
    "integrated_live_shadow": "FAIL: fresh integrated live-shadow evidence missing"
  }
}
```

The unchanged non-target blockers are PICO ADB/package confirmation, PICO private IPv4 reachability, and robot private IPv4 reachability. They require the absent headset and robot connection. The readiness command also always leaves robot actuation and gantry testing `NO-GO` by design.

## Remaining NO-GO list

1. `trained_checkpoint`: genuine approved-lineage native true23 training run of at least 50 policy updates, producing a safe trained promotion checkpoint.
2. `simulation_evidence`: real fixed IsaacLab campaign of 198 episodes / 49,500 steps on that exact checkpoint, followed by approved promotion enablement for current runner/config hashes.
3. `paired_onnx`: offline exporter run after 1 and 2; no hardware, but exact chained hashes required.
4. `integrated_live_shadow`: live PICO and connected G1 robot, plus the complete passing artifact chain; at least 25 real common-window samples at 40–60 Hz, captured within 30 seconds of validation.
5. `pico_adb_apk` and `pico_network`: headset purchase/access, hardened package install, authorized ADB, and private IPv4 connectivity.
6. `robot_network`: connected robot on a configured private IPv4 network.

No physical robot session is authorized from current evidence.
