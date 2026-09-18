# Teleop sim bring-up — live log (started 2026-09-15)

**Goal (from operator, 2026-09-15):** get teleop actually running in simulation on
this workstation. No PICO headset is present, so the live headset capture step
stays unproven; everything downstream of the input boundary must run end to end
against recorded or synthetic PICO packets. Hardware comes later: write the
procedure, do not command the G1.

**Scope guard:** no physical robot operation is authorized by this document.
`free_standing_authorized: false` in the promotion sidecar is unchanged.

## Environment facts established at start

| Fact | Value |
|---|---|
| Repo | `Z:\codex\GR00T-WholeBodyControl-sonic-transfer-23dof` |
| `Z:` free space | **447 MB of 391 GB — effectively full** |
| WSL distro | Ubuntu 22.04.5 LTS, ext4 root with 772 GB free |
| WSL RAM | 7 GB total |
| GPU | RTX 3070 Laptop, 8 GB, driver 576.02 / CUDA 12.9, visible inside WSL |
| XRoboToolkit PC service | `roboticsservice` 1.0.0.0 installed at `/opt/apps/roboticsservice`, **not running** |
| `xrobotoolkit_sdk` | importable only from the *other* repo's venv, `/root/GR00T-WholeBodyControl/.venv_teleop` |
| This repo's `.venv_teleop` | **did not exist** |
| Windows host Python | 3.10.11, no teleop dependencies |

## Blockers found before any code was run

1. **`Z:` is full (447 MB free).** `install_scripts/install_pico.sh` creates
   `.venv_teleop` in the repo root, which needs several GB for torch alone. This
   is the first reason a from-scratch teleop install fails on this machine.
   *Resolution:* the virtualenv is built on WSL ext4 at `/root/venvs/teleop23`
   instead of in the repo. The repo tree is only written for the small editable
   install metadata. Large run artifacts go to `E:\codex-artifacts\`.

2. **No teleop virtualenv for this repo at all.** The only working
   `xrobotoolkit_sdk` build belongs to the parent `GR00T-WholeBodyControl`
   checkout, so nothing in this repo could import it.

3. **XRoboToolkit PC service is installed but not running**, and no PICO app has
   ever connected. This is expected to remain unresolved while no headset is
   present.

## Progress

- [x] Environment survey
- [x] Teleop virtualenv built on ext4 and imports verified
- [x] Minimal no-headset sim teleop path identified and documented
- [x] Sim teleop loop runs end to end on recorded packets
- [x] Hardware procedure written for the later, supervised robot session
- [x] Transport-fault degradation tested (timeout / gap / stale)

## Entry-point survey (codex)

Survey date: 2026-09-15. This survey was read-only apart from this appended
section. `Z:` remains unsuitable for environments or run outputs; `/mnt/e` is
mounted in Ubuntu and maps to `E:\codex-artifacts`.

### Result

Shortest viable no-headset, no-C++ path is
`gear_sonic.scripts.record_g1_true23_saved_teleop_diagnostic`. It loads a saved
PICO causal-packet bundle in-process, applies native23 PICO retargeting, runs
the hash-bound ONNX encoder/decoder in CPU MuJoCo, and writes both a measured
trace and health values: pass/fail, fallback state, minimum base height, and
maximum base tilt. It opens no socket, DDS channel, robot channel, headset
service, or C++ deployment.

The exact saved source and paired policy artifacts exist. The current first
runtime blocker is absent `pyzmq` in `/root/venvs/teleop23`; this runner imports
the saved-packet loader from the ZMQ publisher module, so it needs `zmq` even
though it opens no socket. MuJoCo 3.5.0, onnxruntime 1.23.2, NumPy, and SciPy
are already visible to that venv. The survey did not install anything.

After `pyzmq` is installed in `/root/venvs/teleop23`, run this command from
Windows. It writes only to `E:`, and the output directory must not pre-exist.

```powershell
wsl.exe -d Ubuntu-22.04 -- bash -lc '
  set -euo pipefail
  root=/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof
  out=/mnt/e/codex-artifacts/teleop_sim_bringup_20260915/saved_pico_diag_20260915
  export PYTHONPATH="$root"
  /root/venvs/teleop23/bin/python -m gear_sonic.scripts.record_g1_true23_saved_teleop_diagnostic \
    --repository-root "$root" \
    --encoder-report "$root/artifacts/g1_true23_frozen_lora/paired_encoder_20260905_v2/original_breadth25/model_25.diagnostic.encoder.json" \
    --decoder-report "$root/artifacts/g1_true23_frozen_lora/paired_encoder_20260905_v2/original_breadth25/model_25.diagnostic.decoder.json" \
    --packets "$root/artifacts/g1_true23_frozen_lora/physical_dance_v1/original_sonic_happy.true23.causal_packets.json" \
    --output-directory "$out"
'
```

This is an offline recorded-input diagnostic. It does not establish live PICO
tracking fidelity or headset transport.

### Verified recorded input

Only one file matching the causal-packet bundle contract was found under the
repo artifacts; none was found under `E:\codex-artifacts`.

| Location | Size | Contents / suitability |
|---|---:|---|
| `Z:\codex\GR00T-WholeBodyControl-sonic-transfer-23dof\artifacts\g1_true23_frozen_lora\physical_dance_v1\original_sonic_happy.true23.causal_packets.json` | 4,359,973 bytes | 535 `robot_independent_reference_packets`, plus 535 semantic packets; control indices 10 through 544; SHA-256 `237910ad5dfc370db9645e52f08ba0ca3b0f409a1383d692e1ce1937c5e3dc9d`. Exact schema consumed by the recommended runner and replay publisher. |
| `E:\codex-artifacts` | n/a | No `*causal_packets*.json` file found. Existing artifact folders were searched. |

The test fixture refers to
`artifacts/g1_true23/pico_saved_clip_replay_v1/upright/causal_packets_neutral_calibrated_v1.json`;
that path is absent. Several larger `pico_*.support.json` and `pico_*.report.json`
files exist, but were not claimed as compatible packet bundles because they do
not match this runner's explicit two-key packet schema.

### Entry points

“Pure Python” means no `gear_sonic_deploy/` executable is required for the
listed path. “Socket peer” describes the other process required only for modes
that use a socket.

| Entry point | Required CLI / mode | Socket peer | C++ deployment? | No-headset status |
|---|---|---|---|---|
| `gear_sonic/scripts/pico_manager_thread_server.py` | No required flags. Usual manager mode: `--manager --port 5556 --zmq_feedback_host localhost --zmq_feedback_port 5557`; other flags include `--buffer_size`, `--num_frames_to_send`, `--target_fps`, `--cuda`, visualizer flags, and `--input-source`. | Publishes binary `command`/`planner`/`pose` protocol on TCP 5556. Manager must receive control-session challenge/claim acknowledgement on TCP 5557 from a native deployment receiver. | Yes for manager mode. Its `ControlSessionClient.claim()` requires the native receiver. | Cannot use recorded packets: manager hard-rejects non-XRT input and initializes XRoboToolkit body tracking. No headset means not runnable as teleop source. |
| `gear_sonic/scripts/replay_g1_true23_pico_packets_zmq.py` | `--packets PATH --output PATH`; optional `--bind tcp://127.0.0.1:5557`, `--subscriber-warmup-s`, `--fault`, `--fault-offset`, `--stale-delay-ms`, `--timestamp-clock local|wsl`, `--repeat-count`, `--stop-file`. | Binds a JSON PUB socket. Peer is a localhost SUB consumer: frozen-LoRA live consumer, clean `zmq` mode, or paced receiver. | No. | Yes. Pure recorded-packet publisher. No inference or physics itself. |
| `gear_sonic/scripts/run_g1_true23_clean_mujoco_teleop.py` | Positional mode plus `--output`; `saved` needs `--packets` and exact `--steps` count. `zmq` needs a publisher at `--endpoint`. `--native23-profile` is valid only for `saved`, `zmq`, and `library-replay`; native23 requires `--disable-safety-fallback`. | None in `saved`; JSON PUB peer at endpoint in `zmq`; default endpoint TCP 5557. | No. | Architecture supports it, but cannot run today: default encoder and decoder paths are absent. Every registry native23 encoder/decoder path is also absent. |
| `gear_sonic/scripts/run_g1_true23_frozen_lora_live_teleop.py` | `--repository-root --decoder-report --encoder-report --output`; optional `--endpoint`, `--steps`, receive/startup age limits, fallback hold, `--viewer`, and `--trace-output`. `--legacy-unpaired-diagnostic` is alternative to `--encoder-report`. | JSON PUB on localhost endpoint, normally replay publisher on TCP 5557. Requires two startup packets then fresh, contiguous packets. | No. | Yes after `pyzmq` installation. Uses present paired ONNX reports and can be paired with replay. This is the shortest path that also exercises the localhost transport boundary. |
| `gear_sonic/scripts/record_g1_true23_saved_teleop_diagnostic.py` | `--repository-root --decoder-report --encoder-report --packets --output-directory`; optional source-orientation, virtual-source, and action-unit diagnostic flags. | None. | No. | Yes after `pyzmq` installation. Recommended direct path. It performs retargeting, inference, MuJoCo, and records physical health values. |
| `gear_sonic/scripts/record_g1_true23_clocked_input_sim.py` | `--repository-root --encoder-report --decoder-report --packets --output-directory --scenario nominal|pause|gap|payload|stale-start|end-of-stream`; optional fault/tail parameters. | None. | No. | Yes after `pyzmq` installation. Adds deterministic clock/fault scheduling. More complex than the recommended nominal runner. |
| `gear_sonic/scripts/run_g1_true23_paced_sim.py` and `run_g1_true23_prepared_paced_sim.py` | Both require `--repository-root --encoder-report --decoder-report --output-directory --endpoint --source-controls`; optional `--tail-controls --first-control-index`. | A JSON PUB sender at `--endpoint`, such as replay publisher. | No. | Potentially usable after `pyzmq`; these are pacing/deadline qualification paths, not shortest bring-up. |
| `gear_sonic/scripts/test_g1_true23_paced_saved_stream.py` | `--repository-root --encoder-report --decoder-report --packets --output-directory --scenario end-of-stream|pause|gap|payload`; optional fault/tail/baseline inputs. | Starts its own local XPUB recorded-reference publisher and its own SUB receiver. | No. | Yes after `pyzmq`; fault-test harness, not nominal session. |
| `gear_sonic/scripts/evaluate_g1_true23_frozen_lora_saved_teleop.py` | `--repository-root --decoder-report --packets --output`. | None. | No. | Not runnable today. Its encoder is hard-coded to absent `artifacts/g1_true23/causal_model_250_20260803/causal_model_250.encoder.onnx`. |
| `gear_sonic/scripts/run_sim_loop.py` | Tyro CLI over `SimLoopConfig`; exact arguments depend on supplied WBC YAML/config. | `init_channel()` and selected WBC config determine local simulator channels. No PICO-packet input or policy is wired by this script. | No direct C++ requirement established from this entry point. | Not a closed recorded-PICO -> policy loop. It only launches the generic G1 simulator. |
| `gear_sonic/scripts/run_g1_23dof_mujoco_sim2sim.py` / `run_g1_23dof_sim_validation.py` | Require at least `--checkpoint --output`; sim2sim accepts optional config/MJCF/ONNX/metadata. | None. | No. | Simulation-only but not PICO-packet teleop. They need a compatible torch promotion checkpoint and torch; no compatible checkpoint was selected in this survey. |
| `gear_sonic/scripts/run_g1_true23_frozen_lora_dance_gantry.py` and `run_g1_true23_frozen_lora_live_gantry.py` | Both require explicit authorization/promotion arguments; live version also needs XRT module/app evidence. | Dance launcher uses replay PUB TCP 5557; live launcher starts the real PICO publisher. | Yes. Both require `gear_sonic_deploy/target/release/g1_true23_active_gantry` and physical authorization. | Out of scope: not simulator-only. The binary is present (4,559,232 bytes), but there is no robot/headset authority. |

### Policy artifacts and current availability

The direct saved diagnostic and frozen-LoRA ZMQ consumer use the following
paired ONNX policy. Their report files assert a shared paired encoder state
hash and pass recorded ONNX Runtime parity checks.

| Artifact | Size | Present | Used by |
|---|---:|---|---|
| `artifacts/g1_true23_frozen_lora/paired_encoder_20260905_v2/original_breadth25/model_25.diagnostic.encoder.onnx` | 13,873,259 bytes | Yes; SHA-256 `3806b2b63ebadf4d6cbf9f79b7072f2bf27ab8eb8bc6a9b3042f97739cc5428a`. | Recommended direct runner; frozen-LoRA live/paced/clocked runners via encoder report. |
| `artifacts/g1_true23_frozen_lora/paired_encoder_20260905_v2/original_breadth25/model_25.diagnostic.decoder.onnx` | 149,567,541 bytes | Yes; SHA-256 `f4416889023eb629656fa189649d8cd071cdc3ae61fc1bfd888d07815d21bdc8`. | Same. |
| `artifacts/g1_true23_frozen_lora/paired_encoder_20260905_v2/original_breadth25/model_25.diagnostic.encoder.json` | 1,571 bytes | Yes. | Required CLI report for paired runners. |
| `artifacts/g1_true23_frozen_lora/paired_encoder_20260905_v2/original_breadth25/model_25.diagnostic.decoder.json` | 1,588 bytes | Yes. | Required CLI report for paired runners. |
| `artifacts/external/unitree_rl_mjlab/deploy/robots/g1/config/policy/velocity/v0/exported/policy.onnx` | 878,421 bytes | Yes. | Fallback policy for supervised/paired controllers. |

Clean-runner defaults are absent:

* `artifacts/g1_true23/causal_model_250_20260803/causal_model_250.encoder.onnx` — missing.
* `artifacts/g1_true23/steptouch_balanced_teacher_lowrank_preserve_alpha010_v1.decoder.onnx` — missing.
* Mode-registry encoder and all four registry decoder paths (`pico_fullbody`,
  `pico_internet_fullbody_walk`, `sonic_hand_crawl`, `sonic_elbow_crawl`) —
  missing. Thus adding `--native23-profile` cannot repair clean `saved` mode.

### Two-process localhost-transport variant

Use this only after the direct saved runner works. It covers the publisher and
subscriber transport boundary in addition to retargeting, policy inference,
MuJoCo, and health readout. Both processes run inside the same WSL monotonic
clock domain, so `--timestamp-clock local` is intentional. Output names must
be new.

```powershell
wsl.exe -d Ubuntu-22.04 -- bash -lc '
  set -euo pipefail
  root=/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof
  out=/mnt/e/codex-artifacts/teleop_sim_bringup_20260915/zmq_saved_pico_20260915
  mkdir -p "$out"
  export PYTHONPATH="$root"
  /root/venvs/teleop23/bin/python -m gear_sonic.scripts.replay_g1_true23_pico_packets_zmq \
    --packets "$root/artifacts/g1_true23_frozen_lora/physical_dance_v1/original_sonic_happy.true23.causal_packets.json" \
    --bind tcp://127.0.0.1:5557 --timestamp-clock local --subscriber-warmup-s 3 \
    --output "$out/publisher.json" &
  pub=$!
  /root/venvs/teleop23/bin/python -m gear_sonic.scripts.run_g1_true23_frozen_lora_live_teleop \
    --repository-root "$root" \
    --encoder-report "$root/artifacts/g1_true23_frozen_lora/paired_encoder_20260905_v2/original_breadth25/model_25.diagnostic.encoder.json" \
    --decoder-report "$root/artifacts/g1_true23_frozen_lora/paired_encoder_20260905_v2/original_breadth25/model_25.diagnostic.decoder.json" \
    --endpoint tcp://127.0.0.1:5557 --steps 535 \
    --output "$out/consumer.json" --trace-output "$out/consumer_trace.npz"
  wait "$pub"
'
```

### Ranked hard blockers today

1. `pyzmq` is not installed in `/root/venvs/teleop23`. It blocks imports of the
   replay loader used by the recommended direct runner and by the clocked/ZMQ
   runners. No proposed run begins until this is installed in the ext4 venv.
2. If a clean-runner command is chosen instead, its required default encoder
   and decoder are missing, and all registered native23 profiles also point to
   missing model bytes. This is independent of the PICO bundle and cannot be
   solved by changing CLI flags.
3. Outputs must go to `/mnt/e/codex-artifacts/...`, not the repo on `Z:`. The
   runners intentionally refuse to overwrite existing report/output paths.
4. `run_sim_loop.py` is not a substitute for this path: it has no recorded
   PICO input or policy wiring. It needs a separately configured WBC peer, and
   this survey did not validate one.
5. Live capture and gantry/C++ launchers remain blocked by the stated absence
   of headset, trackers, XRoboToolkit app connection, physical robot, and
   authorization. They are deliberately not part of the proposed simulation
   procedure.


## Blockers 4 and 5: the model checkpoints were not in this worktree

`gear_sonic/scripts/run_g1_true23_clean_mujoco_teleop.py` loads two hash-pinned
ONNX files by fixed relative path:

| Role | Expected path | Pinned SHA-256 |
|---|---|---|
| Encoder | `artifacts/g1_true23/causal_model_250_20260803/causal_model_250.encoder.onnx` | `733353...56ff2` |
| Decoder | `artifacts/g1_true23/steptouch_balanced_teacher_lowrank_preserve_alpha010_v1.decoder.onnx` | `dc4b6c...1ec9a` |

`artifacts/g1_true23/` **did not exist in this checkout at all**, so every run
died in `sha256_file()` with `FileNotFoundError` before any physics ran.

This repo is a **git worktree** of `Z:\codex\GR00T-WholeBodyControl`
(`.git` contains `gitdir: .../worktrees/...`), and `artifacts/` is untracked, so
the directory was simply never materialised here. Both files exist in the parent
checkout at the identical relative path, and both hash **exactly** to the pinned
constants — verified before use, not assumed:

```
733353148bef1eb8dd83a96416b7a89f0b5c3530ceb9e0cec9c25fdb04f56ff2  causal_model_250.encoder.onnx
dc4b6cf4681eafaff6bb6d70d0aad136e9a3a184337490dc32a511e45b31ec9a  steptouch_balanced_teacher_lowrank_preserve_alpha010_v1.decoder.onnx
```

Note that `causal_model_250.decoder.onnx` in the same parent directory is a
*different* file (`f18139aa...`) and is **not** what the teleop path wants; the
decoder is the steptouch alpha010 one. Reading the run script rather than
matching on filename is what caught this.

*Resolution:* a relative symlink, matching the convention this repo already uses
for `low_latency` and `sonic_release`:

```
artifacts/g1_true23 -> ../../GR00T-WholeBodyControl/artifacts/g1_true23
```

Relative rather than absolute so it resolves from Windows, Git Bash and WSL
alike. Copying was not an option: the directory is 1.3 GB and `Z:` has 447 MB.

## First end-to-end run — the pipeline works, the controller still falls

```
python gear_sonic/scripts/run_g1_true23_clean_mujoco_teleop.py saved   --packets artifacts/g1_true23_frozen_lora/physical_dance_v1/original_sonic_happy.true23.causal_packets.json   --steps 535   --output <artifacts>/saved_smoke.json
```

Exit 0 in 6.4 s wall. Note `--steps` must equal the packet count exactly (535)
or the script refuses to start.

Result: `passed: false`, and that is the **expected** outcome, not a setup fault.

| Field | Value |
|---|---|
| `completed_transition_calls` | 189 |
| `fallback_first_transition` | 72 |
| `fallback_trigger` | `base_tilt` |
| `failure_message` | `fallback physical gate failed: height=0.410189, tilt=0.573767` |
| `terminal_base_height_m` | 0.410189 |
| `saved_packet_bundle_sha256` | `237910ad5dfc370db9645e52f08ba0ca3b0f409a1383d692e1ce1937c5e3dc9d` |
| `live_transport_proven` | false |
| `hardware_authorized` | false |

The distinction that matters: every earlier failure was the environment refusing
to start. This one is the robot falling over in simulation after 189 control
transitions of real saved PICO input — the same frozen-SONIC tracking failure
already documented in `MATCHED_WALK003.md`. Infrastructure is no longer the
blocker; controller quality is.


## Corrections to the codex entry-point survey

The survey above ran concurrently with the environment repairs and several of
its statements were already false by the time it finished. Recording the
corrections rather than leaving the errors in place:

| Survey claim | Actual |
|---|---|
| Clean-runner encoder/decoder defaults are "missing" | Present since the `artifacts/g1_true23` symlink. Both hashes verified against the pinned constants, and the `saved` run completed 189 transitions. |
| `pyzmq` is absent from `/root/venvs/teleop23`, and is the "first runtime blocker" | `pyzmq` 27.2.0 was installed with `gear_sonic[teleop]`. It was never a blocker. |
| MuJoCo 3.5.0 | MuJoCo **3.2.3**, matching the pinned runtime in `MATCHED_WALK003.md`. |
| "`/mnt/e` is mounted in Ubuntu and maps to `E:\codex-artifacts`" | It was **not** mounted at that time — `/mnt/e` was an ordinary ext4 directory. Corrected below. |

Findings from the survey that do hold and are worth keeping:

* `pico_manager_thread_server.py` hard-rejects any non-XRT input source and
  initialises XRoboToolkit body tracking on startup. It therefore cannot be
  driven from recorded packets — with no headset it is simply not runnable, and
  the documented three-terminal tutorial path is unavailable for that reason as
  well as the missing binary below.
* `gear_sonic_deploy/target/release/zmq_manager` **does not exist** (only
  `g1_true23_active_gantry`, 4,559,232 bytes). The tutorial's
  `./deploy.sh --input-type zmq_manager sim` cannot run without a C++ build.
* `record_g1_true23_saved_teleop_diagnostic.py` is a genuine socket-free
  alternative that uses the *paired* `paired_encoder_20260905_v2/original_breadth25`
  ONNX pair, which is present in this checkout and needs no symlink.
* Only one recorded causal-packet bundle exists and it is the one already in use.
  The fixture path `artifacts/g1_true23/pico_saved_clip_replay_v1/...` is absent.

## Blocker 6: `E:` was never mounted inside WSL

`C:`, `H:`, `N:`, `X:`, `Y:` and `Z:` were all mounted under `/mnt`; **`E:` was
not**. `/mnt/e` existed as an ordinary directory on the WSL ext4 root, so every
run that wrote "to `E:\codex-artifacts`" from inside WSL silently wrote into the
WSL root filesystem, invisible to Windows.

This is not limited to today's runs. The stray tree contained earlier output,
including `sonic23_teleop_six_hour_20260910/` and `codex_sonic_runtime/`, so
previous sessions were affected too. Any report citing an `E:\codex-artifacts`
path should be checked against the real drive before it is trusted.

*Resolution:*

```bash
mv /mnt/e /mnt/e_stray          # preserved, nothing deleted
mkdir -p /mnt/e
mount -t drvfs E: /mnt/e
echo "E: /mnt/e drvfs defaults 0 0" >> /etc/fstab   # survives restarts
```

`/mnt/e` now reports 486 G with 89 G free and lists the real `codex-artifacts`
contents. **`/mnt/e_stray` is left in place for triage — it has not been deleted
and may contain the only copy of earlier results.**

## Blocker 7: ZMQ consumer timeout shorter than publisher warm-up

`run_g1_true23_clean_mujoco_teleop.py zmq` kept failing with:

```
TimeoutError: timed out waiting for local PICO causal packet
```

Transport was never the problem. A minimal SUB probe against the same publisher
received **535 of 535** packets, first keys `anchor_joint_pos_il29`,
`causal_history_lower_body`, `control_derivative_contract`, ... .

The real cause is a timing mismatch between two independent settings:

* the consumer builds its controller and loads the 149 MB ONNX decoder *before*
  it connects, which takes several seconds from the 9p-mounted `Z:` drive;
* it then polls with `--receive-timeout-ms`, default **2000 ms**;
* the publisher sends nothing until `--subscriber-warmup-s` elapses.

So a warm-up longer than the consumer's poll timeout guarantees a timeout before
the first packet is ever published. Raising the warm-up — the intuitive fix for a
slow joiner — makes it *worse*.

The two values have to be chosen together: the warm-up must cover the consumer's
model-load time, and the poll timeout must exceed whatever warm-up remains after
the consumer connects.


## Working procedure — sim teleop without a headset

Both processes run in WSL Ubuntu-22.04 against the ext4 virtualenv. Note that
`$VAR` expansion is unreliable through `wsl.exe -- bash -lc '...'` from this
host; several early runs silently wrote their output to `/` because the variable
came through empty. Use literal paths.

### One process, no sockets

```bash
cd /mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof
source /root/venvs/teleop23/bin/activate
python gear_sonic/scripts/run_g1_true23_clean_mujoco_teleop.py saved   --packets artifacts/g1_true23_frozen_lora/physical_dance_v1/original_sonic_happy.true23.causal_packets.json   --steps 535   --output /mnt/e/codex-artifacts/teleop_sim_bringup_20260915/saved_smoke.json
```

`--steps` must equal the bundle's packet count exactly. `--output` must not
already exist — the script opens it `O_EXCL`.

### Two processes, over the real localhost transport

Publisher, then consumer:

```bash
python gear_sonic/scripts/replay_g1_true23_pico_packets_zmq.py   --packets artifacts/g1_true23_frozen_lora/physical_dance_v1/original_sonic_happy.true23.causal_packets.json   --bind tcp://127.0.0.1:5557   --subscriber-warmup-s 12   --output /mnt/e/codex-artifacts/teleop_sim_bringup_20260915/zmq_pub.json &

python gear_sonic/scripts/run_g1_true23_clean_mujoco_teleop.py zmq   --endpoint tcp://127.0.0.1:5557   --steps 535   --receive-timeout-ms 20000   --output /mnt/e/codex-artifacts/teleop_sim_bringup_20260915/zmq_sub.json
```

`--receive-timeout-ms 20000` against `--subscriber-warmup-s 12` is the pairing
that matters; see blocker 7. The defaults (2000 ms against 2 s) are too tight
once the consumer has to load a 149 MB decoder off the 9p-mounted `Z:` drive.

### Results

Publisher: `passed: true`, `published_packet_count: 535` of
`source_packet_count: 535`, `wall_duration_ns: 22801492711`.

Consumer, both transports, using the clean hash-bound `causal_model_250` encoder
and `steptouch alpha010` decoder:

| Run | Transitions | Fallback first / trigger | Terminal gate |
|---|---:|---|---|
| `saved` (offline bundle) | 189 | 72, `base_tilt` | `height=0.410189, tilt=0.573767` |
| `zmq` (live localhost transport) | — | — | `height=0.442445, tilt=0.833834` |

The two differ because the ZMQ run is paced in real time, so a different subset
of packets reaches the controller. The `saved` run is deterministic — repeated
runs reproduce 189 transitions and the same terminal values to every digit.

### What this does and does not establish

Established: the environment, the packet source, retargeting, the hash-bound
ONNX encoder/decoder pair, MuJoCo physics, the safety fallback, the physical
gates and the localhost ZMQ transport boundary all work, and a teleop session
now runs from a recorded PICO source to simulated physics without manual
intervention.

Not established, and unchanged by any of this:

* **Live headset capture.** No PICO has connected. `pico_manager_thread_server.py`
  hard-rejects non-XRT input and initialises XRoboToolkit on startup, so it is
  not runnable at all without hardware. Recorded packets cannot qualify it.
* **Tracking fidelity.** The controller falls in every run. That is the same
  frozen-SONIC failure `MATCHED_WALK003.md` documents, and it is a research
  problem, not a configuration one.
* **Anything about hardware.** No robot was contacted; `dds_opened: false`,
  `robot_commands_published: false`, `hardware_authorized: false` in every report.


## Working sim teleop session — passes

The clean `causal_model_250` + `steptouch alpha010` pair falls in every run. The
**paired `breadth25` pair** in `artifacts/g1_true23_frozen_lora/paired_encoder_20260905_v2/original_breadth25/`
completes the same clip upright. That pair is already in this checkout and needs
no symlink.

### Socket-free

```bash
python gear_sonic/scripts/record_g1_true23_saved_teleop_diagnostic.py   --repository-root /mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof   --encoder-report artifacts/g1_true23_frozen_lora/paired_encoder_20260905_v2/original_breadth25/model_25.diagnostic.encoder.json   --decoder-report artifacts/g1_true23_frozen_lora/paired_encoder_20260905_v2/original_breadth25/model_25.diagnostic.decoder.json   --packets artifacts/g1_true23_frozen_lora/physical_dance_v1/original_sonic_happy.true23.causal_packets.json   --output-directory /mnt/e/codex-artifacts/teleop_sim_bringup_20260915/paired_diag
```

`passed: true`, `successful_controls: 535/535`, `fallback_active: false`,
`minimum_base_height_m: 0.631105`, `maximum_base_tilt_rad: 0.224467`,
`final_simulation_time_s: 10.70`.

### Over the live ZMQ transport — the real teleop shape

Publisher as above with `--subscriber-warmup-s 15`, then:

```bash
python gear_sonic/scripts/run_g1_true23_frozen_lora_live_teleop.py   --repository-root /mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof   --encoder-report artifacts/g1_true23_frozen_lora/paired_encoder_20260905_v2/original_breadth25/model_25.diagnostic.encoder.json   --decoder-report artifacts/g1_true23_frozen_lora/paired_encoder_20260905_v2/original_breadth25/model_25.diagnostic.decoder.json   --endpoint tcp://127.0.0.1:5557 --steps 535   --startup-timeout-ms 30000 --receive-timeout-ms 25000   --output /mnt/e/codex-artifacts/teleop_sim_bringup_20260915/live_sub.json   --trace-output /mnt/e/codex-artifacts/teleop_sim_bringup_20260915/live_trace.npz
```

| Field | Value |
|---|---|
| `passed` | **true** |
| `mode` | `live_pico_causal_zmq_to_cpu_mujoco` |
| `completed_live_transitions` | **535 / 535** attempted |
| source frames | 10 through 544 |
| `maximum_reference_age_ns` | 28,252,976 (28.25 ms) against a 100 ms gate |
| `observed_transport_fault` | none |
| `fallback_policy_query_count` | 0 |
| `fallback_transition_count` | 0 |
| `live_transport_proven` | **true** |
| `gain_profile` | `released_retained` |
| `pico_reference_retargeting` | `pinned_native23_forward_kinematics` |
| `physical_dof` / `decoder_output_dof` | 23 / 23 |
| `deployment_ready` | false |
| `tracking_fidelity_qualified` | false |
| authorization | `dds_opened: false`, `robot_channel_opened: false`, `robot_commands_published: false`, `localhost_only: true`, `simulator_only: true` |

Full-body 23-DoF teleop now runs from a recorded PICO source, across the real
localhost transport, through the frozen paired policy, into MuJoCo physics, for
the entire clip, without the safety fallback ever engaging and with 3.5x margin
on the packet-age gate.

Two things this still does not prove, and the report says so itself:
`live_headset_source_proven: false` — no headset has ever connected — and
`tracking_fidelity_qualified: false`, which remains the open research question
from `MATCHED_WALK003.md`.


## Transport-fault behaviour — degrades, does not die

`MATCHED_WALK003.md` lists as an open item: *"Teleop needs the control loop
decoupled from the input stream ... late headset packets must degrade tracking,
never end a session."* That is now tested in sim. Faults were injected by the
publisher at control offset 120 and declared to the consumer via
`--expected-transport-fault`.

| Fault | `observed_transport_fault` | Live transitions | Fallback transitions | `fallback_stable` | Min height (m) | Max tilt (rad) | Detail |
|---|---|---:|---:|---|---:|---:|---|
| `timeout` | `timeout` | 120 / 120 | 100 | true | 0.6814 | 0.2824 | timed out waiting for a live PICO packet |
| `gap` | `gap` | 120 / 120 | 100 | true | 0.6814 | 0.2824 | lost a contiguous 50-Hz packet |
| `stale` | `stale` | 120 / 120 | 100 | true | 0.6814 | 0.2824 | packet age 232,873,229 ns against the 100 ms gate |

All three report `passed: true`. In each case the consumer consumed 120 live
controls, detected the fault, latched the reviewed zero-velocity balance policy
for 100 further transitions, and held the robot upright to a stable finish —
220 completed transitions total. The session ends deliberately, not by crashing,
and the robot does not fall while the input is gone.

This is simulated input loss on a localhost socket. It is evidence about the
consumer's fault handling, not about headset or Wi-Fi behaviour.

## Note on running these commands

Variable expansion is unreliable through `wsl.exe -d Ubuntu-22.04 -- bash -lc '...'`
from this host: `$O` and `$F` both arrived empty, which silently redirected
output to `/` and passed an empty `--fault` to three runs that then proved
nothing. Either use literal paths or put the commands in a script file and run
the file. `/root/fault_tests.sh` is the fault suite in that form.

## Hardware procedure for the later supervised session

Not executed. No robot was contacted in any of this work. Recorded here so the
supervised session does not have to rediscover it.

Prerequisites that are now satisfied: the teleop virtualenv at
`/root/venvs/teleop23`, the `artifacts/g1_true23` symlink, the `E:` mount, and a
paired policy that completes the clip upright in sim with fault handling proven.

Prerequisites that are **not** satisfied and must be before any robot runs:

1. **A PICO headset has to exist and connect.** Nothing in this session touched
   that. Required: headset plus two controllers plus two ankle trackers, the
   XRoboToolkit PICO apk sideloaded, trackers paired and calibrated, and the PC
   service running — `/opt/apps/roboticsservice/RoboticsServiceProcess` is
   installed in WSL but was not running. Verify with
   `gear_sonic/scripts/probe_xrobotoolkit_live.py`, which must report
   `is_body_data_available` true and a body snapshot with `health_calibrated`
   and `health_is_tracking` set.
2. **The C++ deployment has to be built.**
   `gear_sonic_deploy/target/release/zmq_manager` does not exist; only
   `g1_true23_active_gantry` (4,559,232 bytes) is present. The documented
   three-terminal tutorial path cannot run until it is built.
3. **A reviewed promotion.** `free_standing_authorized: false` in the promotion
   sidecar is not a flag to flip. Only one motion has ever been promoted to
   hardware (`original_sonic_happy.true23.causal_packets.json`, hash-pinned);
   any other motion needs its own bundle and its own review.
4. **A human at the e-stop**, per the existing safety guidance — `O` in the C++
   terminal, or A+B+X+Y on the PICO controllers.

Carry forward from the last hardware session, in section 9 and 11 of
`MATCHED_WALK003.md`: verify standing by posture and motor mode counts rather
than FSM id (knees about 0.285 / 0.329 rad, 23 of 29 motors live, torque
holding), wait about 8 s after re-selecting the `ai` service before sending FSM
commands, and recover from limp with `damp -> FSM 4 -> wait fsm_mode 0 -> FSM 801`.
Full policy authority of 1.0 has never been attempted and 0.6 collapsed once the
robot bore its own weight.

## One command

`artifacts/teleop_sim_bringup_20260915/run_teleop_sim.sh` runs the whole passing
session — publisher, consumer, matched timings, summary — and is installed at
`/root/run_teleop_sim.sh` in WSL:

```
wsl.exe -d Ubuntu-22.04 -- bash /root/run_teleop_sim.sh [output-directory]
```

It checks its four prerequisites before starting and exits non-zero if any are
missing. Verified from a clean shell into a fresh output directory:

```
passed                             True
completed_live_transitions         535
attempted_live_transitions         535
observed_transport_fault           None
fallback_transition_count          0
minimum_base_height_m              0.6630031817734308
maximum_base_tilt_rad              0.38851421947556086
maximum_reference_age_ns           25428622
live_transport_proven              True
live_headset_source_proven         False
tracking_fidelity_qualified        False
deployment_ready                   False
authorization  {'dds_opened': False, 'hardware_authorized': False,
                'localhost_only': True, 'robot_channel_opened': False,
                'robot_commands_published': False, 'simulator_only': True}
```

Packet age varies run to run (25.4 ms here, 28.3 ms earlier) and stays far under
the 100 ms gate.

## Evidence

Under `E:\codex-artifacts	eleop_sim_bringup_20260915\`:

| File | Contents |
|---|---|
| `verify_clean/consumer.json`, `verify_clean/publisher.json` | The clean one-command verification run |
| `verify_clean/measured_trace.npz` | Measured states for that run |
| `live_sub.json`, `live_pub.json`, `live_trace.npz` | First passing live-transport session |
| `paired_diag/report.json`, `paired_diag/measured_trace.npz` | Socket-free paired diagnostic, 535/535 |
| `fault_{timeout,gap,stale}_sub.json` and `_pub.json` | The three transport-fault scenarios |
| `saved_smoke.json` | Clean `causal_model_250` pair falling at transition 189 |
| `zmq_pub.json` | Publisher report, 535/535 published |

Environment changes made, all outside the repo except one symlink:

| Change | Location |
|---|---|
| Teleop virtualenv | `/root/venvs/teleop23` (WSL ext4) |
| `E:` drvfs mount plus `/etc/fstab` entry | `/mnt/e` |
| Preserved stray output from the unmounted period | `/mnt/e_stray` — **not deleted, needs triage** |
| Checkpoint symlink | `artifacts/g1_true23 -> ../../GR00T-WholeBodyControl/artifacts/g1_true23` |
| Session runner | `/root/run_teleop_sim.sh`, copied into this artifact directory |
| Fault suite | `/root/fault_tests.sh` |

## Summary

Teleop runs in simulation on this workstation. Seven environment faults blocked
it; all seven are fixed and documented above. The passing configuration is the
paired `breadth25` policy over the localhost ZMQ transport, 535 of 535 controls,
no fallback, 3.5x margin on the packet-age gate, with timeout, gap and stale
input loss all handled by a stable balance hold.

What remains open is what was open before: no headset has ever connected, so
live capture is unproven, and tracking fidelity is not qualified. Those are a
hardware dependency and a research question respectively, not setup faults.


# Headset-readiness work (2026-09-15, later session)

Goal restated by the operator: get full-body teleop running on the physical
23-DoF G1. The headset is the input device and none is present, so the work
below drives every non-headset prerequisite to a verified state.

## Robot is not connected

Both host Ethernet adapters report `Media disconnected`, and `192.168.123.161`
does not answer. Nothing can reach the G1 from this machine right now. No robot
work was attempted.

## Blocker 8: the XRoboToolkit PC service could not start

`MATCHED_WALK003.md` recorded "the PC service is not running". The reason had
never been established. It is a loader path fault:

```
/opt/apps/roboticsservice/RoboticsServiceProcess: error while loading shared
libraries: libBusiness.so: cannot open shared object file
```

`libBusiness.so` and its siblings sit in `/opt/apps/roboticsservice/` beside the
executable, but that directory is not on the loader path, and the shipped `.deb`
installs no wrapper or unit that sets it.

*Resolution:* a systemd unit that sets `LD_LIBRARY_PATH`, installed and enabled
so it survives restarts:

```
/etc/systemd/system/roboticsservice.service
  WorkingDirectory=/opt/apps/roboticsservice
  Environment=LD_LIBRARY_PATH=/opt/apps/roboticsservice:/opt/apps/roboticsservice/lib
  ExecStart=/opt/apps/roboticsservice/RoboticsServiceProcess
  Restart=on-failure
```

`systemctl is-active roboticsservice` reports **active**, and the service listens
on `*:63901` and `127.0.0.1:60061`.

## The SDK path works end to end, minus the headset

With the service up, `gear_sonic/scripts/probe_xrobotoolkit_live.py` initialises
the SDK, connects to `127.0.0.1:60061`, and returns a complete body snapshot
structure:

```json
{"is_body_data_available": false, "get_body_timestamp_ns": 0,
 "get_body_snapshot": {"available": false, "health_calibrated": false,
  "health_is_tracking": false, "health_tracker_count": 0, "...": null}}
```

Every field false or zero is the **correct** result with no headset paired. The
transport, the service and the Python binding are all functioning; only the
hardware is absent.

**Defect found:** the SDK core-dumps during teardown, after returning results —
`terminate called without an active exception`, and the process dumps core. It
does not affect the returned data, but it would prevent a clean session exit and
should be expected during hardware bring-up.

## PICO reachability is satisfied

A PICO connects to the PC service over Wi-Fi, so the service must be reachable
from the LAN. WSL2 defaults to NAT, which would block this. This machine is
already configured `networkingMode=mirrored` in `C:\Users\camer\.wslconfig`,
and WSL holds a real LAN address on `eth2` (`192.168.1.9/24`, gateway
`192.168.1.1`). The service binds `*:63901`, so a headset on the same Wi-Fi can
reach it.

One caution: that address was `192.168.1.182` earlier in the same session and is
now `192.168.1.9`. It is DHCP-assigned. Reserve it before a teleop session, or
the headset will be pointed at a stale address.

## Status of the remaining hardware prerequisites

| Prerequisite | State |
|---|---|
| Teleop virtualenv, checkpoints, sim loop | Done, verified, passing |
| XRoboToolkit PC service | **Running, enabled at boot** |
| `xrobotoolkit_sdk` binding | **Working, connects, returns snapshots** |
| LAN reachability for the headset | **Satisfied** (mirrored networking) |
| PICO headset, controllers, ankle trackers | **Absent - hard blocker, hardware purchase** |
| `zmq_manager` C++ binary | Under investigation |
| Robot network link | Disconnected (cables out) |
| Reviewed promotion for hardware | Not granted; `free_standing_authorized: false` |
| Tracking fidelity | Fails; best available 0.1738 m hand p95 against 0.055 m feasible |


## Correction: `zmq_manager` is not a binary

An earlier note in this document, taken from the codex survey, said
`gear_sonic_deploy/target/release/zmq_manager` "does not exist" and treated that
as a missing build artifact. That framing is wrong and is corrected here.

`zmq_manager` is a **value of the `--input-type` flag**, not an executable.
`gear_sonic_deploy/deploy.sh` ends in:

```
just run g1_deploy_onnx_ref "$TARGET" "$CHECKPOINT_DECODER" "$MOTION_DATA" \
    --obs-config "$OBS_CONFIG" --encoder-file "$CHECKPOINT_ENCODER" \
    --planner-file "$PLANNER" --input-type "$INPUT_TYPE" \
    --output-type "$OUTPUT_TYPE" --zmq-host "$ZMQ_HOST"
```

No file named `zmq_manager` was ever supposed to exist. The executable the
tutorial actually needs is **`g1_deploy_onnx_ref`**, and *that* is what is
missing: `gear_sonic_deploy/target/release/` contains `g1_fsm_command`,
`g1_mode_probe`, `g1_restore_walkrun`, `g1_true23_active_gantry`,
`g1_true23_live_shadow` and `true23_active_gantry_core_harness`, but no
`g1_deploy_onnx_ref`.

Its CMake target and build tree already exist at
`gear_sonic_deploy/build/src/g1/g1_deploy_onnx_ref/`, and the cached
configuration resolved its dependencies successfully:

```
CMAKE_BUILD_TYPE:STRING=Release
TensorRT_INCLUDE_DIR:PATH=/usr/include/x86_64-linux-gnu
TensorRT_nvinfer_LIBRARY:FILEPATH=/usr/lib/x86_64-linux-gnu/libnvinfer.so
onnxruntime_INCLUDE_DIR:PATH=/opt/onnxruntime/include
onnxruntime_LIBRARY:FILEPATH=/usr/local/lib/libonnxruntime.so
```

All build dependencies are present in WSL: TensorRT 10.13.3, onnxruntime,
cmake/make/g++/ninja, `just`, and `libzmq3-dev` 4.3.4 with the `zmq.hpp` C++
header. The binary appears simply never to have been built in this worktree.

The build is being configured into `/root/deploy_build` on ext4 rather than the
in-repo `build/` directory, because `Z:` has 446 MB free and a C++ build of this
size would risk filling it.


## Blocker 9: `g1_deploy_onnx_ref` had never been built — now built and running

The real executable behind the tutorial is `g1_deploy_onnx_ref`. It was absent
from `gear_sonic_deploy/target/release/`. Every dependency it needs is already
installed in WSL, so it simply had never been compiled in this worktree:
TensorRT 10.13.3, onnxruntime, cmake/ninja/g++ 11.4.0, `just`, `libzmq3-dev`
4.3.4 with `zmq.hpp`, and the vendored `unitree_sdk2`.

Configured into ext4 rather than the in-repo `build/`, because `Z:` has ~440 MB
free:

```bash
cmake -S gear_sonic_deploy -B /root/deploy_build -DCMAKE_BUILD_TYPE=Release -G Ninja
cmake --build /root/deploy_build --target g1_deploy_onnx_ref -j 8
```

Configure reported `ZMQ support enabled for main executable`, the Unitree SDK
found, and `ROS2 disabled` — correct for a G1 without a Thor backpack. The build
linked in 11 steps with no errors, and CMake writes the executable straight into
the repo's `gear_sonic_deploy/target/release/g1_deploy_onnx_ref` (5,517,080
bytes).

## Blocker 10: DDS version symlinks checked out as text files

The freshly built binary would not start:

```
error while loading shared libraries:
.../unitree_sdk2/thirdparty/lib/x86_64/libddsc.so.0: file too short
```

`libddsc.so.0` was a **10-byte plain file containing the text `libddsc.so`**, and
`libddscxx.so.0` a 12-byte file containing `libddscxx.so`. These are stored in
git as symlinks; this worktree is on NTFS, so git checked them out as ordinary
files holding the link target as content.

`LD_LIBRARY_PATH` cannot work around this, because the binary records an
**absolute** `DT_NEEDED` path to those exact files. They have to be real links.

*Resolution:* `/root/fix_dds_symlinks.sh` replaces each stub with a proper
relative symlink. It refuses to touch anything larger than 4 KB or whose content
is not exactly the expected target name, and saves each stub as
`*.textstub.bak` first.

```
repaired: libddsc.so.0   -> libddsc.so
repaired: libddscxx.so.0 -> libddscxx.so
```

`./target/release/g1_deploy_onnx_ref --help` now runs and prints its usage,
which confirms the option the tutorial depends on:

```
--input-type <keyboard|gamepad|gamepad_manager|manager|zmq|zmq_manager>
```

Also visible in that usage, and relevant to 23-DoF teleop tuning:

```
--set-compliance <value>: initial VR 3-point compliance
    (0.01=rigid, 0.5=compliant; default: [0.5, 0.5, 0.0])
    1 value (both hands) or 3 values (left_wrist, right_wrist, head)
```

The default third element is **0.0 for the head**, consistent with the finding
that the head point carries no recoverable signal on a 23-DoF G1.


## The authoritative readiness verdict

`gear_sonic/scripts/true23_pico_readiness.py` is the repository's own read-only
go/no-go check for PICO teleop on a true23 G1. Run with no arguments it reports
`shadow_readiness: NO-GO`, `robot_command_authorized: false`,
`gantry_test_authorized: false`, with eight reasons:

| # | Reason | Can this session fix it? |
|---|---|---|
| 1 | `pico_adb_apk`: ADB executable unavailable | No — needs the headset |
| 2 | `pico_network`: private IPv4 host not configured | No — needs the headset |
| 3 | `robot_network`: private IPv4 host not configured | No — robot unplugged |
| 4 | `trained_checkpoint`: **valid native true23 initialization checkpoint only; genuine retraining required** | No — needs a training run |
| 5 | `simulation_evidence`: simulation report missing | Possibly |
| 6 | `paired_onnx`: encoder, decoder or metadata sidecar missing | Possibly |
| 7 | `shadow_binary`: `g1_true23_shadow_gate` missing | **Fixed — built** |
| 8 | `integrated_live_shadow`: fresh evidence missing | Needs 1-4 |

Reason 4 is the important one. The repository's own gate states independently
that the available checkpoint is an *initialization* checkpoint and that
**genuine retraining is required**. That is the same conclusion the fidelity
ranking reached from measurements (best available 0.1738 m hand p95 against
0.055 m demonstrated feasible), arrived at by a different route.

`g1_true23_shadow_gate` was built the same way as the deploy binary
(3,087,064 bytes) and reason 7 is cleared.

Note that reason 6 is stricter than "some ONNX pair exists". The check calls
`verify_validated_true23_artifact`, which requires an embedded metadata sidecar
with hash bindings and `training_evidence.global_step` — a promotion-grade
export, not any of the twelve diagnostic pairs that were swept.

## What the deploy path needs that is still absent

`deploy.sh` defaults to `policy/release/model_{encoder,decoder}.onnx` and
`planner/target_vel/V2/planner_sonic.onnx`. **None of the three exist** in this
checkout; `gear_sonic_deploy/policy/release/` contains only observation-config
YAML files.

Those are the *released SONIC* assets, and released SONIC is refused on 23-DoF
hardware by `pico_g1_preflight.py` (modes 2 and 5 only, both 29-DoF). So the
documented `deploy.sh --input-type zmq_manager sim` tutorial path is a 29-DoF
workflow. The 23-DoF route is the true23 gantry chain, whose binary
`g1_true23_active_gantry` was already present and whose launcher is
`run_g1_true23_frozen_lora_live_gantry.py`.

That launcher is gated on `--promotion`, `--active-promotion`,
`--gantry-authorize`, `--live-shadow-evidence`, `--authorization-id` and
`--pico-client-apk-sha256`. The last requires the headset APK in hand, and the
promotion arguments require a reviewed promotion that does not exist.


# Retraining: feasible here, and now running

The readiness gate and the fidelity ranking agree that retraining is the
critical path. It turns out this workstation can do it.

## The training environment already existed

`/root/.venvs/g1_true23_mjlab` carries mjlab (sourced from the sibling
`GR00T-WholeBodyControl` checkout), torch 2.9.0+cu128, warp 1.12.0 and
`mujoco_warp`. Verified on the GPU:

```
torch 2.9.0+cu128 cuda True
device: NVIDIA GeForce RTX 3070 Laptop GPU
gpu memory free 7.46 GB of 8.59 GB
matmul ok: True
warp 1.12.0 ; mujoco_warp present
```

## Blocker 11: the training bank rejected a modified source file

`train_g1_true23_compact_tracker_v4.py --mode smoke` failed immediately:

```
ValueError: existing-PICO bank source changed:
  gear_sonic/utils/g1_23dof_xr24_soma_stream.py
```

`validate_bank()` pins a SHA-256 for every source file the bank was built from.
That file carries an **uncommitted** working-tree change: a purely additive
`build_full_body()` method, one cached attribute and one import. The pinned
`build()` path and the three-point payload are untouched — the added method's
own docstring says it "leaves the pinned legacy three-point payload unchanged",
and it is only reached when `include_native_body` is set, which the training
path does not set.

So the bank's data is still substantively correct, but the guard is hash-based
and cannot distinguish an additive extension from a behavioural change. That is
the guard working as intended.

*Action taken:* the working-tree edit was copied aside and the committed version
restored, so the hash matches the bank. **The operator's uncommitted work is
preserved** at
`scratchpad/g1_23dof_xr24_soma_stream.WIP.bak` and must be restored afterwards.
The alternative — regenerating the bank against current sources — would rebuild
the 7,266-frame corpus and change the bank identity, so it was not taken
unilaterally.

## Smoke passes

```json
{"actor_parameters": 180014, "num_envs": 8, "planned_updates": 2}
{"completed": true, "updates": 2, "transitions": 384, "elapsed_s": 24.48,
 "reward_audit_max_abs": {"base": 8.09e-06, "returned": 0.0, "stored": 0.0},
 "deployment_ready": false, "hardware_authorized": false,
 "simulator_qualified": false}
```

Returned and stored rewards are exact; the base reward audit differs by 8.09e-06,
within float32 summation bounds.

## Blocker 12: background jobs do not survive `wsl.exe` session exit

The main run was first launched with `nohup ... &` from
`wsl.exe -d Ubuntu-22.04 -- bash -lc`. It printed a pid and was dead moments
later, having written no log at all. Processes started that way are torn down
when the invoking session ends.

*Resolution:* launch long jobs as a transient systemd unit, which is owned by
the distro's init rather than the session:

```bash
systemd-run --unit=native23-training --collect \
  --working-directory=/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof \
  --setenv=PYTHONPATH=/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof \
  --property=StandardOutput=append:/mnt/e/codex-artifacts/teleop_train_20260915/main_v4.log \
  --property=StandardError=append:/mnt/e/codex-artifacts/teleop_train_20260915/main_v4.log \
  /root/.venvs/g1_true23_mjlab/bin/python \
    gear_sonic/scripts/train_g1_true23_compact_tracker_v4.py --mode main \
    --smoke-report .../smoke_v4/outcome.json --output .../main_v4
```

Check with `systemctl is-active native23-training`; stop with
`systemctl stop native23-training`.

## Run in progress

Main configuration is fixed in the script: **2000 updates x 256 environments x
24 rollout steps = 12,288,000 transitions**, matching the "maximum continuous
12.288m transitions" figure in `VR_READINESS.md`. Actor is 180,014 parameters.

Output: `E:\codex-artifacts\teleop_train_20260915\main_v4\`, log `main_v4.log`.

No deployment claim attaches to this run. The script hard-codes
`deployment_ready=False`, `hardware_authorized=False`,
`simulator_qualified=False` in every record it writes.


# The complete path from here to teleop on the physical 23-DoF robot

Established by reading the gates themselves, not by inference. The chain is
strictly sequential — each stage binds hashes from the previous one, so none can
be reordered or skipped.

```
approved warm start
   -> trained *.promotion.pt          (train_agent_trl.py + ModelSaveCallback, >= 50 updates)
   -> g1_23dof_sim_validation report  (198 episodes / 49,500 steps, IsaacLab)
   -> validated ONNX triplet          (export_g1_23dof_onnx: encoder + decoder + sidecar)
   -> integrated live shadow          (real robot + real headset)
   -> reviewed promotion              (free_standing_authorized, gantry authorization)
   -> supervised hardware session
```

## What blocks each stage today

| Stage | Blocker | Nature |
|---|---|---|
| Trained checkpoint | All 30 checkpoints on disk rejected: 28 lack the safe true23 schema header, 2 are initialization-only. No `*.promotion.pt` exists anywhere. | Needs a training run |
| Trained checkpoint | `train_agent_trl.py` requires **IsaacLab**, which is not installed in any environment on this machine | Large install |
| Simulation evidence | Requires the fixed IsaacLab campaign: scenarios `nominal`/`disturbance_50`/`disturbance_100`, seeds 1729/2718/3141, 22 episodes each, 250 steps — 198 episodes, 49,500 steps, bound to the trained checkpoint's hash | Needs IsaacLab + the checkpoint |
| Simulation evidence | `gear_sonic/config/sim_validation/g1_23dof_rev_1_0.json` sets **`promotion_enabled: false`** (twice). `validate_simulation_report` rejects any report while it is false. | **Governance gate — a human review decision, not a code change** |
| Paired ONNX | Exported from the checkpoint plus the simulation report; binds both | Downstream of the above |
| Live shadow | Real robot and real headset streaming together | Hardware |
| Robot session | Reviewed promotion, `free_standing_authorized`, gantry authorization, operator at e-stop | Human authorization |

## An important correction about the training run started earlier

`train_g1_true23_compact_tracker_v4.py` was launched to attack tracking
fidelity. It is a legitimate research trainer, but it **does not feed the
readiness chain**: it writes `compact_*.pt` research checkpoints and an
`outcome.json`, not the weights-only `*.promotion.pt` with the training-evidence
record that `trained_checkpoint` requires. Improving its metrics would not move
the gate.

It also flatlined early. Between update 50 and update 100 — 307,200 additional
transitions — `root_error_mean_m` moved 0.1265 to 0.1268 and `mean_reward` 0.7751
to 0.7761, with `action_std_mean` unchanged at 0.200 and the learning rate pinned
at 1e-5. Both error measures sit **above** their update-1 values (0.1037 and
0.1250). `VR_READINESS.md` already records this same trial configuration
(`train2000_v2`, `driver_v4`) as previously attempted.

## The `promotion_enabled: false` switch

This deserves to be stated plainly because it is the one blocker that looks like
a one-line fix and is not. Setting it true would make `validate_simulation_report`
accept a report, and would therefore manufacture a promotion without the
campaign evidence the flag exists to gate. That is falsifying qualification for a
1.3-metre humanoid that has already collapsed once under its own weight at 0.6
policy authority. It was not changed, and should only change through whatever
review process the project uses, with real campaign evidence attached.


# The MuJoCo route is approved, and IsaacLab is not required

The operator states this workstation cannot run IsaacLab, and that mjlab/mjbatch
should serve instead. That is supported by the repository itself.

`gear_sonic/config/sim_validation/g1_23dof_mujoco_sim2sim_approval.json`:

```json
{
  "kind": "g1_true23_mujoco_sim2sim_approval",
  "promotion_enabled": true,
  "robot_model": "g1_23dof_rev_1_0",
  "mujoco_version": "3.2.3",
  "runner_sha256":  "be956d4d...445226",
  "runtime_sha256": "0897a9c7...373e16",
  "config_sha256":  "1ffce4af...1a25cf"
}
```

**`promotion_enabled` is `true`** on this path, against the IsaacLab config's
`false`, and the pinned `mujoco_version` 3.2.3 is exactly the MuJoCo already
installed. `gear_sonic/scripts/run_g1_23dof_mujoco_sim2sim.py` contains **zero**
IsaacLab references.

So there are two validation routes, and the MuJoCo one is already approved for
promotion while the IsaacLab one is deliberately closed.

## Blocker 13: `core.autocrlf=true` breaks every hash-pinned gate

The approval hashes did not match the files on disk:

```
runner on disk  bc75df1f...  approval be956d4d...
config on disk  bc4dab24...  approval 1ffce4af...
```

The files were not modified. `git config core.autocrlf` is **`true`**, so git
rewrites LF to CRLF on checkout into this NTFS worktree, while the approval
records LF hashes. `sha256_file()` hashes the bytes on disk, so the comparison
fails on line endings alone.

Normalising confirmed it — LF content hashes to exactly the approved values, and
after converting the two files in place:

```
runner  be956d4da52cdbf70e72abb6fa82d7f9b7dd1d86fb0a9c678f58fc6384445226  MATCH
config  1ffce4afaeb82e20323e24cf4645a98259d9378ba5936e9d9d52e426241a25cf  MATCH
```

This was already being fought one file at a time: `.gitattributes` carries
uncommitted `text eol=lf` entries for `run_pico_robotics_service.sh` and
`g1_23dof_rev_1_0.xml` — which is why the MJCF hash matched earlier while these
did not. Three more entries were added for the files this approval pins.

Any other hash-pinned text file in this repository is subject to the same fault.
A repository-wide `* text=auto eol=lf` with `git add --renormalize .` would fix
it generally, but that rewrites many files and there is uncommitted work in the
tree, so it was not done unilaterally.

## Remaining mismatch, and why it is not a bug to paper over

`gear_sonic/utils/g1_23dof_mujoco_sim2sim.py` still does not match
`runtime_sha256`, and this one is a genuine uncommitted modification, not line
endings. The diff is a physics correctness fix:

```python
# Armature changes invalidate compile-time quantities used by the constraint
# solver (dof_M0/invweight0, body_invweight0 and actuator_acc0). Refresh once
# during model construction, before a controller state exists.
mujoco.mj_setConst(model, mujoco.MjData(model))
```

plus a `"derived_constants": "mj_setConst_after_native_physics_override_v1"`
marker in the report. Without it, overriding armature leaves the CPU evaluator
inconsistent with a model compiled from the same intended parameters.

The approval pins the **older** runtime. So the choice is between validating with
a known-inconsistent evaluator, or refreshing the approval to bind the corrected
runtime. The second is correct, but it is a review action — regenerating an
approval config is exactly the kind of governance step that must not be done
silently. Neither the runtime fix nor the approval file was altered here.


# Why this project has not converged: the reference is infeasible

The operator asked for the prior Codex attempts to be reviewed so their mistakes
are not repeated. Twenty sessions in `~/.codex/sessions` touch this repository.
Reading them changes the diagnosis.

## What the earlier sessions concluded

The 2026-09-13 session states plainly:

> "We need a stronger controller starting point. Current training route has
> stalled. ... We already tried native23 learning, so simply starting another
> PPO run isn't a credible plan."

and records the upstream position: NVIDIA documents SONIC at **29 DoF** with
robot-specific training required for other embodiments, and UFO warns that
checkpoints cannot be reused across differing action dimensions. No ready-made
23-DoF full-body tracker exists to borrow.

**This session repeated that exact mistake.** `train_g1_true23_compact_tracker_v4.py`
was launched in main mode before that history was read. It was stopped at update
400 of 2000. It never beat its own starting point: `root_error_mean_m` 0.1037 at
update 1, 0.1249 at update 400, after 2,457,600 transitions. The prior sessions
had already predicted this outcome.

## The actual root cause

`artifacts/teleop_six_hour_20260910/REFERENCE_FLOOR_AND_SELF_CONTACT_AUDIT.md`
audited all 10,674 frames of the selected references against the compiled
MuJoCo 3.2.3 native23 collision model. For the PICO capture:

| Defect | Value |
|---|---|
| Self-collision frames (original-native PICO) | **498**, maximum penetration 83.320 mm |
| Torso / left shoulder-yaw penetration | 114.36 mm at frame 3842 (69.62 s) |
| Left hip-roll / hand penetration | 105.15 mm at frame 1532 (23.42 s) |
| Fixed-body contacts **arm motion cannot change** | **53 frames**; 48 exceed 1 mm, 35 exceed 5 mm |
| Deepest fixed-body contact | knee/knee **22.657 mm** at frame 1464 |
| Others | pelvis/right hip-roll 15.365 mm, torso/right hip-roll 5.991 mm, torso/right hip-yaw 6.777 mm |
| Status | **"No full collision-aware retarget has been produced"** |

The three TWIST2 walking references have **no** fixed-body contacts. Only the
PICO capture does.

A policy cannot track a reference that requires the robot to pass through
itself. This explains the central observation from this session's sweep: twelve
independently trained checkpoints all landed within 21 percent of each other at
roughly 0.18 m hand p95. They are not twelve different-quality controllers — they
are twelve controllers chasing the same impossible target, and the error floor is
set by the reference, not by the weights. **No amount of additional training
moves that floor.**

It also explains why every repair attempt stalled. The unfixable contacts are in
the **legs** (knee/knee, pelvis/hip, torso/hip), while every attempted repair was
**arm-only**. The best accepted fragment reached 20.19 rad/s and 1192.38 rad/s²
of arm motion and still exceeded the speed envelope by a factor of 1.668 when
rejoining the source pose.

## Mistakes to avoid, taken from the record

1. Do not start another PPO run and expect a different result. Three sessions
   plus this one have now done that.
2. Do not build more audit, evidence or provenance machinery. There is a great
   deal of it already and none of it made the robot track better.
3. Do not attempt arm-only reference repair. The blocking contacts are leg and
   torso geometry.
4. Do not promise a timeline. Several sessions did; none held.

## Clean references obtained

The operator asked for TWIST2 references instead of the flawed PICO capture.
`/z/codex/twist2_inspect` is a `blob:none` partial clone of
`https://github.com/amazon-far/TWIST2.git` at commit `d5c7108`, carrying all ten
`0807_yanjie_walk_*.pkl` clips in its tree with only 001 and 010 materialised.

The three clips the importer pins were restored from origin and verified against
the pinned Git blob hashes in `prepare_g1_true23_twist2_replay.py`:

```
0807_yanjie_walk_002.pkl  71597b5788a2c44b68dccf313b4860cf1020fe9e  MATCH
0807_yanjie_walk_003.pkl  3905de55bf48eee593a521d3b561133665b5c3d5  MATCH
0807_yanjie_walk_008.pkl  a1dd65ae10dc0f4abfc44eb7cf75fea8ed78e634  MATCH
```

These unlock `measure_g1_true23_saved_teleop_tracking.py`, the pinned tracking
qualification that could not run earlier in this session for want of a
`g1_true23_public_twist2_replay_import_v1` source report.


# First pinned tracking measurements, on clean public references

With the TWIST2 clips imported, `measure_g1_true23_saved_teleop_tracking.py` ran
for the first time. Rollouts used `projection_cost_20260906_v1/baseline100/model_100`,
the best hand-error checkpoint from this session's sweep.

| | walk002 | walk003 | walk008 |
|---|---|---|---|
| Balance | pass, 656/656 | **fall**, 595/596, min height 0.294 m | pass, 353/353 |
| Base position p95 | 2.441 m | 7.797 m | 7.749 m |
| Base yaw absolute error p95 | 2.249 rad (129°) | 3.108 rad (178°) | 2.534 rad (145°) |
| Joint RMSE legs12 | 0.2286 rad | 0.2499 | 0.2805 |
| Joint RMSE arms10 | 0.5841 rad | 0.5724 | 0.5110 |
| Hand p95, pelvis-relative | 0.465 / 0.449 m | 0.750 / 0.601 | 0.754 / 0.763 |
| Foot p95, pelvis-relative | 0.338 / 0.349 m | 0.594 / 0.619 | 0.524 / 0.473 |

**Independent corroboration.** PROGRESS.md recorded legacy walk002 at "heading p95
124.974 degrees, left-hand pelvis-relative p95 0.468518 m". This run measured
128.8 degrees and 0.465 m with a different checkpoint. The measurement agrees
with history it did not produce.

## Correction to the earlier root-cause claim

An earlier section of this document attributed the tracking failure to the
infeasible PICO reference. These three clips are clean — no fixed-body contacts —
and tracking still fails badly. Both problems are real and independent:

1. The PICO capture is geometrically infeasible (498 self-collision frames, 53
   frames of unfixable leg contacts).
2. The controller does not track locomotion on clean references either.

## Why the world-frame errors are so large, and what they actually mean

The teleop encoder mode in `gear_sonic_deploy/policy/release/observation_config.yaml`
requires exactly:

```yaml
- name: "teleop"
  mode_id: 1
  required_observations:
    - encoder_mode_4
    - motion_joint_positions_lowerbody_10frame_step5
    - motion_joint_velocities_lowerbody_10frame_step5
    - vr_3point_local_target
    - vr_3point_local_orn_target
    - motion_anchor_orientation
```

There is **no root XY position and no world velocity input**. The three-point VR
targets are pelvis-relative. Locomotion in this architecture is supplied by the
**planner** — `deploy.sh` defaults to `--planner-file planner/target_vel/V2/planner_sonic.onnx`,
and the tutorial describes `zmq_manager` as switching between "planner mode
(locomotion commands via ZMQ)" and a streamed motion mode.

The offline saved-clip evaluation feeds motion only, with no planner. So the
robot has no command telling it where to travel, and a base-position error of
several metres against a walking reference is largely an artifact of evaluating
a locomotion-free path — not proof by itself that the controller is incapable.

The metrics that remain meaningful under that caveat are the pelvis-relative
ones, and they are still poor: hands 0.45-0.76 m p95 against the 0.055 m the
prepared positive control achieves, and arm joint RMSE (0.51-0.58 rad) roughly
2.5 times the leg RMSE (0.23-0.28 rad).

## Blocker 14: the deploy assets were missing from this worktree

`gear_sonic_deploy/planner/` did not exist at all, and `policy/release/` held only
YAML. The released assets live in the sibling checkout:

```
planner/target_vel/V2/planner_sonic.onnx   773,952,989 bytes
policy/release/model_encoder.onnx           50,100,513 bytes
policy/release/model_decoder.onnx           40,900,688 bytes
```

Linked in relatively, as with `artifacts/g1_true23`. Note these are the
**released SONIC 29-DoF** assets; `pico_g1_preflight.py` refuses released SONIC
on a 23-DoF robot (`mode_machine` 4). They are useful for exercising the C++
deployment and ZMQ plumbing in simulation, not as the 23-DoF control path.

## Blocker 15: the TWIST2 importer could not run under the pinned NumPy

`prepare_g1_true23_twist2_replay.py` resolved pickle globals through
`np._core.multiarray`, which exists only in NumPy 2.x. The teleop venv pins
NumPy 1.26.4 to match the recorded runtime, where the module is
`numpy.core.multiarray`, so every import died with
`AttributeError: module 'numpy._core' has no attribute 'multiarray'`.

Fixed by resolving whichever layout the interpreter provides, keeping the
restriction to the same two multiarray callables. This path had evidently never
been exercised in this environment.


# The C++ deploy path was attempted and abandoned, deliberately

With `g1_deploy_onnx_ref` built and the released assets linked, the documented
two-terminal sim stack was brought up.

**What worked.** `run_sim_loop.py` runs headless under systemd with
`--no-enable-onscreen` (it otherwise needs a display; WSLg provides `DISPLAY=:0`
but `systemd-run` does not inherit it). The deployment started, parsed six
observations, loaded thirteen reference motions, constructed
`LocalMotionPlannerBase` with 27 modes, and built TensorRT engines.

**What failed.** Two runs, both dying in TensorRT:

1. First run wrote a 396 MiB engine next to the planner ONNX — which is on `Z:`.
   That drive had ~440 MB free, so the write consumed **every remaining byte**
   and the engine deserialised as corrupt:
   `Serialization assertion plan.header.size == blobSize failed`, then
   `terminate called ... Failed to initialize TensorRT engine`, SIGABRT.
   `Z:` was left at **0 bytes free**. The corrupt 415 MB artifact this run
   created was deleted, restoring 397 MB.
2. Second run, with all assets staged on ext4 at `/root/deploy_assets`,
   regenerated the planner engine in 75.9 s (740 MB weights, 746 MiB peak GPU)
   and then **exited silently without writing the engine or logging an error**.
   The decoder engine wrote fine (41 MB); the planner engine never appeared.

**Why this line was stopped rather than debugged further.** It is the wrong path
for this robot. These are the released SONIC assets, and `pico_g1_preflight.py`
refuses released SONIC on a 23-DoF G1 — it accepts only `mode_machine` 2 and 5,
both 29-DoF, and explicitly rejects mode 4. Even a fully working deployment here
would validate plumbing, not provide a control path for the operator's robot.
Combined with a 774 MB planner ONNX against an 8 GB laptop GPU and a host drive
with under 400 MB free, further effort had poor expected value.

The 23-DoF control path remains the native true23 chain, whose binary
(`g1_true23_active_gantry`) was already present and whose launcher is
`run_g1_true23_frozen_lora_live_gantry.py`.

## What actually runs today, for this robot

`artifacts/teleop_sim_bringup_20260915/run_teleop_sim.sh`, unchanged and still
passing: 23-DoF full body, recorded PICO source over the real localhost ZMQ
transport into MuJoCo, 535 of 535 controls, fallback never engaged, 25-28 ms
packet age against a 100 ms gate, and timeout/gap/stale input loss each handled
by a stable balance hold.

That is the fastest working teleop-shaped thing on this machine, and it predates
the deploy experiment.


## Correction (2026-09-17): `systemd-run` does not keep jobs alive in WSL

Blocker 12 above resolved "background jobs do not survive `wsl.exe` session exit"
by launching them as transient systemd units. **That resolution was wrong.**

A 100-iteration training unit launched that way was stopped 41 seconds in. The
journal shows `Stopping ... Deactivated successfully` with no error, and
`journalctl --list-boots` shows the distro booting repeatedly (18:29, 18:34,
18:41 and 18:46 on 2026-09-17). WSL shuts the distro down shortly after the last
`wsl.exe` session detaches, and that stops every process in it, systemd units
included. `.wslconfig` sets no `vmIdleTimeout`.

The 2026-09-15 training unit that appeared to confirm the systemd approach
survived only because a `tail -F` monitor was holding a `wsl.exe` session open
for the whole run.

The correct approach is to run long jobs in the foreground of a `wsl.exe` call
that stays attached for the full duration. Enabled services (`roboticsservice`)
and the `/mnt/e` fstab mount do return automatically on each boot, so they are
unaffected; ad-hoc units are not.
