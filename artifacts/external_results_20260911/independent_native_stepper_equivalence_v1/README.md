# Native stepper equivalence preparation

This package prepares one independent zero-step model witness, followed by one
recorded-command equivalence run. No model inference, fitting, input stream,
PlantFoundation scheduling, real-time experiment or hardware action is included.
The source-only review does not authorize execution. Each concrete stage requires
a later root selection and exact request/launcher review before its clearance is
created. There are no automatic retries or resumed native runs.

The witness serializes the prepared MuJoCo 3.2.3 model twice into buffers initially
filled with different bytes. Complete equality produces expected_model.mjb. It
performs zero mj_step calls. This is an independent reference for the whole native
model, including contact/solver parameters that a partial array inventory omits.

The replay keeps the reviewed native adapter and strict oracle unchanged. Its
first model runs all 15,690 saved expert main steps followed by all 2,500 hold
steps continuously without reset. A separate canonical model then replays the
failed direct5000 trace through its 3,158th step, reproducing control 315/substep 8
and native_joint_bound exactly. A successful equivalence result therefore still
records that the original direct policy failed. It does not credit unrun source
frames or a hold after that failure.

All 21,348 actual native samples must reproduce saved qpos, qvel, manual-PD
commands, actual actuator forces, repeated simulation clock and warning ledgers
byte for byte. Full integration state (291 float64 values, state spec 8191) is
compared at every available control boundary and both case endpoints. The expert
adapter must verify 18,190 captures; the failing direct adapter must retain 3,158
captures and verify only the preceding 3,157. No independent oracle physics is
called by this package. Model identity is checked twice on entry and twice on
exit for each case, including exception exits when possible. Together with the
witness this allows exactly ten serializations. The counted wrapper preserves
each actual output buffer without asking MuJoCo for another serialization.

## Evidence schema

Each witness/replay request pins all consumed native bundle files, source files,
installed native/NumPy code closure and actual immutable traces/qualifications.
The inventory excludes BFM, ONNX, Torch and training corpora. The established
WSL Python standard library, system libraries and operating system remain the
runtime trust boundary. No download or runtime installation is performed.

The process directory preserves the exact command, request, launch receipt,
final clearance, hidden wrapper/child PIDs, stdout/stderr, raw Python exit,
diagnostic verdict and final wrapper exit. Both prerun_hashes.json and
postrun_hashes.json use `all_exact` plus a `files` map whose entries contain
`expected`, `actual`, and `matched`. A separate actual process-absence snapshot
is required by the owner verifier. Null exit codes remain unknown and cannot
pass. Outputs and start locks are created once; failed attempts stay in place.

Stage report API counters distinguish attempts, native returns and denied calls.
Every serialization record includes its native-return state, captured byte hash
and file in `serialization_buffers/00.mjb` onward. Witness and replay numbering
are separate. Both witness files and all eight replay buffers remain available.

Case captured_trace.npz (or partial_captured_trace.npz on an unexpected failure)
contains:

- packed_capture: uint8[N,2984], interpreted as 373 little-endian float64 values:
  integration[0:291], qpos[291:321], qvel[321:350], force[350:373].
- commanded_torque: float64[N,23]; simulation_time: float64[N].
- warning_counts and warning_lastinfo: int32[N,8].
- control and substep: int64[N]; segment: Unicode[N].
- boundary_name and boundary_integration: recorded names and float64[B,291].

The adapter fault file is the adapter's immutable typed JSON bytes, preserving
full291 and actual force information. A returned invalid capture has a separate
capture-return evidence file. The first parity mismatch retains actual/expected
arrays and its control/substep identity. Unrun cases are explicitly recorded.

The owner completion receipt directly binds request/report/exit/launch/clearance
hashes, both complete input maps, process absence and every output. Its pass is
completion accounting only. A separate saved-evidence auditor independently
decodes all captures and fault bytes; no adapter decode helper is reused there.

## Commands prepared, not executed

`prepare_stage.py --stage witness` creates an immutable request and launcher.
After the actual root selection and final review, `--final-review PATH
--root-selected` creates the separate clearance. The review must directly name
the exact request and launch receipt paths and hashes. Only then may the durable
wrapper run hidden. Replay preparation additionally requires a successful,
completed and stopped witness with all source/input hashes intact.

Synthetic tests exercise replay ordering, partial faults, schema rejection,
missing role pins, exact review subjects, serialization budgets/buffer retention,
exit-model checks after constructor/restore errors and launcher accounting. These
tests neither load a MuJoCo model nor serialize/step one. They do not establish
real native equivalence, asynchronous scheduling, timing or balance.
