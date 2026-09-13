# Independent nominal student review

Reviewed the frozen qualified-walk003 student pilot read-only. No reviewer training, policy inference, physical simulation, or solver call. No shared source changed. NumPy-only saved-array verification passed.

Source binding: `source_and_label_audit.json` verifies every source and input hash in both preserved receipts. Initial receipt SHA256 `2aead25c8c9f4d8d0f020d792350163790bcfcdb4a56d15d2924b964da85af3a`; reviewed v2 receipt `37c1e14971080f49f901ada0d17e9da87bf8c547a7c3186879b4ac1020d6dfea`. Label generation correctly remains bound to the preserved v1 sources; v2 binds those same label bytes.

## Findings and disposition

Initial evaluator had two failure-finalization defects: policy inference exceptions could discard accumulated output, and nonfinite physical diagnostics could make strict JSON writing fail. Frozen v2 catches policy faults before advancing physics, retains precontrol full integration state/history/previous action, saves raw arrays, and encodes nonfinite summary values as null with explicit failure reasons. A missing `simulated_seconds` field would also have failed the root qualification clock contract; v2 adds the actual segment clock interval plus absolute endpoints. These fixes were reviewed statically, not fault-injected by this reviewer.

No remaining material blocker found in the reviewed nominal training/control/phase contracts. A completed zero-parity receipt should additionally bind the zero-head SHA and frozen receipt SHA in a separate immutable sidecar: the v2 parity report itself omits those hashes, although the initialization receipt records the head hash and fitting verifies byte-identical initialization before the first optimizer step. This is provenance bookkeeping, not a discovered numerical parity failure.

## Evidence and control semantics

- All 1,269 label targets, previous actions, histories, sensed states, and 1,270 boundary qpos/qvel states exactly equal the qualified hybrid teacher prefix. Controls are exactly 0–1268 and source frames exactly 11–1279. The teacher's 819 source controls, 350 entry/acquisition controls, and 100 return controls are retained. Final BFM standing is excluded from supervised labels.
- Residual labels exactly equal the actual applied MPC target minus the frozen BFM **unclipped** target on the same measured state/history. Independent normalization reproduces teacher action and previous-action recurrence exactly; large actions are not clipped to ±5.
- Existing BFM history convention is preserved: previous action is control c−1; the four stored action lags in the precontrol history are c−2 through c−5. During student control, next previous action uses the preclip combined BFM plus residual action. Terminal BFM uses the raw actor action. Zero residual therefore retains the frozen prior's action convention.
- Student inputs contain the v4 native goal and original29 task goals, with offsets 0,1,2,4,8,16,24,37. The BFM prior uses the original native goal, position gain 1/yaw gain 2/horizon 8. Distinct references are explicitly disclosed. The terminal helper uses yaw gain 4. History/goal/terminal/quiet helper ASTs are unchanged between v1 and v2.
- Network output is linear times native joint span; there is no tanh or fixed correction-radius cap. Fixed training is seed 773, 1,000 optimizer steps, CPU one thread, and all 1,269 examples are training data. Report correctly makes no held-out generalization claim.
- V2 creates a zero head, performs 100 controls of native zero-head-versus-disabled-head equality before fitting, and requires the fitting initialization ONNX bytes to match that zero head before any gradient step. This same-backend parity does not claim bit-exact equality to historical PyTorch inference.

## Physical and scoring contract

Static review confirms native manual PD at 500 Hz, one target at 50 Hz, no post-initialization state copying, and every substep saved before testing joint range (1e−6 tolerance), native joint velocity, **actual actuator force**, quaternion norm, fall, eight warning counts, and independently accumulated clock. Command torque and actual actuator torque are distinct saved arrays. Policy faults and partial physical controls cannot satisfy full completion.

Full lifecycle is 1,569 controls. Learned control covers 0–1268; native BFM standing starts at 1269. The separate 250-control hold starts only after a fully completed lifecycle and keeps the same physical state, history, and accumulated clock. It cannot replace the original lifecycle's standing tail. Source metrics exclude a partial final control. Original-root and original29 hand/head intent remain separate from v4-relative-foot and joint-reference metrics.

Required root auditor report fields are present in v2: clip, motion-reference receipt, failure, physical maxima, eight integer engine warning counts, and simulated seconds. A dictionary override requires the auditor's explicit local `--motion-override` argument. Original source qualification still needs the independent auditor after an actual completed rollout.

This remains an offline, privileged-state, single-motion pilot. It uses 740 ms of received goal packets and up to 760 ms of raw-pose support. Inference timings from an unpaced shared-host rollout do not establish online deadlines. Passing this code/data review does not establish behavioral success, disturbance robustness, or hardware readiness.
