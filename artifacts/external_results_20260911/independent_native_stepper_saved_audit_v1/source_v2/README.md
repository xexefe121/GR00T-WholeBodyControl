Prepared saved-evidence audit; no real native-equivalence result has been audited yet.

`saved_math.py` independently decodes the fixed 373 little-endian float64 payload and the bounded typed-JSON fault evidence. It imports no native adapter, capture decoder, model runtime or optimizer. The audit checks all 18,190 expert samples including the continuous hold and all 3,158 direct samples, including the expected final rejection. Direct verified steps must remain 3,157; captured and returned steps must remain 3,158.

Checks cover q/dq, selected PD torque, actual actuator force, repeated 2ms clock, warnings/lastinfo, every recorded complete291 control boundary and segment endpoint, boundary-to-previous-capture continuity, and the final typed fault/returned-capture evidence. All ten actual MJB serialization buffers must equal the independent witness bytes. The eight replay buffers are the two different-fill entry and exit serializations for each of two separately constructed models; the expert main and hold share one adapter without restoration.

Before an actual audit, provide a JSON object with `witness` and `replay` entries, each containing actual paths for `request`, `clearance`, `report`, `owner`, `process_exit`, `launch_receipt`, `process_absence`, `pre_hashes`, `post_hashes` and a `completion_evidence` list. `prepare_request.py --stages ACTUAL_CONFIG --output NEW_REQUEST` hashes these existing completed artifacts and the exact selected traces. It refuses missing or incomplete stages. It does not launch the native runner or run the audit.

Owner receipts must expose literal request/report/clearance/launch/exit hashes, a passed accounting verdict, current input/output hashes, and the process absence receipt hash. Pre/post ledgers use `all_exact` and a `files` mapping of `{expected, actual, matched}` entries. The auditor independently validates the ledgers and all current hashes. Concrete paths and this schema must be aligned with the final owner source before a real audit request is frozen.

Run `audit_saved.py --request ACTUAL_REQUEST --output NEW_OUTPUT` only after root/owner completion and source clearance. Output creation is exclusive. Failures retain checks, current context, tracebacks and all hashes already consumed. There is no automatic retry, model inference, native step, optimization or replay.

An adapter-equivalence pass preserves the original direct controller's bound failure. It does not qualify balance, independent plant scheduling, real-time performance, or hardware use.
