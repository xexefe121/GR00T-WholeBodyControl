# Native stepper draft review

Read-only review of `independent_native_stepper_v1/source_draft_v1`. No source was edited, imported, or executed. No native, model, optimizer, or worker process was called. This is source preparation feedback, not native equivalence or timing qualification.

Two gaps should be resolved before selecting native equivalence:

1. **The model identity is an incomplete physics fingerprint.** `native_stepper.py:16` and `:22` enumerate selected geometry/options, then `:65` requires exactly that whitelist. The digest therefore cannot detect changes to omitted physics fields such as `geom_solref`, `geom_solimp`, `opt.disableflags`, or `opt.enableflags`. The current error text calling this a complete pinned geometry/options contract is too strong. Bind the actual pinned MuJoCo version and a complete canonical physics payload, potentially a validated complete MJB serialization or exhaustive array/option manifest. Decide the representation after inspecting that version's API; adding a few more arbitrary fields would leave the same class of gap. Verify the chosen identity at entry and exit of the later equivalence experiment.

2. **Exception evidence loses available integration state and forces.** `native_stepper.py:104` falls back to qpos, qvel, ctrl, time and warnings when a step or capture fails. It omits the full 291-component integration vector, actuator force and selected target. Thus an exception after a mutated native step, or a capture rejection at `:121`, loses evidence that may still be readable. Preserve these values independently on a best-effort basis, with per-field retrieval failures. This must not call another native step, reset the state, or grant returned/captured/verified credit. Existing attempted/returned distinctions should remain intact.

The following source details agree with the reviewed contracts:

- Manual PD expression and native effort clamp at `native_stepper.py:197` match oracle `0027210c`. The adapter does not clamp caller targets.
- The independent repeated binary64 `.002` clock advances once per returned native step at `:204`. It never resynchronizes to observed post-step time. Foundation v3 independently accumulates the same clock and additionally requires exact equality.
- `capture_schema.py:73` uses the original joint-bound, speed, actuator-effort, fall, warning and clock checks, including the original initial-step effort exception and thresholds.
- Full291 packing plus qpos/qvel/force totals 373 float64 elements, or 2,984 bytes. The captured ctrl offset `89:112` is consistent with the declared native291 layout. Sidecar qpos/qvel/time and selected-command comparisons are present.
- Initial restore performs setState, forward, setState once during initialization, verifies full291 identity and preserves warnings; no steady-state restore exists. Whether the first native result matches the original copied-data oracle still requires the later explicitly selected parity run.
- `boundary_snapshot()` supplies measured terms from the exact copied BFM observation helper and discards only its dummy action. The unchanged foundation inserts the actually activated incoming action into history. It retains four previous samples before advancing, matching the frozen history timing.
- Clock, history, mailbox, oracle and BFM observation copies match their declared source hashes exactly.

Suggested bounded tests before native execution:

- Mutate an omitted contact/solver field in a fake model and require the complete model-identity check to fail. Include option flags, topology and contact parameters.
- Inject a native-step exception after modifying full291/actuator force. Require attempted=1, returned=0, no second step or reset, a latched failure, and all readable failed-state fields retained.
- Inject capture failures after a returned step, including getState failure, ctrl/time disagreement and malformed sidecars. Require returned=1, capture_attempts=1, captured=0, no subsequent native call, and separate raw evidence without capture credit.
- Check strict-threshold edges and the independent repeated clock beyond the duration where multiplication diverges from repeated addition. These can use immutable synthetic captures and no model.
- Check restore call order, warnings, no second restore, and boundary-history action timing with injected stub APIs.
- The later native parity experiment should compare all per-step qpos, qvel, PD torque, actuator force, time, warning fields and final full291 to the original oracle. Its counts, fixture and launch need separate selection; no such run was performed here.

The directory currently contains adapter/schema drafts and copied dependencies, with no tests, pinned native fixture manifest or execution runner. Those are unfinished preparation, not a regression. Direct-target evaluation launch remains the immediate priority.
