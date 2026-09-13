# Private active-row synchronization: bounded parity passes

The v2 adapter restricts authoritative integration overrides to lanes that will actually step. Rejected lanes retain the original deferred-write semantics while public input buffers still follow frozen advance. All shared code and v1 evidence remain unchanged.

Six comparisons pass bit-for-bit on one fixed saved actual3805 fixture: soft and hard unguided→guided with nonzero gains→unguided. Guided search retains all nine dynamics lanes. The saved final backward direction has max|k|=.67394 and max|K|=97.3842. It is used around the same certified seed's nominal states as an explicit parity stress, not represented as reproduction of the old restoration search.

The hard guided case ends with seven rejected lanes and two surviving lanes. Mixed valid/invalid lanes occur in49 inspection records, beginning knot25/substep7. For all331 hard inspection records, full integration vectors and every bound state, control, warmstart, time, warning, actuator force, body pose and external-force array match the frozen original byte-for-byte. Complete returned states/targets/costs, final buffers and feasibility witnesses also match. Re-entering unguided after this mixed-lane search passes.

Evidence lives in `evidence/*.npz`, `report.json`, `source_binding.json`, and `verify_nonzero_guided.py`. Adapter SHA-256: `92f1890184b037a5cb57d2c065ca1b9fdf85a8d314a0488efc77716fc3667ae2`. No failed v2 comparison occurred; every pair was saved before assertions. Original v1 failures remain preserved in their own directory.

Pinned native MuJoCo3.2.3 / NumPy1.26.4, one Batch/BLAS/OpenMP thread. No full solver, policy inference, training or controller trajectory. Single paired timings include identical audit hooks and a shared host: hard unguided before/after649–671ms original versus201ms private; guided699ms versus662ms. These are diagnostic timings, not a new benchmark or real-time claim.

This closes the specifically identified mixed-active-row parity gap on this fixture. It does not qualify arbitrary interventions, heterogeneous per-lane hidden state or all possible Batch fields. Production integration is still absent and requires explicit ownership coordination; no active full-source job was changed.
