# Separate worker-spinning experiment

The fixed six-case prepared-FK matrix completed: three scenarios pass and three
fail. Nominal source control counts19/656/47 prove that one good run cannot be
selected as reliable timing. At the first failure, inference takes17.589ms of
a22.854ms control; stage timings isolate more than reference conversion cost.
Legacy/prepared virtual physics remain bit-exact. No evidence is removed.

Inference microbenchmarks alone previously rejected non-spinning sessions
because median/p95 was slower. They did not measure its actual scheduled whole
loop. ORT documents that default session-local spinning consumes extra cycles,
and recommends managing pool contention with multiple sessions:
https://onnxruntime.ai/docs/performance/tune-performance/threading.html
There are separate encoder, decoder and inactive balance pools here. Treat
contention as a testable hypothesis, not the established cause of all outliers.

Disable only INTRA/INTER worker spinning in all three CPU sessions. Keep default
thread count, graph optimizations, hashes, precision, packet validation, source
timing,40ms startup buffer,100ms freshness,20ms deadline, gains and actual physics.
No priority elevation, pinned cores, warmup, slower replay or model changes.

Before timed evaluation, verify all656 captured network inputs against saved
raw23/decoder994 exactly, then run complete656-source+250-balance virtual physics
against the full matching prior measured trajectory, every qpos/qvel/time exact.
Only then execute the SAME fixed six-case matrix in actual_nospin_v1, reporting
ALL runs. No favorable-run selection or automatic parameter sweep. A missed
deadline still latches balance and rejects the scenario. Subsequent independent
audit must recompute deadlines/delivery/bounds/trace parity from saved evidence.

This tests runtime determinism under this finite workstation workload, not a
tracking fix, hard-real-time proof, live-headset qualification or hardware release.

