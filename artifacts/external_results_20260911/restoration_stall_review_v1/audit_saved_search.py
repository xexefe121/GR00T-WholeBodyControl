"""Read-only algebra over archived candidates; no solver or physics imports."""
from pathlib import Path
import hashlib
import json
import numpy as np

OUT = Path(__file__).resolve().parent
RUN = OUT.parent / 'restoration_stall_3805_v1'
REPO = Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof')
CONTRACT = REPO / 'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1/contract.json'
report = json.loads((RUN/'report.json').read_text())
contract = json.loads(CONTRACT.read_text())
limits = np.asarray(contract['joint_limits'])
width = limits[:,1]-limits[:,0]
with np.load(RUN/'all_generated.npz',allow_pickle=False) as a:
    searches, seed, K = a['targets_by_search'],a['original_seed'],a['final_K']
    assert np.array_equal(a['final_targets'],seed)
    assert np.array_equal(searches[0],np.repeat(seed[:,None,:],9,axis=1))
    assert np.all(searches >= limits[:,0]) and np.all(searches <= limits[:,1])
    assert np.isfinite(searches).all()
    rows = []
    for index in range(1,8):
        row = dict(mu=report['backward_calls'][index-1]['mu'],lanes=[])
        for lane in range(9):
            alpha = .5**lane
            change = np.abs(searches[index,:,lane]-seed)
            # boxQP constrains k within lo-seed,hi-seed, hence |k|<=joint width.
            # Clipping is non-expansive relative to already-bounded seed.
            # Therefore |K dx| must be at least max(0,|applied delta|-alpha*width).
            lower = np.maximum(change-alpha*width,0)
            first = float(change[0].max())
            first_feedforward_bound = float((alpha*width).max())
            assert first <= first_feedforward_bound+1e-12
            row['lanes'].append(dict(alpha=alpha,merit=report['alpha_searches'][index]['costs'][lane],
                                     max_target_departure_rad=float(change.max()),
                                     first_knot_max_departure_rad=first,
                                     global_boxQP_feedforward_bound_rad=first_feedforward_bound,
                                     minimum_feedback_magnitude_rad=float(lower.max()),
                                     departure_peak_knot_joint=list(map(int,np.unravel_index(np.argmax(change),change.shape)))))
        rows.append(row)
    k_max=float(np.abs(K).max())
    quaternion_norm_max_error=float(np.max(np.abs(np.linalg.norm(a['final_states'][:,3:7],axis=1)-1.)))
certs=report['all_candidate_certificates']
assert len(certs)==64 and not any(c['hard_nominal_pass'] for c in certs)
assert not report['feasible_generated_candidates']
sources = [RUN/'all_generated.npz',RUN/'report.json',CONTRACT,
           RUN/'g1_true23_mjbatch_ilqr_core.py',RUN/'g1_true23_mjbatch_mpc.py',
           RUN/'g1_true23_mjbatch_restoration.py',
           Path('E:/codex_sonic_runtime/mjbatch323_20260910/venv/lib/python3.11/site-packages/mjbatch/__init__.py'),
           Path('E:/codex_sonic_runtime/mjbatch323_20260910/venv/lib/python3.11/site-packages/mjbatch/csrc/batch.h'),
           Path(__file__)]
inventory=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sources]
result=dict(kind='saved_restoration_feedback_amplification_bound',
            calculation='max_j,t max(0,abs(saved_candidate-seed)-alpha*(native_upper-native_lower))',
            initial_target_sequences_all_nine_bitexact=True,all_targets_finite_and_native_bounded=True,
            initial_and_all63_candidates_hard_infeasible=True,
            final_targets_equal_initial=True,final_returned_K_max_abs=k_max,
            final_returned_K_is_last_rejected_sweep_not_executed=True,
            nominal_quaternion_norm_max_error=quaternion_norm_max_error,
            rows=rows,provenance=inventory,
            no_dynamics_derivatives_computed=True,no_solver_or_physics_run=True)
(OUT/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False))
print(json.dumps(dict(final_K_max=k_max,smallest_alpha=[dict(mu=r['mu'],**r['lanes'][-1]) for r in rows]),indent=2))
