"""Independent fixed incoming-matrix review; no backward pass or model imports."""
from pathlib import Path
import json,hashlib,sys,os
import numpy as np
HERE=Path(__file__).resolve().parent
SRC=HERE.parent/'pico_final3800_factored_riccati_condition_v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
paths=sorted(SRC.glob('factored_final_grouped_mu1e+06_knot*.npz'))+[SRC/'factored_final_grouped_mu1e+06_first_failure.npz']
rows=[];arrays={};eps=np.finfo(np.float64).eps
for path in paths:
    z=np.load(path);q0=z['quu'] if 'quu' in z.files else z['luu']+z['B'].T@z['next_value']@z['B']
    knot=int(z['knot']) if 'knot' in z.files else int(path.stem.rsplit('knot',1)[1])
    state_reg=z['regularized_quu'] if 'regularized_quu' in z.files else z['luu']+z['B'].T@(z['next_value']+1e6*np.eye(58))@z['B']
    q=.5*(q0+q0.T);rhs=z['B'].T@z['next_value']@z['A']
    reg=q+np.eye(23) # Exactly the proposed initial fixed control-space mu=1.
    vals=np.linalg.eigvalsh(reg);basevals=np.linalg.eigvalsh(q)
    scale=np.linalg.norm(reg,np.inf)
    row=dict(knot=knot,source_sha256=sha(path),source_name=path.name,
        fixed_control_mu=1.,symmetric_unregularized_min_eigen=float(basevals[0]),
        control_regularized_min_eigen=float(vals[0]),control_regularized_max_eigen=float(vals[-1]),
        estimated_spectral_condition=float(vals[-1]/vals[0]),epsilon_times_condition=float(eps*vals[-1]/vals[0]),
        unregularized_symmetry_max=float(np.max(np.abs(q0-q0.T))),
        control_diagonal_addition_min_max=[float(np.min(np.diag(reg)-np.diag(q))),float(np.max(np.diag(reg)-np.diag(q)))],
        archived_state_regularized_norm_inf=float(np.linalg.norm(state_reg,np.inf)),
        control_regularized_norm_inf=float(scale),unregularized_cross_norm_inf=float(np.linalg.norm(rhs,np.inf)))
    try:
        chol=np.linalg.cholesky(reg)
        # All-control linear algebra diagnostic only, not boxqp or backward solve.
        sol=np.linalg.solve(reg,-rhs)
        resid=reg@sol+rhs
        denom=scale*np.linalg.norm(sol,np.inf)+np.linalg.norm(rhs,np.inf)
        row.update(cholesky_pass=True,linear_system_solution_finite=bool(np.isfinite(sol).all()),
            normalized_solve_residual_inf=float(np.linalg.norm(resid,np.inf)/denom),
            normalized_cholesky_reconstruction_inf=float(np.linalg.norm(chol@chol.T-reg,np.inf)/scale),
            all_control_cross_solution_max_abs=float(np.max(np.abs(sol))))
        arrays[f'knot{knot:02d}_Qsym']=q
        arrays[f'knot{knot:02d}_Qreg']=reg
        arrays[f'knot{knot:02d}_Qux']=rhs
        arrays[f'knot{knot:02d}_all_control_linear_solution']=sol
    except np.linalg.LinAlgError as e:row.update(cholesky_pass=False,error=str(e))
    rows.append(row)
rows.sort(key=lambda x:x['knot'])
np.savez_compressed(HERE/'matrix_checks.npz',**arrays)
r=dict(kind='independent_control_space_LM_fixed_incoming_matrices',native_numpy=np.__version__,python=sys.version,
    environment={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']},
    matrix_files=len(rows),all_fixed_mu1_cholesky_pass=all(r['cholesky_pass'] for r in rows),
    maximum_condition_estimate=max(r['estimated_spectral_condition'] for r in rows),
    maximum_normalized_residual=max(r.get('normalized_solve_residual_inf',float('inf')) for r in rows),rows=rows,
    formulation=dict(Qsym='0.5*(R+B.T@V@B + (R+B.T@V@B).T)',Qreg='Qsym + mu*I_control',Qux='B.T@V@A; no mu term',
        proposed_mu_initial=1.,mu_rejection_multiplier=10.,mu_acceptance_divisor=10.,mu_min=1e-6,mu_stop_above=1e6,
        merit='unchanged original restoration/tracking merit and box bounds',value='factored update with original R/lxx/V, not Qreg; do not add damping to objective',
        validation='finite symmetric Qreg, numerical SPD/Cholesky, finite solved directions; unchanged final nominal and full-native certificates'),
    conclusions=['On all seventeen archived incoming value matrices, fixed control mu1 produces numerical SPD and small backward residual. These checks do not demand a scale-aware change before the single authorized experiment.',
        'Condition remains high; a small residual certifies backward stability, not forward accuracy. Preserve condition/scale diagnostics without silently relaxing the acceptance gates.',
        'These incoming V matrices belong to prior state-damped sweep. A new control-damped backward sweep changes V recursively; this report does not establish that full sweep succeeds.',
        'Scalar control damping changes the search direction and is not algebraically equivalent to state damping. Keep opt-in private and preserve default path byte exactly.'],
    no_eigenvalue_clipping=True,derivative_calls=0,backward_passes=0,boxqp_calls=0,optimizer_calls=0,physics_calls=0,policy_calls=0)
(HERE/'report.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(all_mu1_cholesky_pass=r['all_fixed_mu1_cholesky_pass'],matrices=len(rows),maximum_condition=r['maximum_condition_estimate'],maximum_residual=r['maximum_normalized_residual'],failure_knot=rows[0]),indent=2),flush=True)
