"""Compare fixed control-space LM to archived failing state-space damping; no solve."""
import json
from pathlib import Path
import hashlib
import numpy as np

P = Path('/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/pico_final3800_factored_riccati_condition_v1')
F = P / 'factored_final_grouped_mu1e+06_first_failure.npz'
a = np.load(F)
B, V, R, mu = a['B'], a['next_value'], a['luu'], float(a['mu'])
Q0 = R + B.T @ V @ B
Qstate = a['regularized_quu']
Qcontrol = Q0 + np.eye(23)  # One fixed standard LM damping, lambda=1.
eps = np.finfo(float).eps

def summary(Q):
    sym = (Q + Q.T) / 2
    e = np.linalg.eigvalsh(sym)
    return dict(raw_lower_eigen_min=float(np.linalg.eigvalsh(Q).min()),
                symmetric_eigen_min=float(e.min()), symmetric_eigen_max=float(e.max()),
                spectral_norm=float(np.linalg.norm(Q,2)), symmetry_max=float(np.abs(Q-Q.T).max()),
                epsilon_times_norm=float(eps*np.linalg.norm(Q,2)))

v_eig = np.linalg.eigvalsh(V)
b_norm = np.linalg.norm(B,2)
r_min = np.linalg.eigvalsh(R).min()
report = dict(kind='saved_matrix_fixed_control_space_LM_assessment', solver_calls=0,
              actual_controls_executed=0, knot=int(a['knot']), archived_mu=mu,
              native_numpy=np.__version__, state_value_min_eigen=float(v_eig.min()),
              state_value_max_eigen=float(v_eig.max()), B_norm=float(b_norm), R_min=float(r_min),
              no_damping=summary(Q0), archived_state_damping=summary(Qstate),
              fixed_control_lambda_one=summary(Qcontrol),
              exact_PSD_value_bound='lambda_min(R+B^T V B+lambda I) >= lambda_min(R)+lambda when V is PSD',
              conservative_bound_allowing_observed_negative_V=float(r_min+1+min(v_eig.min(),0)*b_norm*b_norm),
              no_eigenvalue_clipping=True, no_new_diagonal_applied_to_controller=True,
              source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              input_sha256=hashlib.sha256(F.read_bytes()).hexdigest())
(P/'control_lm_assessment.json').write_text(json.dumps(report,indent=2,allow_nan=False))
(P/'control_lm_assessment_source.py').write_bytes(Path(__file__).read_bytes())
np.savez_compressed(P/'control_lm_matrices.npz', Q0=Q0,Qstate=Qstate,Qcontrol=Qcontrol,
                    original_value=V,B=B,R=R,control_lambda=1.)
print(json.dumps(report),flush=True)
