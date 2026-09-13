"""Verify causal pose outputs against an independently rerun shortened prefix."""
import hashlib
import json
from pathlib import Path
import numpy as np

BASE=Path(__file__).resolve().parent
full=BASE/'intent_retarget_v3/walk003'
prefix=BASE/'intent_retarget_v3_prefix_audit/walk003'
with np.load(full/'kinematic_diagnostics.npz',allow_pickle=False) as a, np.load(prefix/'kinematic_diagnostics.npz',allow_pickle=False) as b:
    count=len(b['qpos'])
    checks={key:bool(np.array_equal(a[key][:count],b[key])) for key in b.files}
with np.load(full/'reference.npz',allow_pickle=False) as a, np.load(prefix/'reference.npz',allow_pickle=False) as b:
    derivative_checks={key:bool(np.array_equal(a[key][:count-1],b[key][:-1])) for key in ('joint_vel','body_lin_vel_w','body_ang_vel_w')}
    endpoint_differences={key:float(np.max(np.abs(a[key][count-1]-b[key][-1]))) for key in derivative_checks}
report=dict(kind='retarget_v3_independent_prefix_causality_check',frames=count,source_clip='walk003',
            includes_previous_v2_speed_violation_frame=True,all_pose_and_diagnostic_arrays_exact=all(checks.values()),
            exact_arrays=checks,derivatives_before_prefix_endpoint_exact=derivative_checks,
            endpoint_derivative_difference=endpoint_differences,
            endpoint_difference_reason='Declared offline central derivatives use a following sample; prefix endpoint uses backward difference.',
            full_reference_sha256=hashlib.sha256((full/'reference.npz').read_bytes()).hexdigest(),
            prefix_reference_sha256=hashlib.sha256((prefix/'reference.npz').read_bytes()).hexdigest(),
            source_file_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            dynamics_qualified=False,hardware_authorized=False)
(BASE/'intent_retarget_v3_prefix_causality.json').write_text(json.dumps(report,indent=2,allow_nan=False))
print(json.dumps(report,indent=2))
assert all(checks.values()) and all(derivative_checks.values())
