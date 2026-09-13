"""Preserve the original standing evidence and bind its corrected intent metric."""
import hashlib
import json
from pathlib import Path
import numpy as np

BASE=Path(__file__).resolve().parent/'mjbatch_native323_replay_v1'
old,new=BASE/'standing_executed_target',BASE/'standing_executed_target_v2'
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
old_report,new_report=[json.loads((path/'report.json').read_text()) for path in (old,new)]
for directory,report in ((old,old_report),(new,new_report)):
    assert sha(directory/'trace.npz')==report['trace_sha256']
    assert sha(directory/'provenance.json')==report['provenance_sha256']
assert new_report['metric_revision']==2
with np.load(old/'trace.npz',allow_pickle=False) as a,np.load(new/'trace.npz',allow_pickle=False) as b:
    checks={key:bool(np.array_equal(a[key],b[key])) for key in ('qpos','qvel','target','source_frame','physics_qpos','physics_qvel','physics_torque','physics_substeps')}
assert all(checks.values())
receipt=dict(kind='preserved_original_standing_intent_metric_erratum',original_report_modified=False,
             original_report_sha256=sha(old/'report.json'),original_provenance_sha256=sha(old/'provenance.json'),
             original_trace_sha256=sha(old/'trace.npz'),
             invalid_original_report_fields=['metrics.original_relative_hand_head_p95_m'],
             invalid_original_trace_fields=['original_relative_hand_head_error'],
             reason='Old code compared task_points(actual)[:3] (left foot, right foot, legacy left-hand point) with original29 left hand, right hand, head. It also used a legacy 0.18 m native hand proxy instead of the 0.264 m neutral-source hand convention.',
             original_wrong_p95_m=old_report['metrics']['original_relative_hand_head_p95_m'],
             corrected_directory=str(new),corrected_report_sha256=sha(new/'report.json'),
             corrected_trace_sha256=sha(new/'trace.npz'),corrected_metric_revision=2,
             corrected_p95_m=new_report['metrics']['original_relative_hand_head_p95_m'],
             physical_arrays_bit_identical=checks,physics_results_retain_their_original_scope=True,
             full_source_completed=False,full_body_tracking_qualified=False,hardware_authorized=False)
(old/'metric_provenance_erratum.json').write_text(json.dumps(receipt,indent=2))
(old/'SUPERSEDED_METRIC.md').write_text(
    'The original hand/head intent metric in report.json and trace.npz is invalid and superseded. '
    'Original files remain unchanged. Use ../standing_executed_target_v2/report.json (metric_revision 2) for corrected intent scoring. '
    'All physical pose, velocity, target, and torque arrays are bit-identical; this correction changes scoring only. '
    'See metric_provenance_erratum.json for exact hashes and field names. This was a 1-second standing probe, not full-source qualification.\n')
print(json.dumps(receipt,indent=2))
