"""Record selected concrete input roles without generating task probes."""
import json
from pathlib import Path
B=Path(__file__).resolve().parent;N=B.parent
prior=json.loads((N/'direct_target_fp64_saved_semantics_review_v1/request.json').read_text())['paths']
p={k:prior[k] for k in ('centers','contract','motion','original29')}
p['core']=str(N/'bfm_entry250_actual_oracle_v1/source_snapshot_v1/gear_sonic/utils/g1_true23_mjbatch_ilqr_core.py')
g=N/'velocity_chord_student_v1/generation'
for role,name in {'velocity_features':'features.npy','velocity_target':'teacher_target.npy','velocity_raw':'teacher_feedback_raw.npy',
    'velocity_feedback_clipped':'teacher_feedback_clipped.npy','velocity_native_clipped':'teacher_native_clipped.npy','velocity_value':'perturbed_joint_velocity.npy'}.items():p[role]=str(g/name)
p.update(generation_report=str(g/'report.json'),generation_review=str(N/'velocity_chord_completed_data_review_v1/review.json'),
    generation_audit=str(N/'velocity_chords_independent_v1/report.json'))
(B/'input_config.json').write_text(json.dumps({'paths':p},indent=2)+'\n')
