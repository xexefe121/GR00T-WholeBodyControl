"""Bounded saved-result comparison, no model or physics operations."""
import argparse
import json
from pathlib import Path
from qualified_inputs import load_qualified,sha,exact

parser=argparse.ArgumentParser()
parser.add_argument('--qualification',type=Path,required=True)
parser.add_argument('--qualification-sha256',required=True)
args=parser.parse_args()
q,pins,full,hold=load_qualified(args.qualification,args.qualification_sha256)
out=Path(__file__).parent
base=out.parent
old_path=base/'walk002_full_control_lm_independent_intent_v1/report.json'
old=json.loads(old_path.read_text())
main=json.loads(Path(q['independent_reports']['full_intent']['path']).read_text())
extra=json.loads(Path(q['independent_reports']['hold_intent']['path']).read_text())
assert old['full_lifecycle_source_intent_pass'] and not old['requested_segment_quiet_pass']
assert old['source_metrics']==main['source_metrics']
assert main['requested_segment_quiet_pass'] and extra['requested_segment_quiet_pass']
result=dict(kind='qualified_walk002_same_MPC_vs_recorded_prefix_terminal_BFM',
    qualification_sha256=args.qualification_sha256,original_same_MPC_quiet_failure_preserved=True,
    original_source667_metrics_exactly_unchanged=True,source_metrics=main['source_metrics'],
    original_MPC_terminal=old['quiet_last_three_seconds'],hybrid_original_terminal=main['quiet_last_three_seconds'],
    hybrid_separate_hold=extra['quiet_last_three_seconds'],
    source_controller='qualified original recorded MPC prefix0..1116; no replanning in hybrid',
    terminal_controller='unchanged BFM horizon8/position1/yaw4, actual inference at original terminal start1117',
    result='one qualified offline hybrid lifecycle plus separate quiet hold',new_dynamics=False,
    timing_qualified=False,live_teleoperation_qualified=False,hardware_authorized=False,
    inputs=dict(pins,**{str(old_path):sha(old_path)}))
(out/'result_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
metrics=[('Root XY p95 (m)','root_xy_p95_m'),('Original yaw p95 (deg)','original_yaw_p95_deg'),
    ('Root speed p95 (m/s)','root_linear_speed_p95_mps'),('Joint speed p95 (rad/s)','joint_speed_p95_radps'),
    ('Joint speed max (rad/s)','joint_speed_max_radps'),('Tilt max (rad)','tilt_max_rad')]
rows=['| Quiet metric | Original MPC terminal | Hybrid terminal | Separate hold |','|---|---:|---:|---:|']
for label,key in metrics:rows.append('| '+label+' | '+' | '.join(f"{d[key]:.6f}" for d in (old['quiet_last_three_seconds'],main['quiet_last_three_seconds'],extra['quiet_last_three_seconds']))+' |')
text='''# Qualified walk002 offline hybrid

All667 source metrics exactly match the qualified original MPC run. That original run failed terminal quiet speed gates. Switching to the known BFM terminal controller at original control1117 passes both the original300-control terminal window and separate250-control hold. Root independently reproduced all16,670 native samples exactly.

'''+ '\n'.join(rows)+'''

Video uses recorded physical states only: declared native23 reference on the left, recorded physics on the right, one fixed world camera. The full33.34 seconds includes original28.34-second lifecycle and separate5 seconds. Exact source-end20.34s, controller-handoff22.34s, lifecycle-end28.34s and final33.34s frames are included. Visual sampling is normally100ms with shorter boundary intervals; timestamps stay on original2ms grid. Final displayed endpoint lasts an explicit2ms.

This is an offline saved-MPC-prefix plus actual-terminal-BFM hybrid. Recorded-command replay recomputes native physics from saved targets without model inference. Neither artifact establishes live received-Pico control, realtime MPC or hardware readiness. Original MPC quiet failure remains archived.
'''
(out/'RESULT_COMPARISON.md').write_text(text)
print(json.dumps(dict(comparison_sha256=sha(out/'result_comparison.json'),qualification_sha256=args.qualification_sha256,source_metrics_exact=True)))
