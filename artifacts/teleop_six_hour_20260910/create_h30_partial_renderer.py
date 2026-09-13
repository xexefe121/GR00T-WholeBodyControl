"""Create a separate frozen renderer; never modify an in-use shared renderer."""
from pathlib import Path

base = Path(__file__).resolve().parent
source = (base / 'render_mpc_failed_replay.py').read_text()
changes = (
    ('"""Fixed-world saved MPC replay; preserve the final partial 2ms physics steps."""',
     '"""Fixed-world H30 clipped-feedback partial success; 7s initialization plus 3s source."""'),
    ('from gear_sonic.scripts.evaluate_g1_true23_bfmzero import load_motion',
     '''def load_motion(name):
    report_path=Path('C:/Users/camer/sonic23_sim_artifacts/internet_pico_20260909_v1/released_core_comparison_v1/normal')/name/'report.json'
    report=json.loads(report_path.read_text())
    timeline=report['timeline']
    value=timeline['timeline_path']
    path=Path('C:/'+value[7:]) if value.startswith('/mnt/c/') else Path(value)
    with np.load(path,allow_pickle=False) as z:
        motion={key:z[key].copy() for key in z.files}
    return motion,timeline,path'''),
    ("CASE=ROOT/'artifacts/teleop_six_hour_20260910/mjbatch_native323_replay_v1/walk002_source3_feedback_v2'",
     "CASE=ROOT/'artifacts/teleop_six_hour_20260910/bfm_online_intent_v2/mpc_h30_323_clip1_v1'"),
    ("output=CASE/'visual_comparison_v1'",
     "output=Path('C:/Users/camer/.codex/visualizations/2026/09/10/01a08b79-d89d-7d13-bcf8-33de331c0024/mpc_h30_clipped_partial_v1')"),
    ("assert report['metric_revision']==2 and report['mode']=='feedback' and report['failure']",
     "assert report['metric_revision']==2 and report['mode']=='feedback' and report['failure'] is None\n    assert report['feedback_correction_clip_rad']==.1 and not report['full_source_completed']"),
    ("assert np.all(trace['physics_substeps'][:-1]==10) and trace['physics_substeps'][-1]==7",
     "assert np.all(trace['physics_substeps']==10) and len(trace['physics_substeps'])==500"),
    ("assert source_start_s==7. and abs(times[-1]-source_start_s-1.174)<1e-12",
     "assert source_start_s==7. and abs(times[-1]-source_start_s-3.)<1e-12"),
    ("'RECORDED MPC FEEDBACK REPLAY'", "'H30 MPC: CLIPPED FEEDBACK REPLAY'"),
    ("'Physical MuJoCo 3.2.3 states; offline plan'", "'MuJoCo 3.2.3; offline plan; correction +/-0.1 rad'"),
    ("f'PARTIAL RUN: limit failure at 8.174 s | root error {drift:.3f} m | leg RMSE {leg:.3f} rad'",
     "f'PARTIAL PROBE: first 3.000 s source only | root error {drift:.3f} m | leg RMSE {leg:.3f} rad'"),
    ("ending='FINAL CONTROL: 7 / 10 substeps; desired active sample at 8.180 s' if i==len(actual)-1 else 'Full 7.000 s initialization preserved | same fixed world camera | 0.5 m grid'",
     "ending='PROBE END: no failure here; remaining 10.340 s source and recovery not tested' if i==len(actual)-1 else 'Full 7.000 s initialization preserved | same fixed world camera | 0.5 m grid'"),
    ('contact=[0,250,350,375,400,len(actual)-1]', 'contact=[0,250,350,400,450,len(actual)-1]'),
    ("video=output/'full_initialization_and_failed_source.fixed_world.mp4'",
     "video=output/'full_initialization_and_partial_source.fixed_world.mp4'"),
    ("kept[contact[-1]].save(output/'failure_boundary.fixed_world.png')",
     "kept[contact[-1]].save(output/'partial_probe_end.fixed_world.png')"),
    ("kind='native323_failed_mpc_feedback_replay_vs_original_native_requested_reference'",
     "kind='native323_H30_clipped_feedback_partial_replay_vs_original_native_requested_reference'"),
    ('final_control_substeps=7,', 'final_control_substeps=10,'),
    ("final_desired_pose_convention='active 50Hz command sample419 at nominal8.180s; actual terminal boundary8.174s'",
     "final_desired_pose_convention='active 50Hz command sample510 at nominal10.000s; actual terminal boundary10.000s'"),
    ('physics_reexecuted=False,partial_failure_visible=True,hardware_authorized=False,visual_pass_claim=False,',
     'physics_reexecuted=False,partial_scope_visible=True,hardware_authorized=False,visual_pass_claim=False,\n                 feedback_correction_clip_rad=.1,full_source_completed=False,received_stream_controller=False,'),
)
for old, new in changes:
    assert source.count(old) == 1, old
    source = source.replace(old, new)
destination = base / 'render_mpc_h30_clipped_partial.py'
assert not destination.exists()
destination.write_text(source)
print(destination)
