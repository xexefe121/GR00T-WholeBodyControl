"""Derive a standalone renderer from the previously timestamp-verified adapter."""
from pathlib import Path
import hashlib
import json

HERE=Path(__file__).resolve().parent
base=HERE/'render_mpc_h30_clipped_partial.py'
text=base.read_text()
text=text.replace('"""Fixed-world H30 clipped-feedback partial success; 7s initialization plus 3s source."""',
    '"""Fixed-world PICO checkpoint; full7s entry+3s source, no full-source promotion."""')
start=text.index('def load_motion(name):')
stop=text.index('from gear_sonic.utils.g1_true23_generalist_benchmark',start)
text=text[:start]+'from gear_sonic.scripts.evaluate_g1_true23_bfmzero import load_case_motion\n'+text[stop:]
text=text.replace("CASE=ROOT/'artifacts/teleop_six_hour_20260910/bfm_online_intent_v2/mpc_h30_323_clip1_v1'", """ARCHIVE=Path('E:/codex-artifacts/sonic23_teleop_six_hour_20260910')
CASE=ARCHIVE/'mjbatch_full_v1/pico_v4_native323_relativefoot_full_v1'
CHECKPOINT=CASE.parent/'pico_v4_native323_relativefoot_full_v1_checkpoint_00500.npz'
REVIEW=ARCHIVE/'bfm_online_intent_v2/pico_v4_native323_relativefoot_full_v1_checkpoint_00500_review_v1.json'
BASE_RENDERER=ROOT/'artifacts/teleop_six_hour_20260910/render_mpc_h30_clipped_partial.py'""")
text=text.replace('WIDTH,HEIGHT,TOP,BOTTOM=640,480,88,104','WIDTH,HEIGHT,TOP,BOTTOM=640,480,88,132')
text=text.replace("output=Path('C:/Users/camer/.codex/visualizations/2026/09/10/01a08b79-d89d-7d13-bcf8-33de331c0024/mpc_h30_clipped_partial_v1')", "output=ARCHIVE/'visual_pico_relativefoot_checkpoint500_v1'")
start=text.index("    report=json.loads((CASE/'report.json').read_text())")
stop=text.index("    with np.load(CASE/'trace.npz',allow_pickle=False) as z:",start)
text=text[:start]+"""    review=json.loads(REVIEW.read_text())
    request=json.loads((CASE/'request.json').read_text())
    assert sha(CHECKPOINT)==review['trace_sha256']
    assert sha(CASE/'request.json')==review['request_sha256']
    assert review['strict_physical_limits_pass'] and review['metadata']['completed_full_controls']==500
    assert request['mujoco']=='3.2.3' and request['feedback_correction_clip_rad']==.1
    motion,timeline,motion_path=load_case_motion('pico',review['motion_override'])
    assert sha(motion_path)==review['motion_sha256']
    original_path=Path('C:/Users/camer/sonic23_sim_artifacts/internet_pico_20260909_v1/pico_freedancing_v1/optical_reference_v2/original29.npz')
    assert sha(original_path)==review['original29_sha256']
    with np.load(original_path,allow_pickle=False) as original_archive:
        original_tasks=original_archive['source_task_position_w'].copy()
"""+text[stop:]
text=text.replace("with np.load(CASE/'trace.npz',allow_pickle=False) as z:","with np.load(CHECKPOINT,allow_pickle=False) as z:")
text=text.replace("    actual=trace['qpos']","    metadata=json.loads(trace['checkpoint_metadata'].item())\n    assert metadata==review['metadata']\n    actual=trace['qpos']")
text=text.replace("report['simulated_seconds']","metadata['simulation_time']")
start=text.index("    inputs=[CASE/'report.json'")
stop=text.index('    input_hashes=',start)
text=text[:start]+"""    inputs=[REVIEW,CHECKPOINT,CASE/'request.json',motion_path,original_path,BASE_RENDERER,
            ROOT.parent/'GR00T-WholeBodyControl'/MODEL,ROOT/PHYSICS,Path(__file__)]
"""+text[stop:]
text=text.replace('    def picture(qpos):','    def picture(qpos,frame_index):')
text=text.replace('        grid(renderer.scene,lo,hi)\n        return Image.fromarray',"""        grid(renderer.scene,lo,hi)
        # Reference landmarks stay in their original world coordinates in BOTH panes.
        # They are not translated onto the robot or used to align the camera.
        for point in original_tasks[frame_index]:
            g=renderer.scene.geoms[renderer.scene.ngeom]
            mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_SPHERE,np.array([.023,.023,.023]),point,np.eye(3).ravel(),np.array([.1,.9,1.,.8]))
            renderer.scene.ngeom+=1
        return Image.fromarray""")
text=text.replace('canvas.paste(picture(desired[i]),(0,TOP));canvas.paste(picture(actual[i]),(WIDTH,TOP))',
    'canvas.paste(picture(desired[i],source_indices[i]),(0,TOP));canvas.paste(picture(actual[i],source_indices[i]),(WIDTH,TOP))')
text=text.replace('REQUESTED NATIVE23 REFERENCE','DECLARED v4 NATIVE23 REFERENCE')
text=text.replace('Original native poses; no root alignment','Explicit retarget + floor lift; same world frame')
text=text.replace('H30 MPC: CLIPPED FEEDBACK REPLAY','PHYSICAL MPC CHECKPOINT: PICO')
text=text.replace('MuJoCo 3.2.3; offline plan; correction +/-0.1 rad','Native3.2.3; H30; relative-foot cost; feedback +/-0.1')
text=text.replace("f'WALK002 | physics", "f'PICO | physics")
text=text.replace("f'PARTIAL PROBE: first 3.000 s source only | root error {drift:.3f} m | leg RMSE {leg:.3f} rad'", "f'INCOMPLETE: 150 / 5780 source controls only | adapted root error {drift:.3f} m | leg RMSE {leg:.3f} rad'")
text=text.replace("'PROBE END: no failure here; remaining 10.340 s source and recovery not tested'", "'CHECKPOINT END: remaining112.600s source + recovery absent from this view'")
text=text.replace("'Full 7.000 s initialization preserved | same fixed world camera | 0.5 m grid'", "'Full7s entry preserved | cyan spheres: ORIGINAL29 world hand/head goals | fixed0.5m grid'")
text=text.replace("        return canvas", "        draw.text((16,TOP+HEIGHT+99),'OFFLINE: 740ms preview; planning takes seconds per100ms block | no realtime or full-source pass',font=f18,fill='#ffb6a4')\n        return canvas")
text=text.replace('    contact=[0,250,350,400,450,len(actual)-1]', '    contact=[0,250,350,400,450,len(actual)-1]\n    visual_indices=np.arange(0,len(actual),2,dtype=int)\n    assert visual_indices[-1]==500')
text=text.replace('        for i in range(len(actual)):', '        for visual_slot,i in enumerate(visual_indices):')
text=text.replace('duration=times[i+1]-times[i] if i+1<len(times) else .002', 'duration=times[visual_indices[visual_slot+1]]-times[i] if visual_slot+1<len(visual_indices) else .002')
text=text.replace("'-video_track_timescale','500','-movflags','+faststart',str(video)","'-video_track_timescale','500','-threads','1','-movflags','+faststart',str(video)")
text=text.replace('np.testing.assert_allclose(video_times,times,atol=1e-6,rtol=0)', 'np.testing.assert_allclose(video_times,times[visual_indices],atol=1e-6,rtol=0)')
text=text.replace("'-i',str(video),'-f','null','-'", "'-i',str(video),'-fps_mode','passthrough','-enc_time_base','1:500','-f','null','-'")
text=text.replace("sheet=Image.new('RGB',(1600,1260)","sheet=Image.new('RGB',(1600,1314)")
text=text.replace("resize((800,420)","resize((800,438)")
text=text.replace("(slot//2)*420", "(slot//2)*438")
text=text.replace("kind='native323_H30_clipped_feedback_partial_replay_vs_original_native_requested_reference'", "kind='native323_PICO_relative_foot_MPC_incomplete_checkpoint_vs_declared_v4_reference'")
text=text.replace("source_plan=provenance['plan']", "source_request=str(CASE/'request.json'),independent_checkpoint_review=str(REVIEW)")
text=text.replace("clip='walk002'", "clip='pico'")
text=text.replace("full_requested_source_seconds=source['requested_controls']*.02,", "full_requested_source_seconds=source['requested_controls']*.02,source_controls_shown=150,source_controls_requested=5780,\n                 original29_world_hand_head_markers=True,markers_aligned_to_actual_root=False,\n                 visual_sample_rate_hz=25,visual_control_indices=visual_indices.tolist(),source_control_rate_unchanged_hz=50,\n                 source_preview_seconds=.74,offline_planning=True,planning_timing_label='seconds per100ms block; exact timing absent from immutable checkpoint',")
text=text.replace('np.max(np.abs(video_times-times))', 'np.max(np.abs(video_times-times[visual_indices]))')
target=HERE/'render_pico_relativefoot_checkpoint500.py'
assert not target.exists()
target.write_text(text)
(HERE/'pico_checkpoint500_renderer_derivation.json').write_text(json.dumps(dict(base_renderer_sha256=hashlib.sha256(base.read_bytes()).hexdigest(),
    created_renderer=str(target),created_renderer_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
    changes='Immutable checkpoint adapter; PICO v4 reference; original29 world landmarks; complete10s timeline sampled25fps; E output; explicit partial/offline labels; single-thread encoding.'),indent=2))
print(target)
