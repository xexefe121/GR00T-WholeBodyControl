"""Independent full decoder and physical-clock check for final hybrid visual."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import math
import numpy as np

E=Path('E:/codex-artifacts/sonic23_teleop_six_hour_20260910')
CASE=E/'bfm_online_intent_v2/walk003_terminal_bfm_yaw4_hybrid_v1'
EXT=CASE/'post_lifecycle_hold_5s'
OUT=E/'visual_walk003_terminal_bfm_yaw4_full_v1'
VIDEO=OUT/'full_lifecycle_and_separate_hold.fixed_world.mp4'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

receipt=json.loads((OUT/'render_receipt.json').read_text())
report=json.loads((CASE/'report.json').read_text());extra_report=json.loads((EXT/'report.json').read_text())
assert sha(VIDEO)==receipt['video_sha256']
assert sha(CASE/'trace.npz')==report['trace_sha256']
assert sha(EXT/'trace.npz')==extra_report['trace_sha256']
with np.load(CASE/'trace.npz',allow_pickle=False) as a,np.load(EXT/'trace.npz',allow_pickle=False) as b:
    np.testing.assert_array_equal(a['qpos'][-1],b['qpos'][0])
    np.testing.assert_array_equal(a['qvel'][-1],b['qvel'][0])
    np.testing.assert_array_equal(a['physics_qpos'][-1],b['physics_qpos'][0])
    assert len(a['physics_torque'])==15690 and len(b['physics_torque'])==2500
    np.testing.assert_array_equal(a['physics_substeps'],np.full(1569,10))
    np.testing.assert_array_equal(b['physics_substeps'],np.full(250,10))
    assert abs(a['physics_time'][-1]-b['physics_time'][0])==0.
    total=len(a['physics_torque'])+len(b['physics_torque'])
seconds=total/500
regular=np.asarray([round(k*500/15) for k in range(math.ceil(seconds*15))],dtype=int)
events=[12690,15690,total]
expected_steps=np.unique(np.r_[regular,events])
expected_times=expected_steps/500
ffprobe,ffmpeg=shutil.which('ffprobe'),shutil.which('ffmpeg');assert ffprobe and ffmpeg
probe=json.loads(subprocess.check_output([ffprobe,'-v','error','-select_streams','v:0','-show_frames','-show_entries',
    'frame=best_effort_timestamp_time,width,height,pkt_duration_time','-of','json',str(VIDEO)]))
frames=probe['frames'];times=np.asarray([float(f['best_effort_timestamp_time']) for f in frames])
assert all((f['width'],f['height'])==(1920,872) for f in frames)
np.testing.assert_allclose(times,expected_times,atol=1e-6,rtol=0)
decode=subprocess.run([ffmpeg,'-hide_banner','-v','error','-threads','1','-i',str(VIDEO),'-fps_mode','passthrough',
    '-enc_time_base','1:500','-threads','1','-f','null','-'],capture_output=True,text=True,check=True)
assert not decode.stderr.strip(),decode.stderr
images=[]
for step,name in zip(events,['switch','lifecycle_boundary','extension_final']):
    index=int(np.flatnonzero(expected_steps==step)[0]);path=OUT/('independently_decoded_'+name+'.png')
    subprocess.run([ffmpeg,'-hide_banner','-v','error','-threads','1','-i',str(VIDEO),'-vf',f'select=eq(n\\,{index})',
        '-frames:v','1','-threads','1',str(path)],capture_output=True,text=True,check=True)
    images.append(dict(path=str(path),sha256=sha(path),physical_step=step,physical_seconds=step/500,video_frame=index))
result=dict(kind='independent_yaw4_hybrid_video_decoder_and_original_clock_witness',
    video_sha256=sha(VIDEO),render_receipt_sha256=sha(OUT/'render_receipt.json'),
    lifecycle_trace_sha256=sha(CASE/'trace.npz'),separate_extension_trace_sha256=sha(EXT/'trace.npz'),
    decoder_script_sha256=sha(__file__),frames=len(frames),original_lifecycle_controls=1569,
    original_source_controls=819,separate_extension_controls=250,physics_steps=total,total_seconds=seconds,
    exact_switch_seconds=25.38,exact_lifecycle_boundary_seconds=31.38,exact_extension_final_seconds=36.38,
    displayed_timestamps_all_match_actual2ms_states=True,
    maximum_timestamp_error_seconds=float(np.max(np.abs(times-expected_times))),
    extension_boundary_physical_state_continuous=True,full_video_decoded=True,decoder_stderr_empty=True,
    independently_decoded_images=images,terminal_display_hold_seconds=.002,
    physics_reexecuted=False,live_teleoperation_qualified=False,hardware_authorized=False)
target=OUT/'independent_decode_verification.json';assert not target.exists()
target.write_text(json.dumps(result,indent=2)+'\n')
(OUT/'independent_decoder_snapshot.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(result,indent=2))
