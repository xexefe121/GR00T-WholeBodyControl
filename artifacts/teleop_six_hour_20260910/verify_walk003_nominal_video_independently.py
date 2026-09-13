"""Separate decoder/clock witness for the full walk003 recorded-pass video."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np

BASE=Path('E:/codex-artifacts/sonic23_teleop_six_hour_20260910')
CASE=BASE/'bfm_online_intent_v2/walk003_allmargin_wsl_independent_replay_v1'
OUT=BASE/'visual_walk003_allmargin_nominal_full_v1'
VIDEO=OUT/'full_source_and_lifecycle.fixed_world.mp4'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


report=json.loads((CASE/'report.json').read_text())
receipt=json.loads((OUT/'render_receipt.json').read_text())
audit=json.loads((CASE/'recorded_source_audit_v2.json').read_text())
assert sha(CASE/'trace.npz')==report['trace_sha256']==audit['trace_sha256']
assert audit['recorded_source_tracking_pass'] and all(audit['gates'].values())
assert sha(VIDEO)==receipt['video_sha256']
with np.load(CASE/'trace.npz',allow_pickle=False) as z:
    total_steps=len(z['physics_torque'])
    assert total_steps==15690 and z['physics_substeps'].shape==(1569,)
    np.testing.assert_array_equal(z['physics_substeps'],np.full(1569,10))
    np.testing.assert_array_equal(z['qpos'][-1],z['physics_qpos'][-1])
    last_second=np.abs(z['physics_qvel'][-500:,6:])
    row,joint=np.unravel_index(np.argmax(last_second),last_second.shape)
    event_step=total_steps-499+int(row)
    assert event_step==15563 and joint==10
    event_speed=float(last_second[row,joint])
expected_steps=np.unique(np.r_[np.arange(0,total_steps+1,20),event_step,total_steps])
expected_times=expected_steps*.002
ffprobe,ffmpeg=shutil.which('ffprobe'),shutil.which('ffmpeg');assert ffprobe and ffmpeg
decoded=json.loads(subprocess.check_output([ffprobe,'-v','error','-select_streams','v:0','-show_frames','-show_entries',
    'frame=best_effort_timestamp_time,pkt_duration_time,width,height','-of','json',str(VIDEO)]))
frames=decoded['frames'];times=np.asarray([float(f['best_effort_timestamp_time']) for f in frames])
np.testing.assert_allclose(times,expected_times,atol=1e-6,rtol=0)
assert all((f['width'],f['height'])==(1920,872) for f in frames)
decoded_run=subprocess.run([ffmpeg,'-hide_banner','-v','error','-threads','1','-i',str(VIDEO),'-fps_mode','passthrough',
    '-enc_time_base','1:500','-threads','1','-f','null','-'],capture_output=True,text=True,check=True)
assert not decoded_run.stderr.strip(),decoded_run.stderr
images=[]
for index,name in ((int(np.flatnonzero(expected_steps==event_step)[0]),'independently_decoded_ankle_event.png'),
                   (len(frames)-1,'independently_decoded_final_frame.png')):
    path=OUT/name
    subprocess.run([ffmpeg,'-hide_banner','-v','error','-threads','1','-i',str(VIDEO),'-vf',f'select=eq(n\\,{index})',
        '-frames:v','1','-threads','1',str(path)],capture_output=True,text=True,check=True)
    images.append(dict(path=str(path),sha256=sha(path),frame_index=index,physical_timestamp_seconds=float(times[index])))
result=dict(kind='independent_full_walk003_video_decode_and_physics_clock_witness',video_sha256=sha(VIDEO),
    source_trace_sha256=sha(CASE/'trace.npz'),render_receipt_sha256=sha(OUT/'render_receipt.json'),producer_sha256=sha(__file__),
    frames=len(frames),native_control_slots=1569,full_source_controls=819,source_controls_requested=819,
    physical_steps=total_steps,full_physics_seconds=31.38,source_seconds=16.38,
    maximum_timestamp_error_seconds=float(np.max(np.abs(times-expected_times))),
    exact_ankle_event_physics_step=event_step,exact_ankle_event_time_seconds=float(event_step*.002),
    exact_ankle_event_speed_radps=event_speed,exact_final_time_seconds=float(times[-1]),
    full_video_decoded=True,decoder_stderr_empty=True,independently_decoded_images=images,
    extra_terminal_display_hold_seconds=.002,no_physics_reexecution=True,recorded_source_tracking_pass=True,
    settled_standing_claim=False,live_controller_qualified=False,hardware_authorized=False)
path=OUT/'independent_decode_verification.json';assert not path.exists()
path.write_text(json.dumps(result,indent=2)+'\n')
(OUT/'independent_decoder_snapshot.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(result,indent=2))
