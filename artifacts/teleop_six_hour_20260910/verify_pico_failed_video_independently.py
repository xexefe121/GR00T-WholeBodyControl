"""Separate decoder/timestamp witness for the full available failed PICO video."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np

BASE=Path('E:/codex-artifacts/sonic23_teleop_six_hour_20260910')
CASE=BASE/'mjbatch_full_v1/pico_v4_native323_relativefoot_full_v1'
OUT=BASE/'visual_pico_relativefoot_failed_full_v1'
VIDEO=OUT/'full_available_failed_source.fixed_world.mp4'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


report=json.loads((CASE/'report.json').read_text())
receipt=json.loads((OUT/'render_receipt.json').read_text())
assert sha(CASE/'trace.npz')==report['trace_sha256']
assert sha(VIDEO)==receipt['video_sha256']
with np.load(CASE/'trace.npz') as z:
    total_steps=len(z['physics_torque'])
    assert total_steps==8725 and z['physics_substeps'].shape==(873,) and z['physics_substeps'][-1]==5
    np.testing.assert_array_equal(z['qpos'][-1],z['physics_qpos'][-1])
expected_steps=np.unique(np.r_[np.arange(0,total_steps,20),total_steps-1,total_steps])
expected_times=expected_steps*.002
ffprobe,ffmpeg=shutil.which('ffprobe'),shutil.which('ffmpeg')
assert ffprobe and ffmpeg
decoded=json.loads(subprocess.check_output([ffprobe,'-v','error','-select_streams','v:0','-show_frames','-show_entries',
    'frame=best_effort_timestamp_time,pkt_duration_time,width,height','-of','json',str(VIDEO)]))
frames=decoded['frames'];times=np.asarray([float(f['best_effort_timestamp_time']) for f in frames])
np.testing.assert_allclose(times,expected_times,atol=1e-6,rtol=0)
assert all((f['width'],f['height'])==(1280,700) for f in frames)
decoded_run=subprocess.run([ffmpeg,'-hide_banner','-v','error','-threads','1','-i',str(VIDEO),'-fps_mode','passthrough',
    '-enc_time_base','1:500','-threads','1','-f','null','-'],capture_output=True,text=True,check=True)
assert not decoded_run.stderr.strip(),decoded_run.stderr
last=OUT/'independently_decoded_final_frame.png'
subprocess.run([ffmpeg,'-hide_banner','-v','error','-threads','1','-i',str(VIDEO),'-vf',f'select=eq(n\\,{len(frames)-1})',
    '-frames:v','1','-threads','1',str(last)],capture_output=True,text=True,check=True)
result=dict(kind='independent_failed_video_decode_and_physics_clock_witness',video_sha256=sha(VIDEO),
    source_trace_sha256=sha(CASE/'trace.npz'),render_receipt_sha256=sha(OUT/'render_receipt.json'),producer_sha256=sha(__file__),
    frames=len(frames),native_control_slots=873,full_controls=872,source_full_controls=522,source_partial_substeps=5,
    physical_steps=total_steps,entire_available_physics_seconds=17.45,source_seconds=10.45,
    maximum_timestamp_error_seconds=float(np.max(np.abs(times-expected_times))),
    final_two_timestamp_seconds=times[-2:].tolist(),exact_final2ms_physical_states_present=True,
    full_video_decoded=True,decoder_stderr_empty=True,decoded_final_frame=str(last),decoded_final_frame_sha256=sha(last),
    extra_terminal_display_hold_seconds=.002,no_physics_reexecution=True,hardware_authorized=False)
path=OUT/'independent_decode_verification.json';assert not path.exists()
path.write_text(json.dumps(result,indent=2))
(OUT/'independent_decoder_snapshot.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(result,indent=2))
