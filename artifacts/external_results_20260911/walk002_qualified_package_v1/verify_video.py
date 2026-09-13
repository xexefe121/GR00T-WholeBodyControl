"""Independently inspect encoded video timestamps and decode the exact handoff."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
from qualified_inputs import visual_grid,sha

BASE=Path(__file__).parent.parent
OUT=BASE/'walk002_video_verification_v1'
render=BASE/'walk002_qualified_hybrid_video_v1'
receipt=json.loads((render/'render_receipt.json').read_text())
video=Path(receipt['video'])
assert sha(video)==receipt['video_sha256']
steps,frames,durations=visual_grid()
assert steps.tolist()==receipt['visual_physics_steps'] and frames.tolist()==receipt['visual_reference_frames']
probe=shutil.which('ffprobe');ffmpeg=shutil.which('ffmpeg')
assert probe and ffmpeg
info=json.loads(subprocess.check_output([probe,'-v','error','-select_streams','v:0','-show_frames','-show_streams',
    '-show_entries','frame=best_effort_timestamp_time:stream=duration,nb_frames,width,height','-of','json',str(video)]))
times=np.asarray([float(frame['best_effort_timestamp_time']) for frame in info['frames']])
np.testing.assert_allclose(times,steps*.002,atol=1e-6,rtol=0)
stream=info['streams'][0]
assert int(stream['nb_frames'])==338 and (stream['width'],stream['height'])==(1280,644)
assert abs(float(stream['duration'])-33.342)<1e-6
OUT.mkdir(exist_ok=False)
decoded={}
for step in (11170,14170,16670):
    index=int(np.flatnonzero(steps==step)[0])
    path=OUT/f'decoded_step{step}_{step*.002:.2f}s.png'
    subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-threads','1','-i',str(video),
        '-vf',f'select=eq(n\\,{index})','-frames:v','1','-threads','1',str(path)],check=True)
    decoded[str(step)]=dict(frame_index=index,reference_frame=int(frames[index]),timestamp_seconds=float(times[index]),
        path=str(path),sha256=sha(path))
result=dict(kind='independent_encoded_video_timestamp_and_boundary_verification',all338_timestamps_on_native_grid=True,
    frame_count=len(times),maximum_timestamp_error_seconds=float(np.abs(times-steps*.002).max()),
    full_physical_duration_seconds=33.34,encoded_duration_seconds=float(stream['duration']),explicit_final_display_duration_seconds=.002,
    decoded=decoded,input_video_sha256=sha(video),render_receipt_sha256=sha(render/'render_receipt.json'),
    no_controller_inference_or_dynamics=True)
(OUT/'report.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
