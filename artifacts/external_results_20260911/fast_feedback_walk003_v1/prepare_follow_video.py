"""Provide a closer shared-camera view; retain the existing fixed-world video."""
from pathlib import Path
BASE=Path(__file__).resolve().parent
s=(BASE/'render_video.py').read_text()
s=s.replace("OUT = BASE / 'fast_feedback_walk003_v1/video_v1'","OUT = BASE / 'fast_feedback_walk003_v1/follow_video_v1'")
s=s.replace('full_walk003_and_continuous_hold.fixed_world.mp4','full_walk003_and_continuous_hold.shared_follow_camera.mp4')
s=s.replace("            for j, pose in enumerate((desired[slot], actual[slot])):","            camera.lookat[:] = (desired[slot, :3] + actual[slot, :3]) / 2\n            camera.lookat[2] = .75\n            camera.distance = 2.8\n            for j, pose in enumerate((desired[slot], actual[slot])):")
s=s.replace('Fixed world camera; original source timing','Shared moving camera; original source timing')
s=s.replace('fixed_camera=dict(',"camera_mode='same moving camera for both panels; no state alignment',\n                  final_camera=dict(")
s=s.replace('full_sampled_geometry_projection_ratio=float(ratio)','static_overview_projection_ratio=float(ratio)')
out=BASE/'render_follow_video.py';assert not out.exists();out.write_text(s,encoding='utf-8')
print(out)
