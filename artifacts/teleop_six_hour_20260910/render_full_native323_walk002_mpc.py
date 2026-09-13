"""Fixed-world full lifecycle, with explicit physical and timing failures."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
CASE = Path('E:/codex-artifacts/sonic23_teleop_six_hour_20260910/mjbatch_full_v1/walk002_native323_h30_clip1_v1')
OUTPUT = CASE / 'visual_full_lifecycle_v1'
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from gear_sonic.utils.g1_true23_generalist_benchmark import MODEL, PHYSICS
from gear_sonic.utils.g1_true23_step1b_mujoco import prepare_true23_model

W, H, TOP, BOTTOM = 640, 480, 88, 132


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def font(size):
    return ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf', size)


def grid(scene, lo, hi):
    for axis in range(2):
        for value in np.arange(np.floor(lo[axis] * 2) / 2, np.ceil(hi[axis] * 2) / 2 + .1, .5):
            a, b = np.r_[lo, .003], np.r_[hi, .003]
            a[axis] = b[axis] = value
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).ravel(), np.array([.38, .42, .47, 1.]))
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, .002, a, b)
            scene.ngeom += 1
    for end, color in (([1, 0, .006], [1, .25, .2, 1]), ([0, 1, .006], [.2, .8, .4, 1])):
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).ravel(), np.array(color, float))
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, .006, np.array([0, 0, .006]), np.array(end, float))
        scene.ngeom += 1


def main():
    OUTPUT.mkdir(exist_ok=False)
    frames_dir = OUTPUT / 'frames'
    frames_dir.mkdir()
    report = json.loads((CASE / 'report.json').read_text())
    audit = json.loads((CASE / 'independent_full_lifecycle_audit.json').read_text())
    assert report['trace_sha256'] == sha(CASE / 'trace.npz') == audit['trace_sha256']
    assert sha(CASE / 'report.json') == audit['report_sha256']
    assert audit['strict_joint_range_pass'] is False and audit['full_source_controls'] == 667
    with np.load(CASE / 'trace.npz', allow_pickle=False) as z:
        physical = z['physics_qpos'].copy()
        source_frames = z['source_frame'].copy()
        np.testing.assert_array_equal(z['qpos'], physical[::10])
        assert np.all(z['physics_substeps'] == 10)
    with np.load(CASE / 'independent_intent_metrics.npz', allow_pickle=False) as z:
        excess = z['strict_range_excess_by_physics_step'].copy()
    motion_path = BASE / 'mjbatch_native23_inputs_v1/walk002/native_original.npz'
    timeline_path = BASE / 'mjbatch_native23_inputs_v1/walk002/timeline.json'
    with np.load(motion_path, allow_pickle=False) as z:
        motion = {key: z[key].copy() for key in z.files}
    timeline = json.loads(timeline_path.read_text())
    assert sha(motion_path) == audit['original_native_reference_sha256']
    contact_steps = [0, 3500, 5500, 8736, 10170, 11170, 13670, 14170]
    # 25fps visual sampling preserves the complete simulation clock. Include
    # exact phase boundaries and the first, peak, last 2ms ankle violations.
    selected = sorted(set(range(0, len(physical), 20)) | set(contact_steps) | {8729, 8753, len(physical) - 1})
    selected = np.asarray(selected, dtype=int)
    times = selected * .002
    active = np.maximum(0, (selected - 1) // 10)
    desired_frames = source_frames[active].copy()
    desired_frames[0] = 10
    actual = physical[selected]
    desired = np.column_stack((motion['body_pos_w'][desired_frames, 0], motion['body_quat_w'][desired_frames, 0], motion['joint_pos'][desired_frames]))
    assert times[0] == 0 and times[-1] == 28.34
    input_files = [CASE / name for name in ('report.json', 'request.json', 'trace.npz', 'independent_full_lifecycle_audit.json', 'independent_intent_metrics.npz')]
    input_files += [motion_path, timeline_path, ROOT.parent / 'GR00T-WholeBodyControl' / MODEL, ROOT / PHYSICS, Path(__file__)]
    inputs = {str(path): sha(path) for path in input_files}
    (OUTPUT / 'renderer_snapshot.py').write_bytes(Path(__file__).read_bytes())
    _, model, _ = prepare_true23_model(ROOT.parent / 'GR00T-WholeBodyControl' / MODEL, ROOT / PHYSICS)
    model.vis.global_.offwidth, model.vis.global_.offheight = W, H
    model.vis.headlight.ambient[:] = [.5, .5, .5]
    model.vis.headlight.diffuse[:] = [.8, .8, .8]
    data = mujoco.MjData(model)
    ankle = model.body('left_ankle_roll_link').id
    xy = np.concatenate((actual[:, :2], desired[:, :2]))
    lo, hi = xy.min(0) - .7, xy.max(0) + .7
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [*((lo + hi) / 2), .55]
    camera.distance = max(3.7, float(np.max(hi - lo)) * 1.35)
    camera.azimuth, camera.elevation = 135, -18
    renderer = mujoco.Renderer(model, H, W)
    option = mujoco.MjvOption()
    f24, f20, f18 = font(24), font(20), font(18)

    def picture(qpos, marker=False):
        data.qpos[:] = qpos
        mujoco.mj_kinematics(model, data)
        renderer.update_scene(data, camera=camera, scene_option=option)
        grid(renderer.scene, lo, hi)
        if marker:
            geom = renderer.scene.geoms[renderer.scene.ngeom]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([.035, .035, .035]), data.xpos[ankle], np.eye(3).ravel(), np.array([1., .05, .05, .7]))
            renderer.scene.ngeom += 1
        return Image.fromarray(renderer.render().copy())

    def combined(i):
        step, t = int(selected[i]), float(times[i])
        frame = Image.new('RGB', (2 * W, TOP + H + BOTTOM), '#121820')
        frame.paste(picture(desired[i]), (0, TOP))
        frame.paste(picture(actual[i], bool(excess[step] > 0)), (W, TOP))
        draw = ImageDraw.Draw(frame)
        draw.text((16, 10), 'REQUESTED NATIVE23 REFERENCE', font=f24, fill='#5bbeff')
        draw.text((16, 47), 'Original native poses; fixed world coordinates', font=f20, fill='#c0ccd7')
        draw.text((W + 16, 10), 'RECORDED PHYSICAL MPC SIMULATION', font=f24, fill='#ffb65b')
        draw.text((W + 16, 47), 'Native 3.2.3; strict ankle and foot gates failed', font=f20, fill='#ffb6a4')
        draw.line((W, TOP, W, TOP + H), fill='#475569', width=2)
        control = int(active[i])
        phase = next((p['name'].replace('_', ' ') for p in timeline['phases'] if p['control_start'] <= control < p['control_stop']), 'initial state')
        source_time = min(13.34, max(0., t - 7.))
        root_error = np.linalg.norm(actual[i, :3] - desired[i, :3])
        draw.text((16, TOP + H + 7), f'WALK002 | physics {t:.3f} / 28.340 s | source {source_time:.3f} / 13.340 s | {phase}', font=f20, fill='white')
        draw.text((16, TOP + H + 36), f'Root error now {root_error:.3f} m | source feet p95 0.137 / 0.107 m, root-relative world axes', font=f20, fill='#c0ccd7')
        status = f'VIOLATING NOW: {excess[step]:.6f} rad' if excess[step] > 0 else '25 physical samples breached native ankle range'
        draw.text((16, TOP + H + 65), f'STRICT LIMIT FAIL: left ankle max 0.001825 rad (0.105 deg) | {status}', font=f20, fill='#ff9e90')
        draw.text((16, TOP + H + 98), 'OFFLINE: planning p95 8.607 s / 0.100 s budget; 0.740 s preview | full clock; nominal video 25 fps', font=f18, fill='#c0ccd7')
        return frame

    kept, concat = {}, ['ffconcat version 1.0']
    try:
        for i, step in enumerate(selected):
            frame = combined(i)
            path = frames_dir / f'frame_{i:05d}.png'
            frame.save(path)
            if int(step) in contact_steps:
                kept[int(step)] = frame.copy()
            duration = times[i + 1] - times[i] if i + 1 < len(times) else .002
            concat.extend((f"file 'frames/{path.name}'", 'option framerate 500', f'duration {duration:.6f}'))
            if i % 100 == 0:
                print(json.dumps(dict(rendered=i, total=len(selected))), flush=True)
    finally:
        renderer.close()
    (OUTPUT / 'frames.ffconcat').write_text('\n'.join(concat) + '\n')
    ffmpeg, ffprobe = shutil.which('ffmpeg'), shutil.which('ffprobe')
    assert ffmpeg and ffprobe
    video = OUTPUT / 'full_lifecycle.fixed_world.mp4'
    subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', str(OUTPUT / 'frames.ffconcat'),
                    '-fps_mode', 'vfr', '-enc_time_base', '1:500', '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p',
                    '-video_track_timescale', '500', '-movflags', '+faststart', str(video)], check=True)
    info = json.loads(subprocess.check_output([ffprobe, '-v', 'error', '-select_streams', 'v:0', '-show_frames', '-show_entries',
                       'frame=best_effort_timestamp_time', '-of', 'json', str(video)]))
    video_times = np.asarray([float(row['best_effort_timestamp_time']) for row in info['frames']])
    np.testing.assert_allclose(video_times, times, atol=1e-6, rtol=0)
    subprocess.run([ffmpeg, '-v', 'error', '-i', str(video), '-f', 'null', '-'], check=True)
    tile_height = round((TOP + H + BOTTOM) * 800 / (2 * W))
    sheet = Image.new('RGB', (1600, tile_height * 4), '#121820')
    for index, step in enumerate(contact_steps):
        sheet.paste(kept[step].resize((800, tile_height), Image.Resampling.LANCZOS), ((index % 2) * 800, (index // 2) * tile_height))
    sheet.save(OUTPUT / 'contact_sheet.fixed_world.png')
    kept[8736].save(OUTPUT / 'worst_native_ankle_violation.fixed_world.png')
    kept[14170].save(OUTPUT / 'final_returned_standing.fixed_world.png')
    for path, digest in inputs.items():
        assert sha(path) == digest, path
    receipt = dict(kind='full_native323_walk002_mpc_fixed_world_render', inputs=inputs,
        renderer_snapshot_sha256=sha(OUTPUT / 'renderer_snapshot.py'), full_recorded_controls=1417,
        full_source_controls=667, recorded_physics_steps=14170, recorded_seconds=28.34,
        source_seconds=13.34, initialization_seconds=7., recovery_and_hold_seconds=8.,
        nominal_visual_sampling_hz=25, selected_physics_steps=selected.tolist(), desired_active_source_frames=desired_frames.tolist(),
        exact_2ms_ankle_violation_boundary_frames_included=[8729,8736,8753],
        desired_convention='The active 50Hz command sample; at inserted 2ms frames this is the same zero-order-held target used by the physical controller.',
        fixed_camera=dict(lookat=camera.lookat.tolist(),distance=camera.distance,azimuth=camera.azimuth,elevation=camera.elevation),
        root_alignment=False, yaw_alignment=False, camera_tracking=False, time_warp=False, pose_transforms=False,
        simulation_reexecuted=False, physical_floor_changed=False, annotations_only=True,
        video=str(video), video_sha256=sha(video), video_frames=len(video_times),
        maximum_timestamp_error_s=float(np.max(np.abs(video_times-times))), final_frame_hold_s=.002,
        full_video_decoded=True, full_body_tracking_qualified=False, timing_qualified=False, hardware_authorized=False,
        contact_sheet=str(OUTPUT / 'contact_sheet.fixed_world.png'), contact_physics_steps=contact_steps)
    (OUTPUT / 'render_receipt.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps({key:receipt[key] for key in ('video','video_frames','recorded_seconds','maximum_timestamp_error_s','contact_sheet')}), flush=True)


if __name__ == '__main__':
    main()
