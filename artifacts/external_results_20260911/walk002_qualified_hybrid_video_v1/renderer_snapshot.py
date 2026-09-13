"""Render qualified walk002 offline hybrid and hold from saved physical states."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from qualified_inputs import load_qualified, visual_grid, CONTACT_STEPS

ROOT = Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof')
BASE = Path('E:/codex-artifacts/sonic23_teleop_resume_20260911')
OUT = BASE / 'walk002_qualified_hybrid_video_v1'
WIDTH, HEIGHT, TOP, BOTTOM = 640, 480, 64, 100


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(qualification_path, qualification_sha256):
    qualification, inputs, full_arrays, hold_arrays = load_qualified(qualification_path, qualification_sha256)
    inputs[str(Path(__file__))] = sha(Path(__file__))
    inputs[str(Path(__file__).with_name('qualified_inputs.py'))] = sha(Path(__file__).with_name('qualified_inputs.py'))
    with np.load(qualification['traces']['full']['path'], allow_pickle=False) as full:
        q, dq, times = [full[key].copy() for key in ('physics_qpos', 'physics_qvel', 'physics_time')]
    with np.load(qualification['traces']['hold']['path'], allow_pickle=False) as hold:
        for arr, key in ((q, 'physics_qpos'), (dq, 'physics_qvel'), (times, 'physics_time')):
            np.testing.assert_array_equal(arr[-1], hold[key][0])
        q = np.concatenate((q, hold['physics_qpos'][1:]))
        dq = np.concatenate((dq, hold['physics_qvel'][1:]))
        times = np.concatenate((times, hold['physics_time'][1:]))
    assert q.shape == (16671, 30) and dq.shape == (16671, 29)
    np.testing.assert_allclose(times, np.arange(16671) * .002, atol=2e-10, rtol=0)
    reference_path = Path('E:/codex-artifacts/sonic23_teleop_six_hour_20260910/mjbatch_intent_floor_inputs_v1/walk002/reference.npz')
    receipt_path = reference_path.parent / 'portable_receipt.json'
    receipt = json.loads(receipt_path.read_text())
    assert sha(reference_path) == receipt['reference_sha256']
    with np.load(reference_path, allow_pickle=False) as reference:
        desired_all = np.c_[reference['body_pos_w'][:, 0], reference['body_quat_w'][:, 0], reference['joint_pos']]
    # Exact 100 ms visual sampling is a subset of the unchanged 2 ms physics grid.
    # Include the original lifecycle boundary and the final hold endpoint explicitly.
    steps, frames, durations = visual_grid()
    assert len(desired_all) == 1428
    actual, desired = q[steps], desired_all[frames]
    bundle = ROOT / 'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1'
    manifest = json.loads((bundle / 'manifest.json').read_text())
    assert sha(bundle / 'native_prepared.xml') == manifest['portable_xml_sha256']
    assert sha(bundle / 'prepared_model_arrays.npz') == manifest['prepared_arrays_sha256']
    for name, digest in manifest['meshes'].items():
        assert sha(bundle / 'meshes' / name) == digest
        inputs[str(bundle / 'meshes' / name)] = digest
    for path in (reference_path, receipt_path, bundle / 'manifest.json', bundle / 'native_prepared.xml',
                 bundle / 'prepared_model_arrays.npz', bundle / 'walk002/timeline.json'):
        inputs[str(path)] = sha(path)
    timeline = json.loads((bundle / 'walk002/timeline.json').read_text())
    full_intent = json.loads(Path(qualification['independent_reports']['full_intent']['path']).read_text())
    for qualified_asset in (reference_path, receipt_path, bundle / 'native_prepared.xml', bundle / 'prepared_model_arrays.npz', bundle / 'walk002/timeline.json'):
        assert sha(qualified_asset) in full_intent['hashes'].values(), str(qualified_asset)
    model = mujoco.MjModel.from_xml_path(str(bundle / 'native_prepared.xml'))
    with np.load(bundle / 'prepared_model_arrays.npz', allow_pickle=False) as arrays:
        for name in arrays.files:
            getattr(model, name)[:] = arrays[name]
        mujoco.mj_setConst(model, mujoco.MjData(model))
        for name in arrays.files:
            np.testing.assert_array_equal(getattr(model, name), arrays[name])
    assert mujoco.__version__ == '3.2.3' and (model.nq, model.nv, model.nu) == (30, 29, 23)
    model.vis.global_.offwidth, model.vis.global_.offheight = WIDTH, HEIGHT
    model.vis.headlight.ambient[:] = .55
    model.vis.headlight.diffuse[:] = .8
    data = mujoco.MjData(model)
    ids = np.flatnonzero((model.geom_bodyid != 0) & (model.geom_rgba[:, 3] > 0))
    radii = model.geom_rbound[ids, None]
    low, high = np.full(3, np.inf), np.full(3, -np.inf)
    for pose in np.concatenate((actual, desired)):
        data.qpos[:] = pose
        mujoco.mj_kinematics(model, data)
        low = np.minimum(low, (data.geom_xpos[ids] - radii).min(0) - .04)
        high = np.maximum(high, (data.geom_xpos[ids] + radii).max(0) + .04)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (low + high) / 2
    camera.distance, camera.azimuth, camera.elevation = 2., 135., -20.
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    corners = np.array([[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])])
    for _ in range(150):
        renderer.update_scene(data, camera=camera)
        view = renderer.scene.camera[0]
        offsets = corners - view.pos
        depth = offsets @ view.forward
        tangent = view.frustum_top / view.frustum_near
        ratio = max(np.max(np.abs(offsets @ view.up / depth)) / tangent,
                    np.max(np.abs(offsets @ np.cross(view.forward, view.up) / depth)) / (tangent * WIDTH / HEIGHT))
        if np.all(depth > 0) and ratio <= .92:
            break
        camera.distance *= 1.025
    else:
        raise RuntimeError('Complete sampled route could not fit fixed camera')
    OUT.mkdir(exist_ok=False)
    (OUT / 'frames').mkdir()
    (OUT / 'renderer_snapshot.py').write_bytes(Path(__file__).read_bytes())
    font = ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf', 22)
    small = ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf', 19)
    contact_steps = CONTACT_STEPS
    kept, concat = {}, ['ffconcat version 1.0']
    try:
        for slot, step in enumerate(steps):
            canvas = Image.new('RGB', (WIDTH * 2, TOP + HEIGHT + BOTTOM), '#121820')
            draw = ImageDraw.Draw(canvas)
            draw.text((14, 6), 'DECLARED NATIVE 23-DOF REFERENCE', font=font, fill='#80c6ff')
            draw.text((WIDTH + 14, 6), 'RECORDED NATIVE PHYSICS', font=font, fill='#a5edb4')
            draw.text((14, 35), 'Fixed world camera; original source timing', font=small, fill='#cbd5e1')
            draw.text((WIDTH + 14, 35), 'Qualified source and both quiet windows', font=small, fill='#cbd5e1')
            for j, pose in enumerate((desired[slot], actual[slot])):
                data.qpos[:] = pose
                mujoco.mj_kinematics(model, data)
                renderer.update_scene(data, camera=camera)
                canvas.paste(Image.fromarray(renderer.render().copy()), (j * WIDTH, TOP))
            seconds = step * .002
            if step == 14170:
                phase = 'original lifecycle end / separate hold boundary'
            elif step > 14170:
                phase = f'separate hold +{seconds - 28.34:.2f}/5.0 s'
            else:
                control = max(0, (step - 1) // 10)
                phase = next(p['name'].replace('_', ' ') for p in timeline['phases'] if p['control_start'] <= control < p['control_stop'])
            source_seconds = np.clip(seconds - 7., 0, 13.34)
            draw.text((14, TOP + HEIGHT + 5), f'WALK002 | simulation {seconds:.2f}/33.34 s | source {source_seconds:.2f}/13.34 s | {phase}', font=small, fill='white')
            speed = np.abs(dq[step, 6:]).max()
            root_error = np.linalg.norm(actual[slot, :3] - desired[slot, :3])
            draw.text((14, TOP + HEIGHT + 34), f'16,670 audited physics steps | joint speed now {speed:.3f} rad/s | root error now {root_error:.3f} m', font=small, fill='#b4edc0')
            mode = 'Recorded MPC commands' if step < 11170 else 'Switch to actual terminal BFM' if step == 11170 else 'Actual terminal BFM yaw4'
            draw.text((14, TOP + HEIGHT + 64), mode + ' | OFFLINE HYBRID | live-speed control and real Pico remain unqualified', font=small, fill='#ffd0a3')
            path = OUT / 'frames' / f'frame_{int(step):05d}.png'
            canvas.save(path)
            if step in contact_steps:
                kept[int(step)] = canvas.copy()
            duration = float(durations[slot])
            concat.extend((f"file 'frames/{path.name}'", 'option framerate 500', f'duration {duration:.6f}'))
            if slot % 100 == 0:
                print(json.dumps({'rendered': slot, 'total': len(steps), 'simulation_seconds': seconds}), flush=True)
    finally:
        renderer.close()
    assert set(kept) == set(contact_steps)
    (OUT / 'frames.ffconcat').write_text('\n'.join(concat) + '\n')
    ffmpeg, ffprobe = shutil.which('ffmpeg'), shutil.which('ffprobe')
    assert ffmpeg and ffprobe
    video = OUT / 'full_walk002_hybrid_and_continuous_hold.fixed_world.mp4'
    subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-threads', '1', '-f', 'concat', '-safe', '0',
                    '-i', str(OUT / 'frames.ffconcat'), '-fps_mode', 'vfr', '-enc_time_base', '1:500',
                    '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', '-video_track_timescale', '500',
                    '-threads', '1', '-movflags', '+faststart', str(video)], check=True)
    info = json.loads(subprocess.check_output([ffprobe, '-v', 'error', '-select_streams', 'v:0', '-show_frames',
                    '-show_entries', 'frame=best_effort_timestamp_time', '-of', 'json', str(video)]))
    video_times = np.array([float(frame['best_effort_timestamp_time']) for frame in info['frames']])
    np.testing.assert_allclose(video_times, steps * .002, atol=1e-6, rtol=0)
    sheet = Image.new('RGB', (1280, 1288), '#121820')
    for slot, step in enumerate(contact_steps):
        sheet.paste(kept[step].resize((640, 322), Image.Resampling.LANCZOS), ((slot % 2) * 640, (slot // 2) * 322))
    sheet.save(OUT / 'contact_sheet.png')
    assert all(sha(Path(path)) == digest for path, digest in inputs.items())
    report = dict(kind='full_qualified_walk002_recorded_MPC_prefix_actual_terminal_BFM_saved_state_visual', inputs=inputs,
                  video_sha256=sha(video), video=str(video), contact_sheet_sha256=sha(OUT / 'contact_sheet.png'),
                  source_controls=667, lifecycle_controls=1417, separate_hold_controls=250,
                  physical_steps=16670, display_seconds=33.34, final_display_hold_seconds=.002,
                  frame_count=len(steps), visual_physics_steps=steps.tolist(), visual_reference_frames=frames.tolist(),
                  maximum_timestamp_error_seconds=float(np.max(np.abs(video_times - steps * .002))),
                  fixed_camera=dict(lookat=camera.lookat.tolist(), distance=camera.distance,
                                    azimuth=camera.azimuth, elevation=camera.elevation),
                  full_sampled_geometry_projection_ratio=float(ratio), no_new_dynamics=True,
                  no_root_alignment=True, no_time_warp=True, original_prefix_controls=1117,
                  controller_switch_step=11170, original_terminal_BFM_controls=300,
                  variable_visual_frame_durations_seconds=durations.tolist(), live_teleoperation_qualified=False)
    (OUT / 'render_receipt.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'video': str(video), 'sha256': sha(video), 'frames': len(steps)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--qualification', type=Path, required=True)
    parser.add_argument('--qualification-sha256', required=True)
    args = parser.parse_args()
    main(args.qualification, args.qualification_sha256)
