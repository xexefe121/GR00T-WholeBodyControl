"""Render fixed-world snapshots of the qualified expert branch; no physics."""

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
BASE = Path('E:/codex-artifacts/sonic23_teleop_resume_20260911')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    qualification_path = BASE / 'expert_resumed_root_qualification_v1/qualification.json'
    qualification = json.loads(qualification_path.read_text())
    run = BASE / 'student_actual_oracle_control1_resume1001_v1'
    traces = [run / 'nominal/trace.npz', run / 'post_lifecycle_hold_5s/trace.npz']
    for path, key in zip(traces, ('nominal_trace_sha256', 'extension_trace_sha256')):
        assert sha(path) == qualification[key]
    for item in qualification['independent_reports'].values():
        assert sha(Path(item['path'])) == item['sha256']
    with np.load(traces[0]) as life, np.load(traces[1]) as hold:
        np.testing.assert_array_equal(life['final_integration'], hold['initial_integration'])
        np.testing.assert_array_equal(life['physics_qpos'][-1], hold['physics_qpos'][0])
        assert len(life['source_frame']) == 1569 and len(hold['source_frame']) == 250
        q = np.r_[life['physics_qpos'], hold['physics_qpos'][1:]]
    reference_path = Path('E:/codex-artifacts/sonic23_teleop_six_hour_20260910/mjbatch_intent_floor_inputs_v1/walk003/reference.npz')
    reference_receipt = json.loads((reference_path.parent / 'portable_receipt.json').read_text())
    assert sha(reference_path) == reference_receipt['reference_sha256']
    with np.load(reference_path) as reference:
        desired = np.c_[reference['body_pos_w'][:, 0], reference['body_quat_w'][:, 0], reference['joint_pos']]
    steps = np.array([0, 3500, 5500, 7500, 9500, 11690, 12690, 15690, 18190])
    frames = np.minimum(10 + (steps + 9) // 10, len(desired) - 1)
    actual, target = q[steps], desired[frames]
    bundle = ROOT / 'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1'
    manifest = json.loads((bundle / 'manifest.json').read_text())
    assert sha(bundle / 'native_prepared.xml') == manifest['portable_xml_sha256']
    assert sha(bundle / 'prepared_model_arrays.npz') == manifest['prepared_arrays_sha256']
    for name, digest in manifest['meshes'].items():
        assert sha(bundle / 'meshes' / name) == digest
    native = mujoco.MjModel.from_xml_path(str(bundle / 'native_prepared.xml'))
    with np.load(bundle / 'prepared_model_arrays.npz') as arrays:
        for name in arrays.files:
            getattr(native, name)[:] = arrays[name]
        mujoco.mj_setConst(native, mujoco.MjData(native))
        for name in arrays.files:
            np.testing.assert_array_equal(getattr(native, name), arrays[name])
    width, height = 480, 400
    native.vis.global_.offwidth, native.vis.global_.offheight = width, height
    native.vis.headlight.ambient[:] = .55
    native.vis.headlight.diffuse[:] = .8
    data = mujoco.MjData(native)
    ids = np.flatnonzero((native.geom_bodyid != 0) & (native.geom_rgba[:, 3] > 0))
    radii = native.geom_rbound[ids, None]
    low, high = np.full(3, np.inf), np.full(3, -np.inf)
    for pose in np.r_[actual, target]:
        data.qpos[:] = pose
        mujoco.mj_kinematics(native, data)
        low = np.minimum(low, (data.geom_xpos[ids] - radii).min(0) - .04)
        high = np.maximum(high, (data.geom_xpos[ids] + radii).max(0) + .04)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (low + high) / 2
    camera.distance, camera.azimuth, camera.elevation = 2., 135., -20.
    renderer = mujoco.Renderer(native, height=height, width=width)
    corners = np.array([[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])])
    for _ in range(150):
        renderer.update_scene(data, camera=camera)
        view = renderer.scene.camera[0]
        offsets = corners - view.pos
        depth = offsets @ view.forward
        tangent = view.frustum_top / view.frustum_near
        ratio = max(np.max(np.abs(offsets @ view.up / depth)) / tangent,
                    np.max(np.abs(offsets @ np.cross(view.forward, view.up) / depth)) / (tangent * width / height))
        if np.all(depth > 0) and ratio <= .92:
            break
        camera.distance *= 1.025
    else:
        raise RuntimeError('Full geometry could not fit fixed camera')
    out = BASE / 'expert_resumed_qualified_visual_v1'
    out.mkdir(exist_ok=False)
    font = ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf', 22)
    small = ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf', 18)
    panel_height = height + 65
    sheet = Image.new('RGB', (width * 2 * 3, panel_height * 3 + 100), '#121820')
    draw = ImageDraw.Draw(sheet)
    draw.text((18, 12), 'Native 23-DOF G1 | full walking expert branch + separate standing hold', font=font, fill='white')
    draw.text((18, 48), 'Each pair: declared native reference LEFT / recorded physics RIGHT. Fixed world camera. Offline expert; fast student still unqualified.', font=small, fill='#ffcf9e')
    try:
        for i, step in enumerate(steps):
            panel = Image.new('RGB', (width * 2, panel_height), '#121820')
            d = ImageDraw.Draw(panel)
            phase = ('initial' if step == 0 else 'source' if 3500 <= step <= 11690 else
                     'terminal switch' if step == 12690 else 'lifecycle end' if step == 15690 else 'separate hold end')
            d.text((12, 8), f'{step * .002:.2f}s | {phase} | reference / actual', font=font, fill='white')
            for j, pose in enumerate((target[i], actual[i])):
                data.qpos[:] = pose
                mujoco.mj_kinematics(native, data)
                renderer.update_scene(data, camera=camera)
                panel.paste(Image.fromarray(renderer.render().copy()), (j * width, 45))
            panel.save(out / f'frame_{step:05d}.png')
            sheet.paste(panel, ((i % 3) * width * 2, 100 + (i // 3) * panel_height))
    finally:
        renderer.close()
    sheet.save(out / 'contact_sheet.png')
    report = dict(kind='qualified_expert_recorded_physics_visual_no_dynamics',
                  selected_physics_steps=steps.tolist(), reference_frames=frames.tolist(),
                  physics_steps=18190, source_controls=819, fixed_world_camera=True,
                  camera=dict(lookat=camera.lookat.tolist(), distance=camera.distance,
                              azimuth=camera.azimuth, elevation=camera.elevation),
                  full_selected_geometry_projection_ratio=float(ratio),
                  no_root_alignment=True, no_time_warp=True, no_simulation_rerun=True,
                  full_lifecycle_and_separate_hold_qualified=True, live_teleoperation_qualified=False,
                  hashes={str(p): sha(p) for p in [*traces, qualification_path, reference_path,
                          bundle / 'native_prepared.xml', bundle / 'prepared_model_arrays.npz', Path(__file__)]},
                  image_sha256=sha(out / 'contact_sheet.png'))
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    (out / 'source.py').write_bytes(Path(__file__).read_bytes())
    print(out / 'contact_sheet.png')


if __name__ == '__main__':
    main()
