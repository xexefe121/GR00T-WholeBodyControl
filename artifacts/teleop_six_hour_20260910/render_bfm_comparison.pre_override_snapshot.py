"""Render saved desired native23 poses beside actual BFM MuJoCo states.

The right panel is immutable recorded qpos, not reference pose animation.
Both panels share a fixed camera and original world frame. This script runs
kinematics for visualization only; it executes no policy or dynamics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from gear_sonic.scripts.evaluate_g1_true23_bfmzero import load_motion
from gear_sonic.utils.g1_true23_generalist_benchmark import MODEL, PHYSICS
from gear_sonic.utils.g1_true23_step1b_mujoco import prepare_true23_model

WIDTH, HEIGHT = 640, 480
TOP, BOTTOM = 86, 74
BLUE, ORANGE = (91, 190, 255), (255, 182, 91)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def font(size):
    path = Path('C:/Windows/Fonts/segoeui.ttf')
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def phase_at(control, phases):
    return next((p['name'] for p in phases if p['control_start'] <= control < p['control_stop']), 'initial_state')


def add_grid(scene, bounds):
    lo, hi = bounds
    for axis in range(2):
        for position in np.arange(np.floor(lo[axis] * 2) / 2, np.ceil(hi[axis] * 2) / 2 + .1, .5):
            a = np.array([lo[0], lo[1], .003])
            b = np.array([hi[0], hi[1], .003])
            a[axis] = b[axis] = position
            g = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).ravel(), np.array([.38, .42, .47, 1.]))
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, .002, a, b)
            scene.ngeom += 1
    # World axes remain in identical positions in both panels.
    for endpoint, color in [(np.array([1., 0., .006]), [1., .25, .2, 1.]), (np.array([0., 1., .006]), [.2, .8, .4, 1.])]:
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).ravel(), np.array(color))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, .006, np.array([0., 0., .006]), endpoint)
        scene.ngeom += 1


def run(args):
    case = args.case.resolve()
    output = case / 'visual_comparison_v1'
    output.mkdir(exist_ok=True)
    if (output / 'render_receipt.json').exists():
        raise FileExistsError('completed render evidence already exists')
    report_path, trace_path = case / 'report.json', case / 'trace.npz'
    report_hash, trace_hash = sha(report_path), sha(trace_path)
    report = json.loads(report_path.read_text())
    motion, timeline, source_path = load_motion(report['clip'])
    source_hash = sha(source_path)
    with np.load(trace_path, allow_pickle=False) as archive:
        actual = archive['qpos'].copy()
    assert actual.shape == (report['completed'] + 1, 30)
    desired = np.concatenate((motion['body_pos_w'][10:10 + len(actual), 0], motion['body_quat_w'][10:10 + len(actual), 0], motion['joint_pos'][10:10 + len(actual)]), axis=1)
    assert desired.shape == actual.shape and np.isfinite(actual).all() and np.isfinite(desired).all()
    # Verify same time indexing as the referee, without altering either trace.
    with np.load(trace_path, allow_pickle=False) as archive:
        np.testing.assert_allclose(actual[1:, :3] - desired[1:, :3], archive['root_error'], atol=1e-12, rtol=0)
        np.testing.assert_allclose(actual[1:, 7:] - desired[1:, 7:], archive['joint_error'], atol=1e-12, rtol=0)
    _, model, _ = prepare_true23_model(ROOT.parent / 'GR00T-WholeBodyControl' / MODEL, ROOT / PHYSICS)
    model.vis.global_.offwidth, model.vis.global_.offheight = WIDTH, HEIGHT
    model.vis.headlight.ambient[:] = [.5, .5, .5]
    model.vis.headlight.diffuse[:] = [.8, .8, .8]
    model.vis.headlight.specular[:] = [.1, .1, .1]
    data = mujoco.MjData(model)
    all_xy = np.concatenate((actual[:, :2], desired[:, :2]))
    lo, hi = all_xy.min(0) - .8, all_xy.max(0) + .8
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [*((lo + hi) / 2), .65]
    camera.distance = max(4.3, float(np.max(hi - lo)) * 1.4)
    camera.azimuth, camera.elevation = 135, -20
    options = mujoco.MjvOption()
    options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
    renderer = mujoco.Renderer(model, HEIGHT, WIDTH)
    phases = timeline['phases']
    source = next(p for p in phases if p['name'] == 'source_motion')
    source_start, source_stop = source['control_start'], source['control_stop']
    total = report['completed']
    text24, text20, text18 = font(24), font(20), font(18)

    def picture(qpos):
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=camera, scene_option=options)
        add_grid(renderer.scene, (lo, hi))
        return Image.fromarray(renderer.render().copy())

    def combined(index):
        canvas = Image.new('RGB', (2 * WIDTH, TOP + HEIGHT + BOTTOM), '#121820')
        canvas.paste(picture(desired[index]), (0, TOP))
        canvas.paste(picture(actual[index]), (WIDTH, TOP))
        draw = ImageDraw.Draw(canvas)
        draw.text((20, 12), 'DESIRED NATIVE23 REFERENCE', fill=BLUE, font=text24)
        draw.text((WIDTH + 20, 12), 'ACTUAL MUJOCO SIMULATION', fill=ORANGE, font=text24)
        draw.text((20, 49), 'Kinematic source poses', fill='#c0ccd7', font=text20)
        draw.text((WIDTH + 20, 49), 'Recorded physics qpos; BFM-Zero policy', fill='#c0ccd7', font=text20)
        draw.line((WIDTH, TOP, WIDTH, TOP + HEIGHT), fill='#475569', width=2)
        lifecycle_time = index / 50
        source_time = (index - source_start) / 50
        phase = phase_at(max(0, index - 1), phases).replace('_', ' ')
        drift = np.linalg.norm(actual[index, :3] - desired[index, :3])
        leg_rmse = np.sqrt(np.mean((actual[index, 7:19] - desired[index, 7:19]) ** 2))
        draw.text((20, TOP + HEIGHT + 10), f"{report['clip'].upper()}  |  timeline {lifecycle_time:6.2f}s  |  source {source_time:6.2f}s  |  {phase}", fill='white', font=text20)
        draw.text((20, TOP + HEIGHT + 42), f'Root error {drift:.3f} m   |   Leg joint RMSE {leg_rmse:.3f} rad   |   Same fixed world camera; 0.5 m grid', fill='#bcc8d4', font=text18)
        return canvas

    # The walk is short: show its complete lifecycle. PICO uses declared,
    # separated source windows, plus final standing; no time relabeling.
    if report['clip'] == 'pico':
        segments = [('entry_and_early_motion', 0, min(total, source_start + 600)),
                    ('middle_motion', source_start + 2500, min(total, source_start + 2900)),
                    ('late_motion_and_return', max(source_start, source_stop - 400), total)]
    else:
        segments = [('full_lifecycle', 0, total)]
    outputs = []
    contact_indices = [0, source_start + 1, source_start + 200, (source_start + source_stop) // 2, source_stop, total]
    contact_frames = {}
    try:
        combined(source_start + 200).save(output / 'early_motion.source_vs_actual.fixed_world.png')
        for label, begin, end in segments:
            video_path = output / f'{label}.source_vs_actual.fixed_world.mp4'
            indices = list(range(begin, end + 1, 2))
            if indices[-1] != end:
                indices.append(end)
            with imageio.get_writer(str(video_path), fps=25, codec='libx264', quality=8, macro_block_size=16) as writer:
                for index in indices:
                    frame = combined(index)
                    writer.append_data(np.asarray(frame))
                    if index in contact_indices:
                        contact_frames[index] = frame
            outputs.append(dict(path=str(video_path), frames=len(indices), fps=25, first_boundary=begin,
                                last_boundary=end, original_timeline_seconds=[begin / 50, end / 50], sha256=sha(video_path)))
            print(json.dumps(dict(clip=report['clip'], rendered=label, frames=len(indices))), flush=True)
        for index in contact_indices:
            if index not in contact_frames:
                contact_frames[index] = combined(index)
        thumb_size = (800, 400)
        sheet = Image.new('RGB', (1600, 1200), '#121820')
        for slot, index in enumerate(contact_indices):
            sheet.paste(contact_frames[index].resize(thumb_size, Image.Resampling.LANCZOS), ((slot % 2) * 800, (slot // 2) * 400))
        sheet_path = output / 'contact_sheet.source_vs_actual.fixed_world.png'
        sheet.save(sheet_path)
        combined(source_start + 200).save(output / 'early_motion.source_vs_actual.fixed_world.png')
    finally:
        renderer.close()
    assert sha(report_path) == report_hash and sha(trace_path) == trace_hash and sha(source_path) == source_hash
    receipt = dict(kind='recorded_actual_qpos_vs_native23_kinematic_reference', clip=report['clip'],
                   report_sha256=report_hash, trace_sha256=trace_hash, reference_sha256=source_hash,
                   renderer_sha256=sha(__file__), source_pose_index='10 + recorded_qpos_index',
                   source_actual_alignment_verified=True, camera=dict(lookat=camera.lookat.tolist(),
                   distance=camera.distance, azimuth=camera.azimuth, elevation=camera.elevation),
                   fixed_world_camera=True, robot_pose_transforms_applied=False, physics_reexecuted=False,
                   hardware_operated=False, visual_pass_claim=False, model=str(ROOT.parent / 'GR00T-WholeBodyControl' / MODEL),
                   videos=outputs, contact_sheet=str(sheet_path), contact_sheet_qpos_indices=contact_indices,
                   limitations='Native23 source pose geometry is shown, not absent-joint original29 hand/head intent. Render is visualization of saved state only.')
    (output / 'render_receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(dict(output=str(output), contact_sheet=str(sheet_path), videos=len(outputs))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type=Path, required=True)
    run(parser.parse_args())
