"""Render recorded physical states and declared reference; no physics execution."""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(args):
    manifest = json.loads((args.bundle / 'manifest.json').read_text())
    assert sha(args.bundle / 'native_prepared.xml') == manifest['portable_xml_sha256']
    assert sha(args.bundle / 'prepared_model_arrays.npz') == manifest['prepared_arrays_sha256']
    for name, digest in manifest['meshes'].items():
        assert sha(args.bundle / 'meshes' / name) == digest
    native = mujoco.MjModel.from_xml_path(str(args.bundle / 'native_prepared.xml'))
    with np.load(args.bundle / 'prepared_model_arrays.npz', allow_pickle=False) as archive:
        for name in archive.files:
            getattr(native, name)[:] = archive[name]
        mujoco.mj_setConst(native, mujoco.MjData(native))
        for name in archive.files:
            np.testing.assert_array_equal(getattr(native, name), archive[name])
    assert (native.nq, native.nv, native.nu) == (30, 29, 23)
    with np.load(args.trace, allow_pickle=False) as archive:
        q = (archive['physics_qpos'].copy() if 'physics_qpos' in archive
             else archive['physics_states'][:, :30].copy())
    with np.load(args.reference, allow_pickle=False) as archive:
        reference = np.c_[archive['body_pos_w'][:, 0],
                          archive['body_quat_w'][:, 0], archive['joint_pos']]
    completed_steps = len(q) - 1
    if not 0 < completed_steps <= args.requested_controls * 10:
        raise ValueError('trace length does not match intended segment')
    steps = np.unique(np.rint(np.linspace(0, completed_steps, 6)).astype(int))
    frames = args.start_control + 10 + (steps + 9) // 10
    actual, desired = q[steps], reference[frames]
    args.output.mkdir(parents=True, exist_ok=False)
    data = mujoco.MjData(native)
    native.vis.global_.offwidth = 420
    native.vis.global_.offheight = 380
    native.vis.headlight.ambient[:] = [.55, .55, .55]
    native.vis.headlight.diffuse[:] = [.8, .8, .8]
    center = np.mean(np.r_[actual[:, :2], desired[:, :2]], axis=0)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [*center, .42]
    camera.distance, camera.azimuth, camera.elevation = 2.1, 135, -18
    renderer = mujoco.Renderer(native, height=380, width=420)
    font = ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf', 18)
    title = ImageFont.truetype('C:/Windows/Fonts/segoeuib.ttf', 24)
    canvas = Image.new('RGB', (420 * len(steps), 880), '#121820')
    draw = ImageDraw.Draw(canvas)
    draw.text((14, 8), 'PICO bounded recovery | declared native23 reference (top), recorded physics (bottom)', font=title, fill='white')
    draw.text((14, 43),
              f'{completed_steps * .002:.3f}s recorded / {args.requested_controls * .02:.3f}s requested; full source and live control unqualified',
              font=font, fill='#ffc48c')
    try:
        for column, step in enumerate(steps):
            for row, pose in enumerate((desired[column], actual[column])):
                data.qpos[:] = pose
                mujoco.mj_kinematics(native, data)
                renderer.update_scene(data, camera=camera)
                canvas.paste(Image.fromarray(renderer.render().copy()), (column * 420, 80 + row * 395))
            local_time = step * .002
            source_time = args.start_control * .02 + local_time - 7
            draw.text((column * 420 + 12, 853), f'+{local_time:.3f}s | source {source_time:.3f}s', font=font, fill='white')
    finally:
        renderer.close()
    canvas.save(args.output / 'contact_sheet.png')
    report = dict(
        kind='recorded_state_visualization_no_physics_execution',
        completed_physics_steps=completed_steps, requested_controls=args.requested_controls,
        full_source_qualified=False, steps=steps.tolist(), reference_frames=frames.tolist(),
        camera=dict(lookat=camera.lookat.tolist(), distance=camera.distance,
                    azimuth=camera.azimuth, elevation=camera.elevation),
        input_hashes={str(p): sha(p) for p in (args.trace, args.reference, Path(__file__),
            args.bundle / 'native_prepared.xml', args.bundle / 'prepared_model_arrays.npz')},
        image_sha256=sha(args.output / 'contact_sheet.png'),
    )
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    (args.output / 'source.py').write_bytes(Path(__file__).read_bytes())
    print(args.output / 'contact_sheet.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'trace', 'reference', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--start-control', type=int, required=True)
    parser.add_argument('--requested-controls', type=int, required=True)
    main(parser.parse_args())
