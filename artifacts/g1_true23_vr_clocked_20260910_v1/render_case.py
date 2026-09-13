"""Display every measured EOF-transition state; no dynamics or robot interface."""

import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file


def main():
    directory = Path(__file__).parent / "end-of-stream"
    report_path = directory / "report.json"
    report = json.loads(report_path.read_text())
    trace_path = Path(report["trace_path"])
    if sha256_file(trace_path) != report["trace_sha256"]:
        raise ValueError("measured trace changed")
    native = next(Path(p) for p in report["source_files"] if p.endswith("/g1_23dof_rev_1_0.xml"))
    if sha256_file(native) != report["source_files"][str(native)]:
        raise ValueError("native23 display asset changed")
    output, receipt = directory / "walk_then_balance.measured.mp4", directory / "render.json"
    if output.exists() or receipt.exists():
        raise FileExistsError("render refuses overwrite")
    with np.load(trace_path, allow_pickle=False) as archive:
        poses = archive["qpos"].copy()
        fallback = archive["fallback_mode"].copy()
    if poses.shape != (report["completed_ticks"] + 1, 30) or fallback.shape != (len(poses) - 1,):
        raise ValueError("render must retain every measured state")
    model = mujoco.MjModel.from_xml_path(str(native))
    data = mujoco.MjData(model)
    model.vis.global_.offwidth, model.vis.global_.offheight = 960, 640
    model.vis.headlight.ambient[:] = .5
    renderer = mujoco.Renderer(model, height=640, width=960)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance, camera.azimuth, camera.elevation = 2.8, 120, -18
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 21)
    writer = imageio.get_writer(output, fps=50, codec="libx264", quality=7, macro_block_size=None)
    previews = []
    try:
        for index, pose in enumerate(poses):
            data.qpos[:] = pose
            mujoco.mj_fwdPosition(model, data)
            camera.lookat[:] = [pose[0], pose[1], .65]
            renderer.update_scene(data, camera=camera)
            canvas = Image.new("RGB", (960, 720), (17, 23, 31))
            canvas.paste(Image.fromarray(renderer.render()), (0, 80))
            draw = ImageDraw.Draw(canvas)
            balanced = index > 0 and bool(fallback[index - 1])
            mode = "Input ended: balance fallback (NOT firmware handback)" if balanced else "Public walking clip: full-body SONIC (tracking NOT qualified)"
            draw.text((16, 10), mode, font=font, fill=(248, 204, 127))
            draw.text((16, 43), f"{index / 50:.2f}s | measured 23-DOF SIM | follows root | NOT physical robot", font=font, fill=(237, 240, 244))
            writer.append_data(np.asarray(canvas))
            if index in {0, 650, 700, len(poses) - 1}:
                path = directory / f"frame_{index:04d}.png"
                canvas.save(path)
                previews.append(str(path))
            if index % 300 == 0:
                print(json.dumps(dict(frames=index + 1, total=len(poses))), flush=True)
    finally:
        writer.close()
        renderer.close()
    if sha256_file(trace_path) != report["trace_sha256"]:
        raise ValueError("trace changed while rendering")
    result = dict(output=str(output), sha256=sha256_file(output), frames=len(poses), fps=50,
                  report_sha256=sha256_file(report_path), trace_sha256=report["trace_sha256"],
                  renderer_sha256=sha256_file(Path(__file__)), previews=previews,
                  every_measured_state_rendered=True, camera_follows_root=True,
                  source_tracking_qualified=False, new_dynamics_integrated=False,
                  hardware_authorized=False, deployment_ready=False)
    with receipt.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
