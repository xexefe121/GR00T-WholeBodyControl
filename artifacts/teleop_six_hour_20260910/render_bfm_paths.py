"""Show original-world root path and heading from saved BFM simulation states."""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from gear_sonic.scripts.evaluate_g1_true23_bfmzero import load_motion


def yaw(q):
    w, x, y, z = q.T
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


results = []
for clip in ('pico', 'walk002'):
    case = ROOT / 'artifacts/teleop_six_hour_20260910' / ('bfm_' + clip + '_v1')
    actual = np.load(case / 'trace.npz')['qpos']
    motion, timeline, path = load_motion(clip)
    desired = np.concatenate((motion['body_pos_w'][10:10 + len(actual), 0], motion['body_quat_w'][10:10 + len(actual), 0], motion['joint_pos'][10:10 + len(actual)]), axis=1)
    phase = next(p for p in timeline['phases'] if p['name'] == 'source_motion')
    sample = slice(phase['control_start'] + 1, phase['control_stop'] + 1)
    actual_yaw, desired_yaw = yaw(actual[:, 3:7]), yaw(desired[:, 3:7])
    angle_error = np.abs(np.angle(np.exp(1j * (actual_yaw - desired_yaw))))
    time = np.arange(phase['requested_controls']) / 50
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    for poses, headings, label, color in [(desired, desired_yaw, 'Desired native23 reference', '#087ca7'), (actual, actual_yaw, 'Actual MuJoCo simulation', '#ca5819')]:
        p = poses[sample]
        axes[0].plot(p[:, 0], p[:, 1], color=color, label=label, linewidth=1.7)
        axes[0].scatter(p[0, 0], p[0, 1], color=color, marker='o', s=45)
        axes[0].scatter(p[-1, 0], p[-1, 1], color=color, marker='X', s=65)
        axes[1].plot(time, np.unwrap(headings)[sample] * 180 / np.pi, color=color, linewidth=1.4, label=label)
    axes[0].set(xlabel='World X (m)', ylabel='World Y (m)', title='Root path: circle=start, cross=end')
    axes[0].set_aspect('equal', adjustable='datalim')
    axes[1].set(xlabel='Source time (s)', ylabel='Heading (degrees, unwrapped)', title='Absolute heading in original world frame')
    for ax in axes:
        ax.grid(alpha=.25)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].legend(loc='best', fontsize=9)
    p95 = np.percentile(angle_error[sample], 95) * 180 / np.pi
    fig.suptitle(f'{clip.upper()} | Native23 BFM-Zero | heading error p95 {p95:.1f} degrees\nSource phase only; no alignment, root resets, or time warping', fontsize=14)
    destination = case / 'visual_comparison_v1' / 'world_path_and_heading.png'
    fig.savefig(destination, dpi=170)
    plt.close(fig)
    record = dict(clip=clip,source_yaw_error_deg_p95=float(p95),source_yaw_error_deg_max=float(angle_error[sample].max() * 180 / np.pi),final_desired_xy=desired[-1,:2].tolist(),final_actual_xy=actual[-1,:2].tolist(),final_yaw_error_deg=float(angle_error[-1] * 180 / np.pi),source_heading_start_end_deg=(desired_yaw[[sample.start,sample.stop - 1]] * 180 / np.pi).tolist(),actual_heading_start_end_deg=(actual_yaw[[sample.start,sample.stop - 1]] * 180 / np.pi).tolist(),image=str(destination),no_state_transforms=True)
    results.append(record)
    print(json.dumps(record), flush=True)
(ROOT / 'artifacts/teleop_six_hour_20260910/bfm_visual_observations.json').write_text(json.dumps(results, indent=2))
