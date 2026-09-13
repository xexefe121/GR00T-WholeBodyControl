"""Compare the old failed state segment with the partial hard-feasible repair."""

import json
import hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OLD = Path('E:/codex-artifacts/sonic23_teleop_six_hour_20260910')
NEW = Path('E:/codex-artifacts/sonic23_teleop_resume_20260911')
with np.load(OLD / 'mjbatch_full_v1/pico_v4_native323_freshseed_allmargin_5iter_full_v1/trace.npz') as a:
    old = {k: a[k].copy() for k in ('qpos', 'physics_qpos', 'physics_substeps')}
with np.load(NEW / 'hard_feasibility_continuation_3740_v1/trace.npz') as a:
    new = {k: a[k].copy() for k in ('qpos', 'physics_qpos', 'physics_time')}
bundle = ROOT / 'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1'
reference_path = OLD / 'mjbatch_intent_floor_inputs_v1/pico/reference.npz'
with np.load(reference_path) as a:
    ref = {k: a[k].copy() for k in ('joint_pos', 'body_pos_w', 'body_quat_w')}
with np.load(bundle / 'pico/original29.npz') as a:
    original = a['source_qpos29'].copy()
contract = json.loads((bundle / 'contract.json').read_text())
limits = np.asarray(contract['joint_limits'])
post = np.arange(3740, 3771)
source = (post - 350) * .02
frames = post + 10

def yaw(q):
    w, x, y, z = np.asarray(q).T
    return np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))

def wrap(x):
    return np.arctan2(np.sin(x), np.cos(x))

old_t = np.arange(37400, len(old['physics_qpos'])) * .002 - 7
new_t = new['physics_time'] - 7
fig, ax = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
for joint, name, colour in ((4, 'Left ankle pitch', '#1769aa'), (10, 'Right ankle pitch', '#5b8c38')):
    ax[0, 0].plot(new_t, new['physics_qpos'][:, joint+7], label=name+' / repair', c=colour, lw=1.8)
ax[0, 0].plot(old_t, old['physics_qpos'][37400:, 11], c='#c43d3d', label='Left ankle pitch / old', lw=1.6)
ax[0, 0].axhline(limits[4, 0], c='#202020', ls='--', lw=1, label='Native lower bound')
ax[0, 0].set(ylabel='Joint position (rad)', title='Old bound violation avoided')
ax[0, 0].legend(fontsize=8, loc='best')
ax[0, 1].plot(new_t, new['physics_qpos'][:, 2], c='#1769aa', label='Repair')
ax[0, 1].plot(old_t, old['physics_qpos'][37400:, 2], c='#c43d3d', label='Old')
ax[0, 1].plot(source, original[frames, 2], c='#333333', ls='--', label='Original root intent')
ax[0, 1].set(ylabel='Root height (m)', title='Physical height and original intent')
ax[0, 1].legend(fontsize=8)
yaw_error = np.abs(wrap(yaw(new['qpos'][:, 3:7]) - yaw(original[frames, 3:7]))) * 180/np.pi
ax[1, 0].plot(source, yaw_error, c='#1769aa')
ax[1, 0].axhline(15, c='#333333', ls='--', lw=1, label='Full-source p95 gate: 15°')
ax[1, 0].set(ylabel='Absolute heading error (degrees)', title='Heading still needs correction')
ax[1, 0].legend(fontsize=8)
leg_error = np.sqrt(np.mean((new['qpos'][:, 7:19] - ref['joint_pos'][frames, :12])**2, axis=1))
ax[1, 1].plot(source, leg_error, c='#1769aa')
ax[1, 1].axhline(.15, c='#333333', ls='--', lw=1, label='Full-source aggregate RMSE gate: .15rad')
ax[1, 1].set(ylabel='Per-frame leg-joint RMSE (rad)', title='Leg error against declared native retarget')
ax[1, 1].legend(fontsize=8)
for pane in ax.flat:
    pane.axvline(68.166, c='#c43d3d', ls=':', lw=1)
    pane.set(xlim=(67.8, 68.4), xlabel='Original source time (s)')
    pane.grid(alpha=.18)
fig.suptitle('Native23 PICO repair: 0.6s of requested 2s, then no feasible seed\n'
             'Actual physics; restored initial state at 67.8s. Full-source / live teleoperation unqualified.', fontsize=12)
output = NEW / 'bounded_recovery_plot_v2'
output.mkdir(exist_ok=False)
fig.savefig(output / 'recovery.png', dpi=160)
fig.savefig(output / 'recovery.pdf')
np.savez_compressed(output / 'plotted_metrics.npz', source_time=source,
                    yaw_error_deg=yaw_error, leg_frame_rmse_rad=leg_error)
(output / 'runner_snapshot.py').write_bytes(Path(__file__).read_bytes())
(output / 'report.json').write_text(json.dumps(dict(
    predecessor='bounded_recovery_plot_v1 used unretargeted native leg angles; this version uses the declared v4 acceptance reference',
    metric_controls=30, source_seconds=[67.8, 68.4], initial_sample_excluded_from_aggregate=True,
    original_heading_p95_deg=float(np.percentile(yaw_error[1:], 95)),
    native_retarget_leg_rmse_rad=float(np.sqrt(np.mean(leg_error[1:]**2))),
    full_source_qualified=False,
    input_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
        reference_path, bundle / 'pico/original29.npz', bundle / 'contract.json', Path(__file__),
        OLD / 'mjbatch_full_v1/pico_v4_native323_freshseed_allmargin_5iter_full_v1/trace.npz',
        NEW / 'hard_feasibility_continuation_3740_v1/trace.npz')}
), indent=2) + '\n')
print(output / 'recovery.png')
