"""Read-only supplemental audit of the frozen v4 floor reference pack."""
import hashlib
import json
from pathlib import Path

import numpy as np

BASE = Path('E:/codex-artifacts/sonic23_teleop_six_hour_20260910/mjbatch_intent_floor_inputs_v1')
OUT = BASE.parent / 'intent_floor_v4_supplement_v1'
OUT.mkdir(exist_ok=False)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(x):
    return dict(min=float(np.min(x)), median=float(np.median(x)), p95=float(np.percentile(x, 95)), max=float(np.max(x)))


results = {}
for clip in ('pico', 'walk002', 'walk003', 'walk008'):
    folder = BASE / clip
    receipt = json.loads((folder / 'portable_receipt.json').read_text())
    transform = json.loads((folder / 'floor_transform_receipt.json').read_text())
    assert sha(folder / 'reference.npz') == receipt['reference_sha256']
    assert sha(folder / 'floor_transform_receipt.json') == receipt['reference_floor_transform']['transform_receipt_sha256']
    assert sha(folder / 'frame_lift.npz') == transform['frame_lift_sha256']
    timeline = json.loads((folder / 'original_timeline.json').read_text())
    with np.load(folder / 'frame_lift.npz') as z:
        lift = z['frame_lift_m']
        before = z['before_foot_clearance_m']
        after = z['after_foot_clearance_m']
    phase_rows = []
    for phase in timeline['phases']:
        a, b = phase['frame_start'], phase['frame_stop']
        old, new = before[a:b], after[a:b]
        phase_rows.append(dict(name=phase['name'], frame_start=a, frame_stop=b, frames=b-a,
            lift_m=stats(lift[a:b]), before_minimum_foot_clearance_m=stats(old.min(axis=1)),
            after_minimum_foot_clearance_m=stats(new.min(axis=1)),
            before_both_feet_clear_2mm_frames=int(np.sum(old.min(axis=1) > .002)),
            after_both_feet_clear_2mm_frames=int(np.sum(new.min(axis=1) > .002)),
            before_near_floor_2mm_transitions=[int(np.count_nonzero(np.diff(old[:, i] <= .002))) for i in (0, 1)],
            after_near_floor_2mm_transitions=[int(np.count_nonzero(np.diff(new[:, i] <= .002))) for i in (0, 1)]))
    results[clip] = dict(reference_sha256=sha(folder / 'reference.npz'), portable_receipt_sha256=sha(folder / 'portable_receipt.json'),
        transform_receipt_sha256=sha(folder / 'floor_transform_receipt.json'), phases=phase_rows,
        original29_world_root_error_p95_m=transform['original29_world_root_error_p95_m'],
        original29_world_root_error_max_m=transform['original29_world_root_error_max_m'],
        added_vertical_speed_max_mps=transform['added_vertical_velocity_max_mps'],
        added_vertical_acceleration_max_mps2=transform['added_vertical_acceleration_max_mps2'],
        guard_frames=transform['filter']['guard_active_frames'])

report = dict(kind='frozen_v4_floor_reference_scope_supplement', producer_sha256=sha(Path(__file__)), clips=results,
    original_sources_changed=False, completed_pack_files_changed=False,
    contact_interpretation='Near-floor transitions use geometric sphere clearance <=2mm, not force/contact labels. The 20mm buffer removes nearly all such transitions. This is nonpenetration, not contact feasibility or ballistic flight.',
    causality='Lift pose recurrence uses current/past required clearances only; central-difference exported Z velocity needs the following pose (20ms at50Hz). Parameters selected using all four supplied recordings; not held-out validation.',
    dynamic_tracking_qualified=False, hardware_authorized=False)
(OUT / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
(OUT / 'producer_snapshot.py').write_bytes(Path(__file__).read_bytes())
lines = [
    '# Explicit v4 floor-reference adaptation', '',
    'These frozen reference packs eliminate native foot-sphere penetration in the pose reference. They do not demonstrate stable physical tracking, valid support contacts, collision-free motion, real-time execution, or robot readiness.', '',
    'The transformation adds one nonnegative Z lift to the root and every body at each original 50Hz frame. Joint positions/velocities, quaternions/angular velocities, XY motion, full source timing, and original source files stay byte-identical where applicable. Relative hand/head/foot geometry stays unchanged. The physical floor stays at its original height. Existing self-collisions stay unchanged.', '',
    'A critically damped filter with omega=15/s processes the minimum required foot-sphere lift. Every recording uses the same fixed20mm buffer and explicit max(required+1micrometre, filtered+20mm) clearance guard. Parameters were selected using these four recordings. No guard activated on these recordings. The algorithm does not guarantee bounded derivatives on unseen motions: the clearance guard can introduce spikes.', '',
    'The lift recurrence has zero future-pose support and passed fixed-parameter prefix checks. Exported body Z linear velocity retains the required central-difference convention and therefore needs one following pose,20ms. An online implementation must account for that buffer; the entire position-and-velocity pipeline is not strictly zero-lookahead.', '',
    '| Clip | Source controls | Max lift (mm) | Max added speed (m/s) | Max added acceleration (m/s²) | Max original world-root error (mm) |',
    '|---|---:|---:|---:|---:|---:|',
]
for clip, item in results.items():
    source = next(x for x in item['phases'] if x['name'] == 'source_motion')
    maximum_lift = max(x['lift_m']['max'] for x in item['phases'])
    lines.append(f"| {clip} | {source['frames']} | {maximum_lift*1000:.3f} | {item['added_vertical_speed_max_mps']:.3f} | {item['added_vertical_acceleration_max_mps2']:.3f} | {item['original29_world_root_error_max_m']*1000:.3f} |")
lines += ['',
    'The20mm buffer raises the standing foot clearance from about3mm to23mm. It also lifts already-clear frames. In source motion, the near-floor2mm transition counts change as follows; this is a material reference adaptation and is not hidden by alignment:', '',
    '| Clip | Left transitions before→after | Right transitions before→after | Both feet >2mm clear, before→after |',
    '|---|---:|---:|---:|',
]
for clip, item in results.items():
    source = next(x for x in item['phases'] if x['name'] == 'source_motion')
    old, new = source['before_near_floor_2mm_transitions'], source['after_near_floor_2mm_transitions']
    lines.append(f"| {clip} | {old[0]}→{new[0]} | {old[1]}→{new[1]} | {source['before_both_feet_clear_2mm_frames']}→{source['after_both_feet_clear_2mm_frames']} / {source['frames']} |")
lines += ['',
    'All previously geometrically clear frames remain clear; no positive foot height was lowered. Neither the original nor transformed arrays provide contact labels. The buffer removes nearly all near-floor transitions, so support/contact fidelity must be assessed in physical simulation rather than inferred from this pose validation.', '',
    'The producer report original-world/root-relative hand/head p95 values and transform root-error values cover all saved frames, including initialization and return; they are reference geometry metrics, not actual simulated tracking metrics. This supplemental JSON provides exact per-phase clearance counts and binds frozen reference/receipt hashes.', '',
    'Root load_case_motion checked all10,674 frames in both raw and portable copies: full timing, native joint and adjacent-speed bounds, finite fields/unit quaternions, FK, and derivative conventions. A separate compact-agent native collision audit is required before the floor claim is independently confirmed. No controller or physical plant was changed by this pack.', '',
    'Portable pack: E:/codex-artifacts/sonic23_teleop_six_hour_20260910/mjbatch_intent_floor_inputs_v1. Every clip contains the byte-preserved v3 input, transformed reference, lift array, original timeline, model/physics provenance, transform receipt, and portable receipt. All existing pack files remain untouched by this supplement.', '',
]
(OUT / 'README.md').write_text('\n'.join(lines), encoding='utf-8')
print(json.dumps(dict(output=str(OUT), clips=list(results)), indent=2))
