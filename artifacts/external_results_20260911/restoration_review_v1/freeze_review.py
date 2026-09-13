"""Freeze inspected source bytes and compare them with the active run receipt."""
from pathlib import Path
import hashlib
import json

repo = Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof')
out = Path(__file__).resolve().parent
run = out.parent / 'hard_feasibility_restoration_continuation_3740_v1'
request = json.loads((run / 'request.json').read_text())
names = ['utils/g1_true23_mjbatch_restoration.py',
         'scripts/continue_g1_true23_mpc_hard_feasibility.py',
         'utils/g1_true23_mjbatch_mpc.py',
         'utils/g1_true23_mjbatch_ilqr_core.py',
         'utils/g1_true23_feasibility_referee.py',
         'utils/g1_true23_mjbatch_bfm_seed.py',
         'utils/g1_true23_mjbatch_model.py']
rows = []
for name in names:
    path = repo / 'gear_sonic' / name
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    matching = [(k, v) for k, v in request['input_hashes'].items()
                if k.replace('\\', '/').endswith('/gear_sonic/' + name)]
    assert len(matching) == 1
    frozen = run / (path.stem + '_snapshot.py')
    frozen_digest = hashlib.sha256(frozen.read_bytes()).hexdigest()
    (out / path.name).write_bytes(content)
    rows.append(dict(path=str(path), sha256=digest,
                     active_request_sha256=matching[0][1],
                     active_snapshot_sha256=frozen_digest,
                     all_three_match=digest == matching[0][1] == frozen_digest))
assert all(row['all_three_match'] for row in rows)
(out / 'reviewed_source_hashes.json').write_text(json.dumps(rows, indent=2))
clock = request['initial_simulation_time']
start = clock
for _ in range(1000):
    clock += .002
summary = dict(all_seven_sources_match_active_frozen_run=True,
               requested_controls=request['requested_controls'],
               requested_physics_steps=request['requested_controls'] * 10,
               clock_addition_vs_formula_at_two_seconds=clock-(start+2.),
               no_planner_calls=True, no_physics_calls=True)
(out / 'read_only_checks.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
