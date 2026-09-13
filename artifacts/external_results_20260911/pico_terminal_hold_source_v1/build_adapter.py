"""Create a reviewable artifact copy; no shared source edit, inference or dynamics."""
from pathlib import Path
import difflib
import hashlib
import json
import shutil

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "pico_control_lm_integration_v1" / "repo"
DEST = HERE / "repo"
if (HERE.parent / "pico_terminal_hold_v1").exists():
    raise SystemExit("execution output exists; source must stay frozen")
shutil.copytree(BASE, DEST, dirs_exist_ok=True)
original = BASE / "gear_sonic/scripts/evaluate_g1_true23_mjbatch_mpc.py"
target = DEST / "gear_sonic/scripts/evaluate_g1_true23_mjbatch_mpc.py"
text = original.read_text()
edits = []


def replace(old, new):
    global text
    if text.count(old) != 1:
        raise ValueError(f"expected unique source anchor: {old[:100]}")
    text = text.replace(old, new)
    edits.append(dict(old=old, new=new))


replace('import numpy as np\n', 'import numpy as np\n\nfrom gear_sonic.utils.pico_terminal_continuity import Continuity, START, BOUNDARY, EXTENSION\n')
replace('    seed_preview = max(\n', '    continuity = Continuity(args, native, fresh_seed, timeline)\n    requested = BOUNDARY + EXTENSION\n    seed_preview = max(\n')
replace('    (args.output / "request.json").write_text(json.dumps(request, indent=2))',
        '    request.update(continuity.request_fields())\n    (args.output / "request.json").write_text(json.dumps(request, indent=2))')
replace('    data = mujoco.MjData(native)\n    data.qpos[:], data.qvel[:] = reference_states[10, :30], reference_states[10, 30:]\n    mujoco.mj_forward(native, data)',
        '    data = continuity.initialize()')
replace('            "source_frame",\n', '            "source_frame",\n            "global_control",\n')
replace('    warm_targets = None\n    completed = 0\n    next_checkpoint = args.checkpoint_controls',
        '    warm_targets = continuity.warm.copy()\n    completed = START\n    next_checkpoint = BOUNDARY + args.checkpoint_controls')
replace('    expected_physics_time = float(data.time)\n', '    expected_physics_time = continuity.expected_time\n')
replace('    while completed < requested:\n', '''    while completed < requested:
        if completed == BOUNDARY and not continuity.boundary_verified:
            try:
                trace = continuity.verify_boundary(
                    trace, plans, data, fresh_seed, warm_targets, expected_physics_time)
            except Exception as error:
                failure = dict(kind="tail_boundary_mismatch", message=str(error),
                               control=completed, extension_controls_executed=0)
                break
            plans = []
            restoration_events = []
''')
replace('            frame = completed + 11\n', '            frame = min(completed + 11, len(reference_states) - 1)\n')
replace('                source_first_state_frame=completed + 10,\n',
        '                source_first_state_frame=min(completed + 10, len(reference_states) - 1),\n')
replace('                source_frame=frame,\n', '                source_frame=frame,\n                global_control=completed,\n')
start = text.index('            source_slice = slice(phase["control_start"], min(completed, phase["control_stop"]))')
stop = text.index('            next_checkpoint = completed + args.checkpoint_controls', start)
old = text[start:stop]
replace(old, '''            continuity.checkpoint(trace, plans, data, fresh_seed, warm_targets,
                                  expected_physics_time, completed, finite_json(failure))
''')
start = text.index('    arrays = {name: np.asarray(values) for name, values in trace.items()}')
stop = text.index('\n\n\nif __name__ == "__main__":', start)
old = text[start:stop]
replace(old, '''    return continuity.finalize(trace, plans, data, fresh_seed, warm_targets,
                               expected_physics_time, completed, finite_json(failure),
                               finite_json(restoration_events))''')
replace('    parser.add_argument("--output", type=Path, required=True)\n    run(parser.parse_args())',
        '''    parser.add_argument("--preceding-run", type=Path, required=True)
    parser.add_argument("--preceding-endpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    try:
        result = run(parser.parse_args())
    except Exception:
        import traceback
        parsed = parser.parse_args()
        if parsed.output.is_dir() and not (parsed.output / "report.json").exists():
            (parsed.output / "exception.json").write_text(json.dumps(dict(
                kind="incomplete_hold_exception", traceback=traceback.format_exc(),
                success=False, requested_controls=EXTENSION), indent=2))
        raise
    raise SystemExit(0 if result["probe_completed"] else 1)''')
target.write_text(text)
shutil.copyfile(HERE / "continuity.py", DEST / "gear_sonic/utils/pico_terminal_continuity.py")
(HERE / "adapter.diff").write_text(''.join(difflib.unified_diff(original.read_text().splitlines(True), text.splitlines(True),
                                      fromfile=str(original), tofile=str(target))))
(HERE / "source_transformations.json").write_text(json.dumps(edits, indent=2))
files = [path for path in DEST.rglob("*.py")]
receipt = dict(kind="artifact_only_tail_reexecution_and_terminal_hold", shared_source_edited=False,
               source_evaluator_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
               hashes={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files})
(HERE / "frozen_source_hashes.json").write_text(json.dumps(receipt, indent=2))
print(json.dumps(dict(files=len(files), evaluator_sha256=receipt["hashes"][str(target)])))
