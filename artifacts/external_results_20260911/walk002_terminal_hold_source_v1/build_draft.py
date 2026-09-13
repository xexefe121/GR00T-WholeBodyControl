"""Prepare an unbound artifact-only continuation; no inference or dynamics."""
from pathlib import Path
import difflib
import hashlib
import json
import shutil

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / 'pico_control_lm_integration_v1/repo'
DEST = HERE / 'repo'
if (HERE.parent / 'walk002_terminal_hold_v1').exists():
    raise SystemExit('execution output exists; source must stay frozen')
shutil.copytree(BASE, DEST, dirs_exist_ok=True)
helper = (HERE.parent / 'pico_terminal_hold_source_v1/repo/gear_sonic/utils/pico_terminal_continuity.py').read_text()
helper = helper.replace('"438e6420685666b104d9f944620bc9df6745763727848d8709fb223eb47f36cb"', '"UNBOUND_MAIN_TRACE"')
helper = helper.replace('"7741b36128e30c342ad0cf79a2b73cbd2b7697b5b714dde84eed8ed219d131a7"', '"UNBOUND_ROOT_ENDPOINT"')
helper = helper.replace('6500', '1400').replace('6530', '1417').replace('6495', '1395')
helper = helper.replace('clip="pico"', 'clip="walk002"')
helper = helper.replace('compared_controls=30, compared_physics_steps=300', 'compared_controls=17, compared_physics_steps=170')
helper = helper.replace('reexecution_controls=30 if self.boundary_verified', 'reexecution_controls=17 if self.boundary_verified')
helper = helper.replace('import hashlib\n', 'import copy\nimport hashlib\n')
helper = helper.replace('json.dumps(finite_json(plans), indent=2, allow_nan=False))\n        checks = {}',
                        'json.dumps(finite_json(self.tail_plan_summary(plans)), indent=2, allow_nan=False))\n        checks = {}')
anchor = '    def verify_boundary(self, trace, plans, data, fresh, warm, expected_time):'
methods = '''    def tail_plan_summary(self, plans):
        result = copy.deepcopy(plans)
        if not result or result[-1]["control"] != 1415 or result[-1]["controls_executed"] != 2:
            raise ValueError("tail must end after two actual controls of plan1415")
        result[-1]["prospective_commit_controls"] = result[-1]["controls_committed"]
        result[-1]["controls_committed"] = 2
        return result

    def pending_commit(self, plans, planned_states, planned_targets, gains, local, count, completed):
        # These are the live outputs of the existing1415 solve, never a new solve at1417.
        if (completed, local, count) != (1417, 2, 5):
            raise ValueError("terminal boundary must preserve plan1415 locals2..4")
        if not plans or plans[-1]["control"] != 1415 or plans[-1]["controls_executed"] != 2:
            raise ValueError("wrong pending plan or executed count")
        if (np.asarray(planned_states).shape, np.asarray(planned_targets).shape, np.asarray(gains).shape) != ((31, 59), (30, 23), (30, 23, 58)):
            raise ValueError("pending full H30 plan schema mismatch")
        atomic(self.args.output / "pending_plan1415.npz", dict(
            planned_states=planned_states.copy(), planned_targets=planned_targets.copy(), gains=gains.copy(),
            origin_plan_control=np.asarray(1415), boundary_control=np.asarray(1417),
            pending_local_first=np.asarray(local), pending_controls=np.asarray(count - local),
            execution_started=np.asarray(False)))
        result = copy.deepcopy(plans[-1])
        result.update(control=1417, origin_plan_control=1415, pending_local_first=2,
                      origin_planned_commit_controls=5, controls_committed=3, controls_executed=0,
                      imminent_control_checks=[], source_first_state_frame=1427,
                      origin_solve_ms=result["solve_ms"], solve_ms=0.,
                      solve_reused_across_terminal_boundary=True)
        return result

'''
assert helper.count(anchor) == 1
helper = helper.replace(anchor, methods + anchor)
(DEST / 'gear_sonic/utils/walk002_terminal_continuity.py').write_text(helper)
original = BASE / 'gear_sonic/scripts/evaluate_g1_true23_mjbatch_mpc.py'
target = DEST / 'gear_sonic/scripts/evaluate_g1_true23_mjbatch_mpc.py'
text = original.read_text()
edits = []


def replace(old, new):
    global text
    if text.count(old) != 1:
        raise ValueError('expected unique source anchor: ' + old[:100])
    text = text.replace(old, new)
    edits.append(dict(old=old, new=new))


replace('import numpy as np\n', 'import numpy as np\n\nfrom gear_sonic.utils.walk002_terminal_continuity import Continuity, START, BOUNDARY, EXTENSION\n')
replace('    seed_preview = max(\n', '    continuity = Continuity(args, native, fresh_seed, timeline)\n    requested = BOUNDARY + EXTENSION\n    seed_preview = max(\n')
replace('    (args.output / "request.json").write_text(json.dumps(request, indent=2))',
        '    request.update(continuity.request_fields())\n    (args.output / "request.json").write_text(json.dumps(request, indent=2))')
replace('    data = mujoco.MjData(native)\n    data.qpos[:], data.qvel[:] = reference_states[10, :30], reference_states[10, 30:]\n    mujoco.mj_forward(native, data)',
        '    data = continuity.initialize()')
replace('            "source_frame",\n', '            "source_frame",\n            "global_control",\n')
replace('    warm_targets = None\n    completed = 0\n    next_checkpoint = args.checkpoint_controls',
        '    warm_targets = continuity.warm.copy()\n    completed = START\n    next_checkpoint = BOUNDARY + args.checkpoint_controls')
replace('    expected_physics_time = float(data.time)\n', '    expected_physics_time = continuity.expected_time\n')
replace('        for local in range(count):\n', '''        for local in range(count):
            if completed == BOUNDARY and not continuity.boundary_verified:
                try:
                    pending_plan = continuity.pending_commit(
                        plans, planned_states, planned_targets, gains, local, count, completed)
                    trace = continuity.verify_boundary(
                        trace, plans, data, fresh_seed, warm_targets, expected_physics_time)
                except Exception as error:
                    failure = dict(kind="tail_boundary_mismatch", message=str(error),
                                   control=completed, extension_controls_executed=0)
                    break
                plans = [pending_plan]
                restoration_events = []
''')
replace('            frame = completed + 11\n', '            frame = min(completed + 11, len(reference_states) - 1)\n')
replace('                source_first_state_frame=completed + 10,\n',
        '                source_first_state_frame=min(completed + 10, len(reference_states) - 1),\n')
replace('                source_frame=frame,\n', '                source_frame=frame,\n                global_control=completed,\n')
start = text.index('            source_slice = slice(phase["control_start"], min(completed, phase["control_stop"]))')
stop = text.index('            next_checkpoint = completed + args.checkpoint_controls', start)
replace(text[start:stop], '''            continuity.checkpoint(trace, plans, data, fresh_seed, warm_targets,
                                  expected_physics_time, completed, finite_json(failure))
''')
start = text.index('    arrays = {name: np.asarray(values) for name, values in trace.items()}')
stop = text.index('\n\n\nif __name__ == "__main__":', start)
replace(text[start:stop], '''    return continuity.finalize(trace, plans, data, fresh_seed, warm_targets,
                               expected_physics_time, completed, finite_json(failure),
                               finite_json(restoration_events))''')
replace('    parser.add_argument("--output", type=Path, required=True)\n    run(parser.parse_args())', '''    parser.add_argument("--preceding-run", type=Path, required=True)
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
(HERE / 'adapter.diff').write_text(''.join(difflib.unified_diff(original.read_text().splitlines(True), text.splitlines(True),
                                      fromfile=str(original), tofile=str(target))))
(HERE / 'source_transformations.json').write_text(json.dumps(edits, indent=2))
files = list(DEST.rglob('*.py'))
receipt = dict(kind='PREPARE_ONLY_UNBOUND', execution_authorized=False, root_main_audit_pending=True,
               main_trace_binding='UNBOUND_MAIN_TRACE', root_endpoint_binding='UNBOUND_ROOT_ENDPOINT',
               shared_source_edited=False, source_evaluator_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
               hashes={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files})
(HERE / 'draft_source_hashes.json').write_text(json.dumps(receipt, indent=2))
print(json.dumps(dict(files=len(files), evaluator_sha256=receipt['hashes'][str(target)], status='UNBOUND; no execution')))
