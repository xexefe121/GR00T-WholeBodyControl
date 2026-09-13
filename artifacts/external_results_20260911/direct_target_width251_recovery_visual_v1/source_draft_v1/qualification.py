"""Saved-report gates only; importing this module never opens a task array."""
import hashlib
import json
import ntpath
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
NEW = BASE.parent
ROOT = Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof')
RECOVERY = NEW / 'direct_target_width512_expert_recovery_v2'
BUNDLE = ROOT / 'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1'
FIXED_RECEIPTS = {
    'owner': '3160652e153c08769671a639f569449d1ed2a776f83ddffe905dd4d3f65aa7e1',
    'main_physics': '9904a031f8197b5414ee08e43782e816f5aeecf40b05b8a8f8451e7be664b239',
    'main_intent': '7d5cc86822a609d2c7d443072c4abf009bd1176caddebe93b362f05513746b3a',
    'hold_physics': '1d963fcf4635bc53b0c9fd1df3939a969ee0c46ea66e3608504c43f540672856',
    'hold_intent': '78eca0bddb2782755b7dbaa7b1fc26cffddf7251985957312b00a733f114e038',
}
EXACT_FIELDS = (
    'all_qpos_bitexact', 'all_qvel_bitexact', 'all_command_torque_bitexact',
    'physics_actuator_force_bitexact', 'all_physics_time_bitexact',
    'physics_warning_number_bitexact', 'physics_warning_lastinfo_bitexact',
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def normalize(path):
    text = str(path).replace('\\', '/')
    match = re.match(r'^/mnt/([a-zA-Z])/(.*)$', text)
    if match:
        text = match[1] + ':/' + match[2]
    require(re.match(r'^[a-zA-Z]:/', text), 'Expected absolute Windows/WSL path')
    return ntpath.normpath(text).replace('\\', '/').casefold()


def bound(mapping, subject):
    normalized = {}
    for path, digest in mapping.items():
        key = normalize(path)
        require(key not in normalized or normalized[key] == digest, 'Conflicting path aliases')
        normalized[key] = digest
    require(normalized.get(normalize(subject['path'])) == subject['sha256'],
            'Missing or changed direct subject: ' + str(subject['path']))


def exact_int(mapping, key, expected):
    require(type(mapping.get(key)) is int and mapping[key] == expected, key)


def validate_reports(reports, subjects):
    """Pure metadata contract, usable with synthetic dictionaries."""
    owner = reports['owner']
    for key in ('completion_accounting_passed', 'requested_recovery_completed',
                'raw_exit_known', 'all_postrun_pins_exact', 'processes_absent'):
        require(owner.get(key) is True, 'Owner ' + key)
    for key in ('raw_python_exit_code', 'exit_code'):
        exact_int(owner, key, 0)
    require(owner.get('accounting_uncertainty') == [], 'Owner uncertainty')
    for role in ('recovery_request', 'main_trace', 'main_report', 'hold_trace', 'hold_report'):
        bound(owner['output_sha256'], subjects[role])
    for role in ('reference', 'bundle_manifest', 'contract', 'native_xml', 'native_arrays'):
        bound(owner['input_sha256'], subjects[role])
    for segment, count, start in (('main', 1569, 0), ('hold', 250, 1569)):
        producer = reports[segment + '_report']
        trace = subjects[segment + '_trace']
        require(producer.get('full_segment_completed') is True, segment + ' producer incomplete')
        require(producer.get('failure') is None, segment + ' producer failure')
        for key in ('requested_controls', 'completed_controls', 'completed_full_controls'):
            exact_int(producer, key, count)
        exact_int(producer, 'physics_steps', count * 10)
        exact_int(producer, 'partial_substeps', 0)
        require(producer.get('trace_sha256') == trace['sha256'], segment + ' producer trace')
        physics = reports[segment + '_physics']
        for key in ('independent_segment_pass', 'recorded_trace_reproduced_through_last_sample',
                    'requested_segment_completed'):
            require(physics.get(key) is True, segment + ' physics ' + key)
        exact_int(physics, 'intended_segment_controls', count)
        exact_int(physics, 'physics_steps', count * 10)
        exact_int(physics, 'compared_physics_steps', count * 10)
        exact_int(physics, 'recorded_partial_substeps', 0)
        exact_int(physics, 'private_replay_steps_beyond_recorded_prefix', 0)
        require(physics.get('first_failure') is None, segment + ' strict failure')
        for key in EXACT_FIELDS:
            require(physics['original_trace_comparison'].get(key) is True, segment + ' ' + key)
        for role in (segment + '_trace', segment + '_report', 'contract', 'native_xml', 'native_arrays'):
            bound(physics['input_hashes'], subjects[role])
        intent = reports[segment + '_intent']
        for key in ('independent_physical_pass', 'intended_segment_completed', 'requested_segment_quiet_pass'):
            require(intent.get(key) is True, segment + ' intent ' + key)
        exact_int(intent, 'requested_controls', count)
        exact_int(intent, 'recorded_controls', count)
        exact_int(intent, 'global_start', start)
        for role in (segment + '_trace', segment + '_physics', 'reference', 'reference_receipt',
                     'bundle_manifest', 'contract', 'native_xml', 'native_arrays'):
            bound(intent['hashes'], subjects[role])
    main = reports['main_intent']
    require(main.get('full_lifecycle_source_intent_pass') is True, 'Full source intent')
    exact_int(main['source_metrics'], 'source_controls', 819)
    bound(reports['hold_intent']['hashes'], subjects['main_trace'])
    return {'offline_recovery_full_scope_qualified': True, 'fast_student_qualified': False,
            'requested_main_controls': 1569, 'requested_continuous_hold_controls': 250}


def paths_and_pins():
    """Called only by the explicitly selected renderer, before any native import."""
    paths = {
        'owner': RECOVERY / 'owner_completion.json',
        'recovery_request': RECOVERY / 'execution_request.json',
        'main_trace': RECOVERY / 'nominal/trace.npz', 'main_report': RECOVERY / 'nominal/report.json',
        'hold_trace': RECOVERY / 'post_lifecycle_hold_5s/trace.npz',
        'hold_report': RECOVERY / 'post_lifecycle_hold_5s/report.json',
        'reference': Path('E:/codex-artifacts/sonic23_teleop_six_hour_20260910/mjbatch_intent_floor_inputs_v1/walk003/reference.npz'),
        'bundle_manifest': BUNDLE / 'manifest.json', 'contract': BUNDLE / 'contract.json',
        'native_xml': BUNDLE / 'native_prepared.xml', 'native_arrays': BUNDLE / 'prepared_model_arrays.npz',
        'font': Path('C:/Windows/Fonts/segoeui.ttf'),
        'renderer_source': BASE / 'source_draft_v1/render_contact_sheet.py',
        'qualification_source': Path(__file__).resolve(),
    }
    paths['reference_receipt'] = paths['reference'].parent / 'portable_receipt.json'
    for role in ('main_physics', 'main_intent', 'hold_physics', 'hold_intent'):
        paths[role] = NEW / ('direct_target_width251_expert_' + role + '_v1/report.json')
    subjects = {role: {'path': p.as_posix(), 'sha256': sha(p)} for role, p in paths.items()}
    for role, digest in FIXED_RECEIPTS.items():
        require(subjects[role]['sha256'] == digest, 'Changed fixed receipt: ' + role)
    roles = ('owner', 'main_report', 'hold_report', 'main_physics', 'main_intent', 'hold_physics', 'hold_intent')
    reports = {role: json.loads(paths[role].read_text()) for role in roles}
    validate_reports(reports, subjects)
    reference = json.loads(paths['reference_receipt'].read_text())
    require(reference['reference_sha256'] == subjects['reference']['sha256'], 'Declared reference')
    manifest = json.loads(paths['bundle_manifest'].read_text())
    require(manifest['portable_xml_sha256'] == subjects['native_xml']['sha256'], 'Native XML')
    require(manifest['prepared_arrays_sha256'] == subjects['native_arrays']['sha256'], 'Native arrays')
    pins = {s['path']: s['sha256'] for s in subjects.values()}
    for name, digest in manifest['meshes'].items():
        path = (BUNDLE / 'meshes' / name).resolve()
        require(path.is_relative_to((BUNDLE / 'meshes').resolve()), 'Mesh outside bundle')
        require(sha(path) == digest, 'Changed mesh: ' + name)
        pins[path.as_posix()] = digest
    return paths, pins
