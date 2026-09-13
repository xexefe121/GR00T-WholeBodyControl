"""Freeze concrete inputs for the root-selected collection, without inference."""
import argparse
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).parent
NEW = BASE.parent
ROOT = Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof')
OLD = Path('E:/codex-artifacts/sonic23_teleop_six_hour_20260910')
BUNDLE = ROOT / 'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1'
ONNX = ROOT / 'artifacts/teleop_six_hour_20260910/bfm_onnx_v2'
SOURCE = BASE / 'source_snapshot_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def win(path):
    text = str(path).replace('\\', '/')
    if text.startswith('/mnt/'):
        text = text[5].upper() + ':' + text[6:]
    p = Path(text)
    return p if p.is_absolute() else ROOT / p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--final', action='store_true')
    args = parser.parse_args()
    assert not (BASE / 'collection').exists()
    pins = {}
    def pin(path, expected=None):
        p = win(path)
        assert 'walk008' not in str(p).lower()
        digest = sha(p)
        if expected is not None:
            assert digest == expected, str(p)
        pins[str(p).replace('\\', '/')] = digest
        return str(p).replace('\\', '/')

    cases = []
    for clip, total, stop, source_count, qualified, snapdir, snapshot_sha, history, previous in (
        ('pico', 6530, 6230, 5780, 'pico_full_root_qualification_v1', 'pico_all_control_snapshots_independent_v1',
         '81775928dc211394a13150da42047da2ad72cd14fa100daa57576bbf25ca210e',
         'fresh_seed_measured_history', 'fresh_seed_previous_action'),
        ('walk002', 1417, 1117, 667, 'walk002_hybrid_root_qualification_v1', 'walk002_hybrid_all_control_snapshots_independent_v1',
         'c653576b73449f607fc8bffede1594b9851c456820795c58b6d2c6b8c663e724', 'history', 'previous_action')):
        qp = NEW / qualified / 'qualification.json'
        qual = read(qp)
        case = dict(clip=clip, total_controls=total, moving_start=250, moving_stop=stop,
                    selected_rows=stop-250, source_controls=source_count, history_key=history, previous_key=previous,
                    qualification=pin(qp), trace=pin(qual['traces']['full']['path'], qual['traces']['full']['sha256']),
                    timeline=pin(BUNDLE / clip / 'timeline.json'),
                    reference=pin(OLD / 'mjbatch_intent_floor_inputs_v1' / clip / 'reference.npz'),
                    original29=pin(BUNDLE / clip / 'original29.npz'),
                    snapshots=pin(NEW / snapdir / 'control_snapshots.npz', snapshot_sha),
                    snapshot_report=pin(NEW / snapdir / 'report.json'))
        for entry in [*qual['traces'].values(), *qual['independent_reports'].values()]:
            pin(entry['path'], entry['sha256'])
        report = read(NEW / snapdir / 'report.json')
        pin(NEW / snapdir / 'request.json', report['request_sha256'])
        for p, h in report['hashes'].items():
            pin(p, h)
        for name in ('portable_receipt.json', 'report.json'):
            pin(OLD / 'mjbatch_intent_floor_inputs_v1' / clip / name)
        pin(BUNDLE / clip / 'native_original.npz')
        cases.append(case)

    for name in ('manifest.json', 'contract.json', 'native_prepared.xml', 'prepared_model_arrays.npz'):
        pin(BUNDLE / name)
    for name, digest in read(BUNDLE / 'manifest.json')['meshes'].items():
        pin(BUNDLE / 'meshes' / name, digest)
    for name in ('manifest.json', 'actor.onnx', 'backward.onnx'):
        pin(ONNX / name)
    for p in SOURCE.rglob('*.py'):
        pin(p)
    for p in (BASE / 'prepare_sources.py', BASE / 'copied_dependencies.json', Path(__file__),
              NEW / 'pico_walk002_label_coverage_assessment_v1/assessment.json'):
        pin(p)
    if args.final:
        pin(BASE / 'focused_final.xml')
        pin(BASE / 'preflight_draft.json')
    normalization = pin(NEW / 'fast_controller_phase_fit_v1/fit/teacher_fit.npz')
    labels = {}
    for name, folder in (('walk003_old', 'fast_controller_nominal_pilot_v1'),
                         ('walk003_query1', 'fresh_expert_labels_resume_v1'),
                         ('walk003_query250', 'bfm_entry250_labels_v1')):
        labels[name] = pin(NEW / folder / 'labels/labels.npz')
    runtime = Path('E:/codex_sonic_runtime/mjbatch323_20260910/venv/lib/python3.11/site-packages')
    for folder in ('mujoco', 'numpy/core', 'scipy/spatial/transform'):
        for p in (runtime / folder).glob('*.so*'):
            pin(p)
    ort = Path('E:/codex_sonic_runtime/bfm_seed_20260910/onnx_deps/onnxruntime')
    pin(ort / '__init__.py')
    for p in (ort / 'capi').glob('*'):
        if p.is_file() and ('.so' in p.name or p.suffix == '.py'):
            pin(p)
    result = dict(kind='root_selected_two_clip_actual_moving_labels', frozen=args.final,
                  root_selected_extraction=True, fitting_authorized=False, collector_physics_authorized=False,
                  selected_rows=6847, feature_count=1069, baseline_goal='original_h8_position1_yaw2',
                  selection='PICO250..6229 and walk002250..1116; original qualified targets, histories, source indices; full291 root reconstructed snapshots; unchanged old normalization; no walk008 or fitting.',
                  bundle=str(BUNDLE).replace('\\', '/'), onnx=str(ONNX).replace('\\', '/'),
                  onnx_dependencies='E:/codex_sonic_runtime/bfm_seed_20260910/onnx_deps',
                  normalization=normalization, existing_labels=labels, cases=cases, input_hashes=pins)
    path = BASE / ('collector_frozen_inputs.json' if args.final else 'collector_draft_inputs.json')
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(manifest=str(path), sha256=sha(path), input_hashes=len(pins), frozen=args.final)))


if __name__ == '__main__':
    main()
