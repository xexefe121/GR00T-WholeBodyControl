"""Artifact-local provenance helpers; no model or physics initialization."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

BASE = Path(__file__).resolve().parent.parent
NEW = BASE.parent
OLD = NEW.parent / 'sonic23_teleop_six_hour_20260910'
KEYS = ('actions', 'base_ang_vel', 'dof_pos', 'dof_vel', 'projected_gravity')
ROWS = 3057
AXES = 23
SIGNS = (-1., 1.)
ACTOR_BUDGET = 143679
BACKWARD_BUDGET = 3057

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def local(path):
    value = str(path).replace('\\', '/')
    if sys.platform != 'win32' and len(value) > 2 and value[1] == ':':
        value = '/mnt/' + value[0].lower() + value[2:]
    return Path(value)

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')

def archive(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}

def exact(actual, expected, name=''):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or actual.dtype != expected.dtype or actual.tobytes() != expected.tobytes():
        raise AssertionError('byte parity: '+name+' '+str((actual.shape,actual.dtype,expected.shape,expected.dtype)))

def assert_frozen():
    receipt = read(BASE/'generation_frozen_inputs.json')
    for name, digest in receipt['source_sha256'].items():
        assert sha(Path(__file__).parent/name) == digest, name
    for name, digest in receipt['input_sha256'].items():
        assert sha(local(name)) == digest, name
    return receipt

def assert_clearance(path):
    receipt = assert_frozen()
    clearance = read(path)
    assert clearance['approved'] is True
    assert clearance['generation_only'] is True
    assert clearance['frozen_receipt_sha256'] == sha(BASE/'generation_frozen_inputs.json')
    assert clearance['launcher_sha256'] == sha(BASE/'run_generation_durable.ps1')
    review = local(clearance['review_path'])
    assert sha(review) == clearance['review_sha256']
    assert clearance['selected_proposal_sha256'] == receipt['selected_proposal_sha256']
    return receipt, clearance

def dataset_configs():
    return [dict(name='old', labels=NEW/'fast_controller_nominal_pilot_v1/labels/labels.npz',
                 trace=OLD/'bfm_online_intent_v2/walk003_terminal_bfm_yaw4_hybrid_v1/trace.npz'),
            dict(name='query1', labels=NEW/'fresh_expert_labels_resume_v1/labels/labels.npz',
                 trace=NEW/'student_actual_oracle_control1_resume1001_v1/nominal/trace.npz'),
            dict(name='query250', labels=NEW/'bfm_entry250_labels_v1/labels/labels.npz',
                 trace=NEW/'bfm_entry250_actual_oracle_v1/nominal/trace.npz')]
