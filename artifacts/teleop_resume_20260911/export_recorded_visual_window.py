"""Extract an explicitly labelled recorded-state window for visualization only."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(args):
    producer = json.loads(args.report.read_text())
    original_sha = digest(args.trace)
    assert producer['trace_sha256'] == original_sha
    assert 0 <= args.start_control < args.stop_control <= producer['completed_controls']
    with np.load(args.trace, allow_pickle=False) as archive:
        counts = archive['physics_substeps']
        assert np.all(counts == 10)
        states = archive['physics_qpos']
        assert len(states) == len(counts) * 10 + 1
        q = states[args.start_control * 10:args.stop_control * 10 + 1].copy()
        assert np.isfinite(q).all()
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output / 'visual_window.npz', physics_qpos=q)
    receipt = dict(
        kind='recorded_state_visual_window_only_not_controller_qualification',
        original_requested_controls=producer['requested_controls'],
        original_completed_controls=producer['completed_controls'],
        original_failure=producer['failure']['kind'] if producer['failure'] else None,
        original_full_source_completed=producer['full_source_completed'],
        global_start_control=args.start_control,
        global_stop_control=args.stop_control,
        included_physics_steps=len(q) - 1,
        dynamics_executed=False,
        full_source_qualified=False,
        input_hashes={str(p): digest(p) for p in (args.trace, args.report, Path(__file__))},
        output_sha256=digest(args.output / 'visual_window.npz'),
    )
    (args.output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    (args.output / 'source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(receipt))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('trace', 'report', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('start-control', 'stop-control'):
        parser.add_argument('--' + name, type=int, required=True)
    main(parser.parse_args())
