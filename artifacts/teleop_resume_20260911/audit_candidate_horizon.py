"""Independent continuous native replay of a saved private target proposal."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys


def run(args):
    sys.path.insert(0, str(args.frozen_repo))
    import mujoco
    import numpy as np
    from gear_sonic.utils.g1_true23_mjbatch_mpc import load_native_bundle, sha256

    assert mujoco.__version__ == '3.2.3'
    module_spec = importlib.util.spec_from_file_location('independent_referee', args.oracle)
    oracle = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(oracle)
    native, contract, *_ = load_native_bundle(args.bundle, args.clip)
    with np.load(args.candidate, allow_pickle=False) as archive:
        targets = archive['targets'].copy()
        integration = archive['initial_integration'].copy()
        state_spec = mujoco.mjtState(int(archive['integration_state_spec']))
    if args.requested_controls <= 0 or targets.shape != (args.requested_controls, 23):
        raise ValueError('candidate must cover the explicit requested horizon')
    args.output.mkdir(parents=True, exist_ok=False)
    data = mujoco.MjData(native)
    mujoco.mj_setState(native, data, integration, state_spec)
    mujoco.mj_forward(native, data)
    mujoco.mj_setState(native, data, integration, state_spec)
    report, trace = oracle.inspect_native_segment(native, data, targets, contract)
    report.update(
        kind='independent_saved_private_proposal_native_replay',
        requested_controls=args.requested_controls,
        full_horizon_physical_pass=bool(report['feasible']
            and report['physics_steps'] == 10 * args.requested_controls),
        actual_controller_controls_executed=0,
        controller_replanning_validated=False,
        full_source_qualified=False,
        input_hashes={str(path): sha256(path) for path in (
            args.candidate, args.oracle, Path(__file__),
            args.bundle / 'contract.json', args.bundle / 'native_prepared.xml',
            args.bundle / 'prepared_model_arrays.npz',
            args.frozen_repo / 'gear_sonic/utils/g1_true23_mjbatch_mpc.py')},
    )
    np.savez_compressed(args.output / 'trace.npz', targets=targets,
                        initial_integration=integration,
                        integration_state_spec=int(state_spec), **trace)
    report['trace_sha256'] = sha256(args.output / 'trace.npz')
    (args.output / 'oracle_snapshot.py').write_bytes(args.oracle.read_bytes())
    (args.output / 'runner_snapshot.py').write_bytes(Path(__file__).read_bytes())
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: report[key] for key in (
        'full_horizon_physical_pass', 'physics_steps', 'feasible', 'maximum')}), flush=True)
    return 0 if report['full_horizon_physical_pass'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('frozen-repo', 'oracle', 'bundle', 'candidate', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--clip', required=True, choices=('pico', 'walk002', 'walk003', 'walk008'))
    parser.add_argument('--requested-controls', type=int, required=True)
    raise SystemExit(run(parser.parse_args()))
