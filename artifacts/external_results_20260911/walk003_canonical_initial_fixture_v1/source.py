"""Export independently reconstructed canonical initial state for native replay."""

import argparse
import json
from pathlib import Path
import sys


def main(args):
    sys.path.insert(0, str(args.frozen_repo))
    import mujoco
    import numpy as np
    from gear_sonic.utils.g1_true23_mjbatch_mpc import (
        load_motion_override, load_native_bundle, motion_states, sha256,
    )

    assert mujoco.__version__ == '3.2.3'
    model, contract, original, timeline, manifest = load_native_bundle(args.bundle, args.clip)
    motion, receipt = load_motion_override(args.reference, args.bundle, args.clip, model,
                                           contract, original, timeline, manifest)
    states = motion_states(motion)
    data = mujoco.MjData(model)
    data.qpos[:], data.qvel[:] = states[10, :30], states[10, 30:]
    mujoco.mj_forward(model, data)
    assert data.time == 0 and not np.any(data.warning.number)
    state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(model, state_spec))
    mujoco.mj_getState(model, data, state, state_spec)
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / 'initial_integration_state.npz'
    np.savez_compressed(path, state_spec=int(state_spec), state_vector=state,
                        qpos=data.qpos.copy(), qvel=data.qvel.copy(),
                        qacc_warmstart=data.qacc_warmstart.copy())
    report = dict(
        kind='independent_canonical_initial_state_fixture', clip=args.clip,
        source_frame=10, canonical_control=0, physics_steps_executed=0,
        initialization='native MjData; declared motion state at frame10; native mj_forward once',
        full_source_qualified=False, reference_receipt=receipt,
        state_spec=int(state_spec), state_vector_length=len(state),
        fixture_sha256=sha256(path),
        input_hashes={str(p): sha256(p) for p in (
            args.reference, Path(__file__), args.bundle / 'contract.json',
            args.bundle / 'native_prepared.xml', args.bundle / 'prepared_model_arrays.npz',
            args.frozen_repo / 'gear_sonic/utils/g1_true23_mjbatch_mpc.py')},
    )
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    (args.output / 'source.py').write_bytes(Path(__file__).read_bytes())
    print(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('frozen-repo', 'bundle', 'reference', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--clip', required=True, choices=('pico', 'walk002', 'walk003', 'walk008'))
    main(parser.parse_args())
