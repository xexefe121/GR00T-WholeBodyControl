"""Convert a saved restoration integration vector into a root-auditor fixture; no steps."""

import argparse
import json
from pathlib import Path
import sys


def main(args):
    sys.path.insert(0, str(args.frozen_repo))
    import mujoco
    import numpy as np
    from gear_sonic.utils.g1_true23_mjbatch_mpc import load_native_bundle, sha256

    native, *_ = load_native_bundle(args.bundle, args.clip)
    with np.load(args.archive, allow_pickle=False) as a:
        vector = a['initial_integration'].copy()
        spec = int(a['integration_state_spec'])
    assert spec == int(mujoco.mjtState.mjSTATE_INTEGRATION)
    data = mujoco.MjData(native)
    mujoco.mj_setState(native, data, vector, spec)
    roundtrip = np.empty_like(vector)
    mujoco.mj_getState(native, data, roundtrip, spec)
    np.testing.assert_array_equal(vector, roundtrip)
    with np.load(args.matching_trace, allow_pickle=False) as a:
        np.testing.assert_array_equal(data.qpos, a['physics_qpos'][-1])
        np.testing.assert_array_equal(data.qvel, a['physics_qvel'][-1])
        assert data.time == a['physics_time'][-1]
        warning_name = 'physics_warning_number' if 'physics_warning_number' in a else 'physics_warning_counts'
        assert not np.any(a[warning_name][-1])
        assert not np.any(a['physics_warning_lastinfo'][-1])
    args.output.mkdir(parents=True, exist_ok=False)
    out = args.output / 'initial_integration_state.npz'
    np.savez_compressed(out, state_spec=spec, state_vector=vector, qpos=data.qpos.copy(),
        qvel=data.qvel.copy(), qacc_warmstart=data.qacc_warmstart.copy(),
        warning_counts=data.warning.number.copy(), warning_lastinfo=data.warning.lastinfo.copy())
    report = dict(kind='saved_full_integration_to_independent_fixture_no_physics',
        actual_time=float(data.time), no_physics_steps=True,
        full_state_roundtrip_bitexact=True, matching_executed_trace_final_qpos_qvel_time=True,
        archived_warning_counts_and_lastinfo_zero=True,
        input_hashes={str(p): sha256(p) for p in (args.archive, args.matching_trace, Path(__file__),
            args.bundle / 'contract.json', args.bundle / 'native_prepared.xml',
            args.bundle / 'prepared_model_arrays.npz')}, fixture_sha256=sha256(out))
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    (args.output / 'source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('frozen-repo', 'bundle', 'archive', 'matching-trace', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--clip', required=True)
    main(parser.parse_args())
