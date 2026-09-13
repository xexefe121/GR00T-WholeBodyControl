"""Counterexamples run only through mocked file reads; never evaluated in physics."""
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import gear_sonic.scripts.evaluate_g1_true23_bfmzero as evaluator


class Archive(dict):
    @property
    def files(self):
        return list(self)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def main():
    base, timeline, path = evaluator.load_motion('walk003')
    rows = []
    for key, value, label in (('joint_vel', 1e6, 'impossible joint velocity'),
                              ('body_lin_vel_w', 1e6, 'impossible body linear velocity'),
                              ('body_ang_vel_w', 1e6, 'impossible body angular velocity'),
                              ('joint_pos', 1e6, 'out of range and FK-inconsistent joints'),
                              ('joint_vel', np.nan, 'nonfinite joint velocity')):
        candidate = Archive({k: v.copy() for k, v in base.items()})
        candidate[key][...] = value
        with patch.object(evaluator, 'load_motion', return_value=(base, timeline, path)), patch.object(evaluator.np, 'load', return_value=candidate):
            try:
                evaluator.load_case_motion('walk003', 'never_opened_mock_counterexample.npz')
                accepted, reason = True, None
            except ValueError as error:
                accepted, reason = False, str(error)
        rows.append(dict(label=label, field=key, accepted=accepted, reason=reason))
    result = dict(kind='load_case_motion_validation_mocked_counterexamples', physics_run=False, output_npz_created=False, cases=rows)
    output = evaluator.ROOT / 'artifacts/teleop_six_hour_20260910/intent_motion_validation_counterexamples.json'
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
