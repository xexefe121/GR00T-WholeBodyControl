"""Focused unchanged-path comparison; does not repeat the closed replay study."""
import argparse,json
from pathlib import Path
import numpy as np
from continue_pico_shoulder_recovery import Study


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    base=Path('/mnt/e/codex-artifacts')
    study=Study(base/'native23_preview_repair_20260913/strict_newer_v1')
    with np.load(base/'native23_shoulder_recovery_20260914/continuation_v1/baseline_trace.npz') as z:expected=z['states']
    assert not hasattr(study.controller,'sustained_braking_filter')
    actual=[]
    for control in range(3):
        study.admit(control);result=study.step(control);assert result['failure'] is None
        actual.extend(result['physical'])
    np.testing.assert_array_equal(actual,expected[1:31])
    from gear_sonic.utils.g1_true23_controller_state import wrapper_contract
    assert 'sustained_braking_filter' not in wrapper_contract(study.controller)
    args.output.write_text(json.dumps(dict(passed=True,physics_steps=30,exact=True,filter_enabled=False,
        unchanged_wrapper_when_disabled=True,full_archived_study_repeated=False),indent=2)+'\n')
    print(args.output.read_text())


if __name__=='__main__':main()
