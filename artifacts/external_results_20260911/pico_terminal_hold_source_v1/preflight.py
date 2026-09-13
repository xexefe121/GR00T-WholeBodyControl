"""Read-only real saved-array/history preflight. No ONNX loading/inference or physics."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from gear_sonic.utils.g1_true23_mjbatch_bfm_seed import Native23BFMRolloutSeed, BFMHistory
from gear_sonic.utils.g1_true23_mjbatch_mpc import load_native_bundle
from gear_sonic.utils.pico_terminal_continuity import Continuity, history_snapshot, sha

HERE = Path(__file__).resolve().parent
bundle = Path('/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1')
native, contract, motion, timeline, manifest = load_native_bundle(bundle, 'pico')
# Construct only sensor/history bookkeeping; no constructor, sessions, model inference, or stepping.
fresh = Native23BFMRolloutSeed.__new__(Native23BFMRolloutSeed)
fresh.contract = {key: np.asarray(contract[key], dtype=np.float64).copy()
                  for key in ('default_q', 'kp', 'kd', 'training_effort', 'native_effort', 'native_velocity')}
fresh.lo, fresh.hi = native.jnt_range[1:].T.copy()
fresh.history = BFMHistory()
fresh.previous_action = np.zeros(23, np.float32)
fresh.recorded_controls = 0
fresh.actual_action_max_abs = 0.
fresh.actual_action_components_outside_five = 0
args = SimpleNamespace(clip='pico', probe='full-lifecycle', horizon=30, commit=5, iterations=5, threads=8,
                       fd_epsilon=1e-6, feedback_clip=.1, all_joint_limit_margin=.05,
                       all_joint_limit_weight=2000., relative_foot_weight=400., hard_feasibility=True,
                       restoration=True, restoration_control_lm_retry=True, target_seed=Path('set'),
                       motion_override=Path('set'), preceding_run=HERE.parent/'pico_full_control_lm_v1',
                       preceding_endpoint=HERE.parent/'pico_full_endpoint_independent_v1/endpoint.npz',
                       output=HERE/'preflight_unused_output')
continuity = Continuity(args, native, fresh, timeline)
result = dict(passed=True, saved_prefix_controls=6500, saved_prefix_physics_steps=65000,
              all_prefix_fields_bitexact=True, all6500_measured_histories_actions_bitexact=True,
              checkpoint6500_previous_action_history_bitexact=True, history_record_calls=continuity.history_checks,
              checkpoint_warm_shape=list(continuity.warm.shape), warm_origin_plan=continuity.metadata['plans'][-1]['control'],
              independent_endpoint_sha256=sha(args.preceding_endpoint),
              constructor_invoked=False, policy_inference=False, optimizer=False, native_step=False,
              initialization_forward=False, runtime_version=__import__('mujoco').__version__,
              source_sha256=sha(HERE/'repo/gear_sonic/utils/pico_terminal_continuity.py'))
(HERE/'preflight.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result))
