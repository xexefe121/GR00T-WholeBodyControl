# Factory model and sampled damping results

Both variants are stopped. Neither completed any of the four recordings or
reached its continuous 30-second final standing hold. Neither replaced the
retained Pico controller or factory23 dance.

The firmware's `g1_walk_f1.1_d0.65_smpl_foot.mnn` and
`g1_walk_f1.1_d0.65_smpl_foot_dof_v_limit.mnn` are byte-identical, SHA256
`ee9cbe52b66c1eb090391932b9af0869f6aa0b3f3cc327b03f7201921286106d`.
They were tested once, directly through MNN and the existing full-body receiver.
The recovered sknee0 profile supplies frequency 1.1 Hz and duty fraction 0.65.
The same settings were used for every recording. This is a tested candidate
adapter, not a claim that every vendor runtime detail was reconstructed.

The separate damping variant wraps the retained learned Pico controller.
It derives fixed diagonal damping from the inverse native mass matrix at the
global factory pose, and compensates position targets at control boundaries
to preserve the requested instantaneous torque unless targets clip. Additional
damping then opposes velocity departures between boundaries. Actual applied
targets are mapped to equivalent original-gain targets for factory memory.
No physical model parameter, state, reference or native limit was changed.

| Recording | Requested | SMPL-foot factory | Boundary-matched damping |
|---|---:|---:|---:|
| walk002 | 58.34 s | 17.674 s, joint bound | 11.192 s, fall |
| walk003 | 61.38 s | 11.322 s, joint speed | 14.868 s, fall |
| Pico | 160.60 s | 15.406 s, joint bound | 40.066 s, joint bound |
| held-out walk008 | 52.28 s | 9.732 s, joint speed | 11.732 s, joint bound |

All full-body tracking verdicts fail. These were unpaced native MuJoCo 3.2.3
rollouts with unchanged 500 Hz physics and 50 Hz command intervals. Typical
inference was below 3 ms; independent timing was not qualified. Full traces,
original tracking metrics, actual gains and failure evidence are saved under
`onboard_factory_firmware_v1/human_foot_candidates_v1/smpl_foot/raw_received`
and `onboard_factory_firmware_v1/boundary_matched_damping_v1`.

Optional simulation flags are `--factory-model PATH --factory-profile sknee0`
with `--factory-locomotion`, or `--boundary-matched-damping` with a native
locomotion controller. Default model, profile and gains remain unchanged.
