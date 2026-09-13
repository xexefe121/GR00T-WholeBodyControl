"""Saved-array comparison only; no controller, model, or physics imports/calls."""
from pathlib import Path
import csv
import hashlib
import json
import platform
import sys
import traceback

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
NEW = BASE.parent
INPUTS = Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1')
PACKAGES = {'fp64': NEW/'direct_target_fp64_export_evaluation_v2',
            'direct5000': NEW/'direct_target_student_evaluation_v1',
            'expert': NEW/'bfm_entry250_actual_oracle_v1'}
COLORS = {'expert': '#087e8b', 'fp64': '#c23b33', 'direct5000': '#7356a8'}
LABELS = {'expert': 'Qualified query250 expert', 'fp64': '55000 head / FP64 export', 'direct5000': 'Earlier direct5000'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())
def write(path, value): Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
def exact(a, b): return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()
def rmse(a, axis=-1): return np.sqrt(np.mean(a*a, axis=axis))


def first_numeric_row(a, b):
    rows = np.flatnonzero(np.any(a != b, axis=tuple(range(1, a.ndim))))
    return int(rows[0]) if len(rows) else None


def stats(values):
    return dict(count=len(values), mean=float(np.mean(values)),
                p50=float(np.percentile(values, 50)), p95=float(np.percentile(values, 95)),
                max=float(np.max(values)), over_20ms=int(np.sum(values > 20)))


def main():
    out = BASE/'results_v2'
    out.mkdir(exist_ok=False)
    paths = [Path(__file__), INPUTS/'contract.json', INPUTS/'walk003/timeline.json']
    for key, package in PACKAGES.items():
        paths += [package/'nominal/trace.npz', package/'nominal/report.json']
        if key != 'expert': paths += [package/'nominal/strict_failure_state.npz', package/'evaluation_completion_verification.json']
    paths += [NEW/'direct_target_fp64_independent_physics_v1/report.json',
              NEW/'direct_target_fp64_independent_intent_v1/report.json',
              NEW/'direct_target_independent_physics_v1/report.json',
              NEW/'bfm250_expert_root_qualification_v1/qualification.json',
              PACKAGES['fp64']/'source_draft_v1/direct_runtime.py',
              PACKAGES['fp64']/'post_lifecycle_hold_5s/report.json']
    qualification = read(NEW/'bfm250_expert_root_qualification_v1/qualification.json')
    for subject in qualification['independent_reports'].values():
        assert sha(subject['path']) == subject['sha256']
        paths.append(Path(subject['path']))
    pins = {str(p): sha(p) for p in paths}
    write(out/'input_hashes.json', pins)
    write(out/'request.json', dict(kind='selected_saved_only_failure_diagnostic', input_hashes=pins,
          model_calls=0, native_steps=0, optimizer_updates=0, new_teacher_queries=0,
          original_requested_controls=1569, conditional_hold_requested_controls=250))
    checks = {}

    def check(name, condition):
        checks[name] = bool(condition)
        if not condition: raise AssertionError(name)

    try:
        traces = {key: dict(np.load(package/'nominal/trace.npz', allow_pickle=False)) for key, package in PACKAGES.items()}
        reports = {key: read(package/'nominal/report.json') for key, package in PACKAGES.items()}
        contract = read(INPUTS/'contract.json'); timeline = read(INPUTS/'walk003/timeline.json')
        limits = np.asarray(contract['joint_limits'], np.float64)
        velocity = np.asarray(contract['native_velocity'], np.float64)
        joints = contract['joint_names']; hip = joints.index('right_hip_roll_joint')
        expert = traces['expert']
        check('original_scope', timeline['total_requested_controls'] == 1569 and timeline['source_frames'] == 819)
        check('source_starts_350', next(p for p in timeline['phases'] if p['name']=='source_motion')['control_start'] == 350)
        check('expert_qualified_trace', qualification['nominal_trace_sha256'] == pins[str(PACKAGES['expert']/'nominal/trace.npz')])
        result = dict(kind='saved_only_direct_target_failure_comparison', model_calls=0, native_steps=0,
                      optimizer_updates=0, teacher_queries=0, original_requested_controls=1569,
                      original_duration_seconds=31.38, source_controls_requested=819, source_start_seconds=7.0,
                      hold_controls_requested=250, hold_controls_executed=0,
                      runtime=dict(python=sys.version, numpy=np.__version__, matplotlib=matplotlib.__version__, platform=platform.platform()),
                      input_hashes=pins, runs={}, comparisons={})
        series = {}
        for key, trace in traces.items():
            n = len(trace['target']); steps = len(trace['physics_torque'])
            ends = np.r_[0, np.cumsum(trace['physics_substeps'])]
            check(key+'_step_lengths', len(trace['physics_qpos']) == steps+1 == len(trace['physics_qvel']) and ends[-1] == steps)
            check(key+'_control_clock', exact(trace['global_control'], np.arange(n, dtype=np.int64)))
            check(key+'_source_frames', exact(trace['source_frame'], np.arange(n, dtype=np.int64)+11))
            check(key+'_boundary_qpos', exact(trace['qpos'], trace['physics_qpos'][ends]))
            check(key+'_boundary_qvel', exact(trace['qvel'], trace['physics_qvel'][ends]))
            check(key+'_expected_time', exact(trace['physics_time'], trace['physics_expected_time']))
            speeds = np.abs(trace['physics_qvel'][1:,6:])/velocity
            check(key+'_speed_ratios', exact(trace['velocity_ratio'], speeds.max(axis=1)))
            check(key+'_target_bounds', np.all((trace['target'] >= limits[:,0]) & (trace['target'] <= limits[:,1])))
            series[key] = dict(control_time=trace['physics_time'][ends[:-1]], boundary_time=trace['physics_time'][ends],
                               physics_time=trace['physics_time'], speed_ratio=speeds.max(axis=1))
            if key == 'expert': continue
            check(key+'_original_requested_1569', reports[key]['requested_controls'] == 1569)
            check(key+'_report_trace_hash', reports[key]['trace_sha256'] == pins[str(PACKAGES[key]/'nominal/trace.npz')])
            check(key+'_all_initial291_exact', exact(trace['initial_integration'], expert['initial_integration']))
            for field in ['target','physics_qpos','physics_qvel','physics_torque','physics_time']:
                count = 250 if field=='target' else 2500 if field=='physics_torque' else 2501
                check(key+'_startup_'+field, exact(trace[field][:count], expert[field][:count]))
            for local, reference, count in [('physics_actuator_torque','physics_actuator_force',2500),('physics_warning_counts','physics_warning_number',2501),('physics_warning_lastinfo','physics_warning_lastinfo',2501)]:
                check(key+'_startup_'+local, exact(trace[local][:count],expert[reference][:count]))
            for field in ['control_integration_before','control_history_before','control_previous_action_before']:
                check(key+'_entry250_'+field, exact(trace[field][250], expert[field][250]))
            check(key+'_native_clamp_exact', exact(trace['target'], np.clip(trace['raw_proposal'], limits[:,0], limits[:,1])))
            clipped = trace['raw_proposal'] != trace['target']
            clip_rows = np.flatnonzero(np.any(clipped, axis=1) & (trace['global_control'] >= 250))
            first_clip = int(clip_rows[0]) if len(clip_rows) else None
            target_difference = trace['target'] - expert['target'][:n]
            q_difference = trace['physics_qpos'][:,7:] - expert['physics_qpos'][:steps+1,7:]
            dq_difference = trace['physics_qvel'][:,6:] - expert['physics_qvel'][:steps+1,6:]
            q_first = first_numeric_row(trace['physics_qpos'], expert['physics_qpos'][:steps+1])
            dq_first = first_numeric_row(trace['physics_qvel'], expert['physics_qvel'][:steps+1])
            pre_first = first_numeric_row(trace['control_integration_before'], expert['control_integration_before'][:n])
            worst = int(np.argmax(np.abs(target_difference[250])))
            last = steps; last_speed_joint = int(np.argmax(speeds[-1]))
            clip_info = None if first_clip is None else dict(control=first_clip, time=float(series[key]['control_time'][first_clip]),
                joints=[dict(index=int(j), name=joints[j], raw=float(trace['raw_proposal'][first_clip,j]),
                             applied=float(trace['target'][first_clip,j]), excess=float(abs(trace['raw_proposal'][first_clip,j]-trace['target'][first_clip,j]))) for j in np.flatnonzero(clipped[first_clip])])
            result['runs'][key] = dict(requested_controls=1569, attempted_controls=n, complete_controls=int(np.sum(trace['physics_substeps']==10)),
                partial_substeps=int(trace['physics_substeps'][-1]), native_steps=steps, end_time=float(trace['physics_time'][-1]),
                source_controls_executed=int(np.sum((trace['global_control'] >= 350) & (trace['global_control'] <1169))),
                failure=reports[key]['failure'], first_clip=clip_info, clipped_learned_controls=len(clip_rows),
                startup_250_timing_ms=stats(trace['inference_ms'][:250]), learned_timing_ms=stats(trace['inference_ms'][250:]),
                inference_counts=reports[key]['inference_counts'], segment_elapsed_seconds=reports[key]['elapsed_seconds'],
                final_speed_witness=dict(joint_index=last_speed_joint, joint=joints[last_speed_joint],
                     velocity=float(trace['physics_qvel'][-1,6+last_speed_joint]), limit=float(velocity[last_speed_joint]), ratio=float(speeds[-1,last_speed_joint])),
                initial_target_mismatch=dict(control=250, source_frame=261, rmse_rad=float(rmse(target_difference[250])),
                     worst_joint=joints[worst], maximum_absolute_difference_rad=float(abs(target_difference[250,worst])),
                     learner_target=float(trace['target'][250,worst]), expert_target=float(expert['target'][250,worst]),
                     all_joint_difference_rad=target_difference[250].tolist()),
                first_target_difference_control=first_numeric_row(trace['target'], expert['target'][:n]),
                first_physics_qpos_difference_step=q_first, first_physics_qvel_difference_step=dq_first,
                first_full_integration_precontrol_difference=pre_first,
                first_physics_qpos_difference_time=float(trace['physics_time'][q_first]),
                final_joint_pose_rmse_vs_same_clock_expert=float(rmse(q_difference[-1])),
                final_joint_speed_rmse_vs_same_clock_expert=float(rmse(dq_difference[-1])),
                final_root_position_distance_vs_same_clock_expert=float(np.linalg.norm(trace['physics_qpos'][-1,:3]-expert['physics_qpos'][steps,:3])))
            series[key].update(target_rmse=rmse(target_difference), pose_rmse=rmse(q_difference), speed_rmse=rmse(dq_difference))
            rows = []
            for c in range(250,n):
                a,b=int(ends[c]),int(ends[c+1]); jj=int(np.argmax(speeds[a:b].max(axis=0)))
                rows.append(dict(control=c,source_frame=c+11,start_time=float(trace['physics_time'][a]),end_time=float(trace['physics_time'][b]),
                    substeps=b-a,clipped_joint_count=int(clipped[c].sum()),target_rmse_vs_expert=float(rmse(target_difference[c])),
                    hip_roll_target=float(trace['target'][c,hip]),hip_roll_expert_target=float(expert['target'][c,hip]),
                    hip_roll_raw=float(trace['raw_proposal'][c,hip]),hip_roll_start_q=float(trace['physics_qpos'][a,7+hip]),
                    hip_roll_end_q=float(trace['physics_qpos'][b,7+hip]),hip_roll_end_dq=float(trace['physics_qvel'][b,6+hip]),
                    interval_max_speed_ratio=float(speeds[a:b].max()),interval_max_speed_joint=joints[jj],inference_ms=float(trace['inference_ms'][c])))
            with (out/(key+'_per_control.csv')).open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        new, old = traces['fp64'], traces['direct5000']
        common = min(len(new['target']),len(old['target']))
        result['comparisons']['new_vs_old'] = dict(first_target_difference_control=first_numeric_row(new['target'][:common],old['target'][:common]),
            first_feature_difference_control=first_numeric_row(new['features'][:common],old['features'][:common]),
            control250_feature_byte_exact=exact(new['features'][250],old['features'][250]),
            control250_target_rmse=float(rmse(new['target'][250]-old['target'][250])),
            control250_max_target_difference=float(np.max(np.abs(new['target'][250]-old['target'][250]))))
        physics = read(NEW/'direct_target_fp64_independent_physics_v1/report.json')
        check('root_reproduced_saved_samples', physics['recorded_trace_reproduced_through_last_sample'] and physics['physics_steps']==2916)
        check('root_failure_boundary', physics['first_failure']['physics_step']==2916 and physics['first_failure']['control_index']==291 and physics['first_failure']['substep']==6)
        check('failure_correct_joint', result['runs']['fp64']['final_speed_witness']['joint'] == 'right_hip_roll_joint')
        check('head_42_no_BFM', result['runs']['fp64']['learned_timing_ms']['count']==42 and reports['fp64']['inference_counts']['1_actor_attempted']==0 and reports['fp64']['inference_counts']['1_backward_attempted']==0)
        result['timing_scope'] = 'Saved inference_ms measures propose(), including measured terms/history staging, feature construction, ORT, output mapping and count checks. It excludes commit, native steps and asynchronous scheduling. Learned rows use only the head, but this is not an isolated ORT kernel timer or real-time proof.'
        result['limitations'] = [
            'Both learned runs fail during acquisition before any of the original 819 source controls; original requested scope is not shortened.',
            'Same-clock expert targets were produced at the expert own states, not counterfactual labels at learner states. No teacher map or inference is reevaluated here.',
            'Comparison of the earlier 5000 head and later 55000 FP64 export combines training and export changes; it does not isolate the causal effect of precision.',
            'Exact initial/prefix equality and divergence measurements describe saved trajectories; they do not establish a unique cause or select a corrective method.',
            'Only 42 new learned-phase timing observations; no asynchronous plant, lock contention, real-time scheduling, sustained balance, Pico or hardware qualification.',
            'Last new control contains six actual substeps, earlier control eight. Plots use actual saved times and never fabricate the unfinished 20 ms boundary.'
        ]
        plot(traces,series,result,out,velocity,hip)
        check('all_input_hashes_unchanged', pins == {p:sha(p) for p in pins})
        result.update(saved_checks=checks, passed_saved_accounting=all(checks.values()), policy_qualified=False,
                      output_hashes={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='report.json'})
        write(out/'report.json',result)
        summary(result,out)
        print(json.dumps(dict(passed_saved_accounting=True,report_sha256=sha(out/'report.json'),runs=result['runs'],comparisons=result['comparisons']),indent=2))
    except BaseException as exc:
        write(out/'failure.json',dict(exception=repr(exc),traceback=traceback.format_exc(),checks=checks,input_hashes=pins))
        raise


def plot(traces,series,result,out,velocity,hip):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig=plt.figure(figsize=(15,13),layout='constrained')
    grid=fig.add_gridspec(4,2,height_ratios=[.8,1.5,1.5,1.5])
    scope=fig.add_subplot(grid[0,:])
    for start,stop,color,label in [(0,5,'#dce5ea','Startup'),(5,7,'#ffe6b8','Acquisition'),(7,23.38,'#d8ede6','819 source controls'),(23.38,25.38,'#ffe6b8','Return'),(25.38,31.38,'#dce5ea','Terminal')]:
        scope.axvspan(start,stop,color=color);scope.text((start+stop)/2,.87,label,ha='center',va='center',fontsize=9)
    for row,key in enumerate(['fp64','direct5000']):
        end=result['runs'][key]['end_time'];y=.53-row*.24
        scope.plot([0,end],[y,y],color=COLORS[key],lw=5);scope.scatter([end],[y],marker='x',color=COLORS[key],s=60)
        scope.text(end+.3,y,f'{LABELS[key]}: failed {end:.3f} s; 0 source',va='center',color=COLORS[key])
    scope.set(xlim=(0,31.38),ylim=(0,1),yticks=[],xlabel='Original lifecycle time (s) â€” full 31.38 s requested; conditional 5 s hold unrun')
    axes=[fig.add_subplot(grid[r,c]) for r in [1,2,3] for c in [0,1]]
    for key in ['expert','direct5000','fp64']:
        t=traces[key];s=series[key];mask=(s['physics_time']>=4.9)&(s['physics_time']<=6.34)
        color=COLORS[key]
        axes[0].plot(s['physics_time'][mask],t['physics_qpos'][mask,7+hip],color=color,label=LABELS[key])
        count=min(len(t['target']),317)
        edges=s['boundary_time'][:count+1]
        axes[1].stairs(t['target'][:count,hip],edges,color=color,label=LABELS[key])
        axes[2].plot(s['physics_time'][mask],t['physics_qvel'][mask,6+hip],color=color)
        axes[3].plot(s['physics_time'][1:][mask[1:]],s['speed_ratio'][mask[1:]],color=color)
        if key!='expert':
            axes[4].plot(s['control_time'][250:],s['target_rmse'][250:],color=color,label=LABELS[key]+' target')
            axes[4].plot(s['physics_time'][mask],s['pose_rmse'][mask],color=color,ls='--',label=LABELS[key]+' pose')
            axes[5].plot(t['global_control'][250:],t['inference_ms'][250:],color=color,marker='.',label=LABELS[key]+f" ({len(t['target'])-250} calls)")
    axes[0].set(title='Right hip roll: actual joint position',ylabel='rad')
    axes[1].set(title='Right hip roll: applied target, exact held intervals',ylabel='rad')
    axes[2].set(title='Right hip roll: speed failure at 5.832 s',ylabel='rad/s')
    axes[2].axhline(-velocity[hip],ls='--',color='black',lw=1);axes[2].axhline(velocity[hip],ls='--',color='black',lw=1)
    axes[2].scatter([result['runs']['fp64']['end_time']],[traces['fp64']['physics_qvel'][-1,6+hip]],color=COLORS['fp64'],s=35,zorder=5)
    axes[3].set(title='Largest native speed ratio across all 23 joints',ylabel='|speed| / native limit')
    axes[3].axhline(1,color='black',ls='--',lw=1)
    axes[4].set(title='23-joint RMSE vs expert at same recorded clock',ylabel='rad; target solid / actual pose dashed')
    axes[5].set(title='Learned-phase proposal timing only (250 startup BFM excluded)',xlabel='Control index',ylabel='ms')
    axes[5].axhline(20,color='black',ls='--',lw=1);axes[5].legend(fontsize=8)
    for ax in axes[:5]:ax.set(xlim=(4.9,6.34),xlabel='Actual recorded simulation time (s)');ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);axes[4].legend(fontsize=8)
    fig.suptitle('Native23 walk003 â€” saved failure comparison\nNew FP64 head: right hip-roll speed âˆ’22.399 rad/s > 20 limit; earlier head: ankle range failure at 6.316 s',fontsize=16)
    fig.supxlabel('No new physics or inference. Expert comparison uses expert own states. Timing excludes simulation and does not prove real-time control.',fontsize=10)
    fig.savefig(out/'failure_comparison.png',dpi=160)
    fig.savefig(out/'failure_comparison.svg')
    plt.close(fig)


def summary(result,out):
    n=result['runs']['fp64'];o=result['runs']['direct5000'];tim=n['learned_timing_ms']
    text=f'''# Saved FP64 failure diagnostic

The original 1569-control lifecycle failed at control 291/substep 6, 5.832 s. Right hip roll reached -22.398976 rad/s against 20 rad/s. Root independently reproduced all 2916 native steps. Zero of 819 source controls and zero hold controls executed.

Both learner runs share the expert's complete initial state and exact 250-control startup. New target first differs at control {n['first_target_difference_control']}; initial 23-joint target RMSE is {n['initial_target_mismatch']['rmse_rad']:.9f} rad. First position divergence appears at physics step {n['first_physics_qpos_difference_step']}; first different precontrol is {n['first_full_integration_precontrol_difference']}.

New first learned target clamp: control {n['first_clip']['control'] if n['first_clip'] else 'none'}, versus earlier direct5000 control {o['first_clip']['control'] if o['first_clip'] else 'none'}. CSV files preserve every learned control's target, right hip-roll state, clamp count and maximum joint-speed ratio.

New 42 head-only-phase proposal times: median {tim['p50']:.6f} ms, p95 {tim['p95']:.6f} ms, maximum {tim['max']:.6f} ms. Startup 250 BFM timings are separately reported. The timer includes measured-state/history staging, features, head call and output checks; it excludes commit/native stepping and is not a pure ORT timer or independent 500 Hz plant proof.

Earlier direct5000 failed at 6.316 s on left ankle pitch range. Longer survival is not a tracking qualification; both stop before source begins at 7 s. Same-clock expert targets are descriptive comparisons, not counterfactual teacher labels at learner states. Training and export both differ between the two heads, so this comparison does not isolate precision as a cause.

All source/input hashes are pinned and rechecked. This artifact uses saved arrays only: zero model, physics, optimizer or teacher-map calls. See report.json for checks and full limitations; failure_comparison.png/.svg for the plot.
'''
    (out/'REPORT.md').write_text(text,encoding='utf-8')


if __name__ == '__main__': main()
