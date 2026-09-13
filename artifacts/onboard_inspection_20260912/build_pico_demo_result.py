"""Assemble the completed simulation runs into a concise delivery artifact."""
from pathlib import Path
import hashlib
import json
import numpy as np

HERE=Path(__file__).resolve().parent
FW=Path(r'E:\codex-artifacts\sonic23_teleop_resume_20260911\onboard_factory_firmware_v1')
DEMO=FW/'received_pico_demo_v1'

def logged_report(log):
    text=log.read_text(encoding='utf-8-sig')
    return Path(text.rsplit('Result: ',1)[1].strip())

def entry(path):
    r=json.loads(path.read_text())
    return dict(report=str(path),seconds=r['physical_steps']*.002,
        physical_complete=r['physical_complete'],tracking=r.get('source',{}).get('passed',False),
        quiet30=r.get('continuous_quiet30',{}).get('passed',False),
        fault_quiet30=(r.get('fault_continuous_quiet30') or {}).get('passed',False),
        fault_scenario_passed=r.get('fault_scenario_passed',False),
        control_misses=r['controller_deadline_misses'],late_physics_ticks=r['plant_finishes_over2ms_late'],
        spin_us=r.get('plant_spin_us',0),fault_capture=r.get('fault_standing_capture_enabled',False),
        controller_ms=r['controller_ms_p50_p95_max'],source=r.get('source',{}),
        packet_fault=r.get('packet_report',{}).get('fault'))

def main():
    rows={'pico':entry(FW/'received_sim_runs/20260913_044020_384_pico/report.json'),
          'pico_input_loss':entry(FW/'received_sim_runs/20260913_045639_671_pico/report.json'),
          'spin100_nominal':entry(logged_report(DEMO/'final_nominal.log')),
          'spin100_input_loss':entry(logged_report(DEMO/'final_inputloss.log'))}
    for clip in ('walk002','walk003','walk008'):
        path=FW/'target_filter_v1'/('independent_'+clip+'.log')
        rows[clip]=entry(logged_report(path))
    original=FW/'received_sim_runs/20260913_044020_384_pico'
    with np.load(original/'trace.npz') as z:base_vx=float(z['states'][0,30])
    for name,folder,expected in (
        ('pico_plus003','20260913_044427_039_pico',.03),
        ('pico_minus003','20260913_044738_052_pico',-.03)):
        path=FW/'received_sim_runs'/folder
        with np.load(path/'trace.npz') as z:delta=float(z['states'][0,30])-base_vx
        if abs(delta-expected)>1e-12:raise ValueError('Recorded initial velocity does not match its declared scenario')
        rows[name]=entry(path/'report.json');rows[name]['verified_initial_vx_delta_mps']=delta
    digest=hashlib.sha256((DEMO/'controller.onnx').read_bytes()).hexdigest()
    result=dict(controller_sha256=digest,leg_target_filter_alpha=.9,
        future_reference_frames=0,hardware_commands=False,general_live_teleop_qualified=False,runs=rows)
    rendered=original/'video_fast/render.json'
    if rendered.exists():result['video']=json.loads(rendered.read_text())
    (DEMO/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    nominal=rows['pico'];source=nominal['source']
    lines=['# Received-pose native23 Pico simulation result','',
        'Full-body teleop remains unqualified. This artifact records a runnable simulation demo and its remaining limits.','',
        f"Recorded zero-spin Pico run: {nominal['seconds']:.3f}s; physical completion {nominal['physical_complete']}; continuous30s quiet hold {nominal['quiet30']}; control misses {nominal['control_misses']}; late physics ticks {nominal['late_physics_ticks']}.",
        f"Controller median/p95/max: {nominal['controller_ms'][0]:.3f}/{nominal['controller_ms'][1]:.3f}/{nominal['controller_ms'][2]:.3f}ms.",'',
        'The controller uses the native-trained checkpoint, actual applied-command history and leg target filter alpha0.9. Physics advances independently at500Hz; the controller and received-packet producer run at50Hz. The fault-capture option stops the gait only after the generated braking reference is stationary and the robot enters its quiet standing region. Neural balance feedback and the explicit-rearm latch remain active. No future motion, per-motion gains, root forces, state resets or physical-limit changes are used. Robot feedback is privileged MuJoCo floating-base state.','',
        '| Recording | Physical seconds | Completed | Quiet30s | Control misses | Late physics ticks | Full-body tracking |',
        '|---|---:|---|---|---:|---:|---|']
    for name in ('pico','walk002','walk003','walk008'):
        r=rows[name]
        lines.append(f"| {name} | {r['seconds']:.3f} | {r['physical_complete']} | {r['quiet30']} | {r['control_misses']} | {r['late_physics_ticks']} | {r['tracking']} |")
    lines+=['',f"Pico source errors: root p95 {source['root_p95_m']:.3f}m; feet {source['foot_relative_p95_m'][0]:.3f}/{source['foot_relative_p95_m'][1]:.3f}m; leg RMSE {source['leg_rmse_rad']:.3f}rad. The existing limits remain0.20m root,0.12m each foot, and0.15rad legs. Hand/head objectives also remain enforced.",'',
        '| Scenario | Completed | Required quiet30s | Control misses | Late physics ticks |',
        '|---|---|---|---:|---:|']
    for name in ('pico_plus003','pico_minus003','pico_input_loss','spin100_nominal','spin100_input_loss'):
        r=rows[name];quiet=r['fault_quiet30'] if 'input_loss' in name else r['quiet30']
        lines.append(f"| {name} | {r['physical_complete']} | {quiet} | {r['control_misses']} | {r['late_physics_ticks']} |")
    lines+=['','The velocity perturbations were verified from saved initial physical states. Primary nominal, perturbation and walking runs use zero-spin timing. The standing-capture option affects only latched input faults; it was disabled in those nominal runs. Input loss occurs at control2000 (40s). Capture-off input loss stayed upright but kept stepping. Capture-on with zero spin settled quietly but had one late physics tick. The100-microsecond spin trial passed input-loss standing and timing, but its nominal repeat had three late physics ticks. It is not the default; zero-spin timing is retained. All results remain preserved. A complete zero-miss run is demonstrated; repeatable timing across all scenarios remains unresolved.','',
        'Run from PowerShell:','',
        f"```powershell\n& '{HERE / 'RUN_PICO_NATIVE23_DEMO.ps1'}' -Render\n```",'',
        'No hardware publisher is launched. Both bounded training pilots finished and stopped. Earlier Orin access was read-only; the latest connection check reports this laptop Ethernet disconnected.','',
        'Evidence:','']
    if 'video' in result:
        lines.insert(2,f"[Watch the complete160.6-second independent-clock Pico run](<{Path(result['video']['video']).as_posix()}>)\n")
    for name,r in rows.items():lines.append(f"- [{name}](<{Path(r['report']).as_posix()}>)")
    (HERE/'PICO_DEMO_RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(HERE/'PICO_DEMO_RESULT.md')

if __name__=='__main__':main()
