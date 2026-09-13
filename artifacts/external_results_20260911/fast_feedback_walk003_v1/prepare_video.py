"""Adapt the existing full-duration fixed-camera renderer to this actual rollout."""
from pathlib import Path
BASE=Path(__file__).resolve().parent
source=Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_resume_20260911/render_pico_qualified_full.py')
s=source.read_text()
s=s.replace('ROOT = Path(__file__).resolve().parents[2]',"ROOT = Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof')")
s=s.replace("OUT = BASE / 'pico_qualified_full_video_v1'","OUT = BASE / 'fast_feedback_walk003_v1/video_v1'")
start=s.index("    qualification_path = BASE / 'pico_full_root_qualification_v1/qualification.json'")
stop=s.index('    with np.load(qualification',start)
s=s[:start]+'''    run = BASE / 'fast_feedback_walk003_v1/baseline_v1'
    qualification_path = run / 'report.json'
    result = json.loads(qualification_path.read_text())
    assert result['full_motion_and_hold_completed']
    assert all(result[k]['quiet_standing_diagnostic']['quiet_standing_diagnostic_pass'] for k in ('main','hold'))
    qualification = {'traces': {'full': {'path': str(run/'nominal/trace.npz')}, 'hold': {'path': str(run/'post_lifecycle_hold_5s/trace.npz')}}}
    inputs = {str(qualification_path): sha(qualification_path), str(Path(__file__)): sha(Path(__file__))}
    for key,segment in (('full','main'),('hold','hold')):
        path = Path(qualification['traces'][key]['path'])
        assert sha(path) == result[segment]['trace_sha256']
        inputs[str(path)] = sha(path)
'''+s[stop:]
s=s.replace('67801','18191').replace('67800','18190').replace('65300','15690')
s=s.replace('/pico/reference.npz','/walk003/reference.npz').replace("'pico/timeline.json'","'walk003/timeline.json'")
s=s.replace('steps = np.arange(0, 18191, 50)','steps = np.unique(np.r_[np.arange(0, 18191, 50), 11690, 15690, 18190])')
s=s.replace('contact_steps = (0, 3500, 21000, 38000, 39000, 50000, 15690, 18190)','contact_steps = (0, 3500, 5500, 7500, 9500, 11690, 15690, 18190)')
s=s.replace('130.6','31.38').replace('135.6','36.38').replace('115.6','16.38')
s=s.replace('PICO | simulation','WALK003 | simulation').replace('67,800 audited physics steps','18,190 fresh physics steps')
s=s.replace('OFFLINE PLANNER | 0.74 s prepared preview | live-speed control and real Pico remain unqualified','FAST STATE FEEDBACK | prepared motion | live teleoperation and real Pico remain unqualified')
s=s.replace('duration = .1 if slot + 1 < len(steps) else .002','duration = (steps[slot+1]-step)*.002 if slot + 1 < len(steps) else .002')
s=s.replace('full_pico_and_continuous_hold.fixed_world.mp4','full_walk003_and_continuous_hold.fixed_world.mp4')
s=s.replace('full_qualified_PICO_and_continuous_hold_saved_state_visual','full_fast_feedback_walk003_and_continuous_hold_saved_state_visual')
s=s.replace('source_controls=5780, lifecycle_controls=6530','source_controls=819, lifecycle_controls=1569')
s=s.replace('physical_steps=18190, display_seconds=36.38','physical_steps=18190, display_seconds=36.38')
s=s.replace('"""Render all qualified PICO time, plus its hold, from saved physical states."""','"""Render the complete fast feedback motion and hold from its fresh native trace."""')
out=BASE/'render_video.py';assert not out.exists();out.write_text(s,encoding='utf-8')
print(out)
