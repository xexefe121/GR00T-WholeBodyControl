"""Test stronger terminal position feedback, with unchanged reference and plant."""
from pathlib import Path
BASE=Path(__file__).resolve().parent
original=BASE.parent/'direct_target_width251_evaluation_v2/source_draft_v1/terminal_yaw4_goal.py'
s=original.read_text()
needle='    delta *= min(1.,.6/max(np.linalg.norm(delta),1e-8))'
assert s.count(needle)==1
s=s.replace(needle,'    delta *= 4.0  # Controller position feedback gain; reference remains unchanged.\n'+needle)
out=BASE/'terminal_position4.py';assert not out.exists();out.write_text(s,encoding='utf-8')
for variant,source,process,output in (
    ('baseline','run_controller.py','position4_baseline_process_v1','position4_baseline_v1'),
    ('perturbed','run_initial_perturbation.py','position4_perturbed_process_v1','position4_perturbed_v1')):
    s=(BASE/source).read_text()
    needle='from direct_runtime import DirectStudentRuntime'
    s=s.replace(needle,needle+'\nimport student_linear_runtime\nfrom terminal_position4 import terminal_goal_yaw4 as position4_goal\nstudent_linear_runtime.terminal_goal_yaw4=position4_goal')
    s=s.replace('original_timing=True,','original_timing=True,terminal_position_feedback_gain=4.0,')
    name='run_position4_'+variant+'.py';out=BASE/name;assert not out.exists();out.write_text(s,encoding='utf-8')
    launch=(BASE/'run_trial_once.py').read_text().replace('baseline_process_v1',process).replace("BASE/'baseline_v1'","BASE/'"+output+"'").replace('run_controller.py',name)
    out=BASE/('launch_position4_'+variant+'.py');assert not out.exists();out.write_text(launch,encoding='utf-8')
print('Prepared baseline and +0.03 m/s initial-velocity trials; terminal position gain 4, unchanged native limits/reference.')
