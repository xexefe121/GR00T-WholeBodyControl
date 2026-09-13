"""Test the same feedback law from a declared small initial velocity offset."""
from pathlib import Path
BASE=Path(__file__).resolve().parent
s=(BASE/'run_controller.py').read_text()
before="    kp,kd,effort=[np.asarray(c[k]) for k in ('kp','kd','native_effort')]"
after="""    # A robustness trial changes only initial root velocity, before any step.
    data.qvel[0]+=.03
    mujoco.mj_forward(native,data)
    kp,kd,effort=[np.asarray(c[k]) for k in ('kp','kd','native_effort')]"""
assert s.count(before)==1;s=s.replace(before,after)
s=s.replace('original_timing=True,requested_main_controls=1569','original_timing=True,initial_root_world_x_velocity_offset_mps=.03,requested_main_controls=1569')
out=BASE/'run_initial_perturbation.py';assert not out.exists();out.write_text(s,encoding='utf-8')
launcher=(BASE/'run_trial_once.py').read_text().replace('baseline_process_v1','initial_perturbation_process_v1').replace("BASE/'baseline_v1'","BASE/'initial_perturbation_v1'").replace('run_controller.py','run_initial_perturbation.py')
out=BASE/'run_initial_perturbation_once.py';assert not out.exists();out.write_text(launcher,encoding='utf-8')
print('Prepared one fresh full-motion/5 s hold trial with initial root x velocity +0.03 m/s.')
