"""Adapt a separate frozen copy of the PICO saved-state renderer, no rendering."""
import hashlib
import json
from pathlib import Path

OUT=Path(__file__).parent
ROOT=Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof')
ORIGINAL=ROOT/'artifacts/teleop_resume_20260911/render_pico_qualified_full.py'
text=ORIGINAL.read_text()
changes=[]
def change(old,new):
    global text
    assert old in text,old
    changes.append(dict(old=old,new=new,count=text.count(old)))
    text=text.replace(old,new)
change('"""Render all qualified PICO time, plus its hold, from saved physical states."""',
       '"""Render qualified walk002 offline hybrid and hold from saved physical states."""')
change('import hashlib','import argparse\nimport hashlib')
change('from PIL import Image, ImageDraw, ImageFont','from PIL import Image, ImageDraw, ImageFont\nfrom qualified_inputs import load_qualified, visual_grid, CONTACT_STEPS')
change('ROOT = Path(__file__).resolve().parents[2]',"ROOT = Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof')")
change("OUT = BASE / 'pico_qualified_full_video_v1'","OUT = BASE / 'walk002_qualified_hybrid_video_v1'")
start=text.index('def main():')
stop=text.index('    with np.load(qualification[',start)
old=text[start:stop]
change(old,"def main(qualification_path, qualification_sha256):\n    qualification, inputs, full_arrays, hold_arrays = load_qualified(qualification_path, qualification_sha256)\n    inputs[str(Path(__file__))] = sha(Path(__file__))\n    inputs[str(Path(__file__).with_name('qualified_inputs.py'))] = sha(Path(__file__).with_name('qualified_inputs.py'))\n")
change('67801','16671')
change('67800','16670')
change('mjbatch_intent_floor_inputs_v1/pico/reference.npz','mjbatch_intent_floor_inputs_v1/walk002/reference.npz')
change("    steps = np.arange(0, 16671, 50)\n    frames = np.minimum(np.where(steps == 0, 10, (steps - 1) // 10 + 11), len(desired_all) - 1)",
       '    steps, frames, durations = visual_grid()\n    assert len(desired_all) == 1428')
change("bundle / 'pico/timeline.json'","bundle / 'walk002/timeline.json'")
change("    timeline = json.loads((bundle / 'walk002/timeline.json').read_text())",
       "    timeline = json.loads((bundle / 'walk002/timeline.json').read_text())\n    full_intent = json.loads(Path(qualification['independent_reports']['full_intent']['path']).read_text())\n    for qualified_asset in (reference_path, receipt_path, bundle / 'native_prepared.xml', bundle / 'prepared_model_arrays.npz', bundle / 'walk002/timeline.json'):\n        assert sha(qualified_asset) in full_intent['hashes'].values(), str(qualified_asset)")
change('contact_steps = (0, 3500, 21000, 38000, 39000, 50000, 65300, 16670)','contact_steps = CONTACT_STEPS')
change("'Full motion and separate quiet hold pass'","'Qualified source and both quiet windows'")
change('if step >= 65300:', 'if step >= 14170:')
change("            if step >= 14170:\n                phase = f'separate hold +{seconds - 130.6:.1f}/5.0 s'",
       "            if step == 14170:\n                phase = 'original lifecycle end / separate hold boundary'\n            elif step > 14170:\n                phase = f'separate hold +{seconds - 130.6:.2f}/5.0 s'")
change('seconds - 130.6','seconds - 28.34')
change('source_seconds = np.clip(seconds - 7., 0, 115.6)','source_seconds = np.clip(seconds - 7., 0, 13.34)')
change("f'PICO | simulation {seconds:.1f}/135.6 s | source {source_seconds:.1f}/115.6 s | {phase}'",
       "f'WALK002 | simulation {seconds:.2f}/33.34 s | source {source_seconds:.2f}/13.34 s | {phase}'")
change("f'67,800 audited physics steps | joint speed now {speed:.3f} rad/s | root error now {root_error:.3f} m'",
       "f'16,670 audited physics steps | joint speed now {speed:.3f} rad/s | root error now {root_error:.3f} m'")
change("draw.text((14, TOP + HEIGHT + 64), 'OFFLINE PLANNER | 0.74 s prepared preview | live-speed control and real Pico remain unqualified', font=small, fill='#ffd0a3')",
       "mode = 'Recorded MPC commands' if step < 11170 else 'Switch to actual terminal BFM' if step == 11170 else 'Actual terminal BFM yaw4'\n            draw.text((14, TOP + HEIGHT + 64), mode + ' | OFFLINE HYBRID | live-speed control and real Pico remain unqualified', font=small, fill='#ffd0a3')")
change('duration = .1 if slot + 1 < len(steps) else .002','duration = float(durations[slot])')
change("'full_pico_and_continuous_hold.fixed_world.mp4'","'full_walk002_hybrid_and_continuous_hold.fixed_world.mp4'")
change("kind='full_qualified_PICO_and_continuous_hold_saved_state_visual'","kind='full_qualified_walk002_recorded_MPC_prefix_actual_terminal_BFM_saved_state_visual'")
change('source_controls=5780, lifecycle_controls=6530','source_controls=667, lifecycle_controls=1417')
change('physical_steps=16670, display_seconds=135.6','physical_steps=16670, display_seconds=33.34')
change('no_root_alignment=True, no_time_warp=True, prepared_preview_seconds=.74,\n                  raw_pose_support_upper_bound_seconds=.76, live_teleoperation_qualified=False)',
       "no_root_alignment=True, no_time_warp=True, original_prefix_controls=1117,\n                  controller_switch_step=11170, original_terminal_BFM_controls=300,\n                  variable_visual_frame_durations_seconds=durations.tolist(), live_teleoperation_qualified=False)")
change("if __name__ == '__main__':\n    main()","if __name__ == '__main__':\n    parser = argparse.ArgumentParser()\n    parser.add_argument('--qualification', type=Path, required=True)\n    parser.add_argument('--qualification-sha256', required=True)\n    args = parser.parse_args()\n    main(args.qualification, args.qualification_sha256)")
target=OUT/'render_walk002_qualified.py'
target.write_text(text)
(OUT/'pico_renderer_original.py').write_bytes(ORIGINAL.read_bytes())
(OUT/'renderer_transformations.json').write_text(json.dumps(dict(original=str(ORIGINAL),original_sha256=hashlib.sha256(ORIGINAL.read_bytes()).hexdigest(),
    generated_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),changes=changes),indent=2)+'\n')
print(target)
