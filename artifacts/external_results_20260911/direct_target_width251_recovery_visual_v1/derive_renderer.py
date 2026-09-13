"""Source-only derivative of the original qualified-expert contact sheet."""
from pathlib import Path
import hashlib,json
BASE=Path(__file__).resolve().parent;NEW=BASE.parent
TEMPLATE=Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_resume_20260911/render_qualified_expert.py')
def put(path,text):
    with path.open('x',encoding='utf-8',newline='') as f:f.write(text)
def replace(text,old,new):
    assert text.count(old)==1,(old,text.count(old));return text.replace(old,new)
def main():
    source=BASE/'source_draft_v1';source.mkdir(exist_ok=False)
    with (BASE/'render_qualified_expert_original.py').open('xb') as f:f.write(TEMPLATE.read_bytes())
    text=TEMPLATE.read_text()
    text=replace(text,'import hashlib\nimport json','import argparse\nimport hashlib\nimport json')
    text=replace(text,'import mujoco\nimport numpy as np\nfrom PIL import Image, ImageDraw, ImageFont\n','from qualification import paths_and_pins\n')
    text=replace(text,"ROOT = Path(__file__).resolve().parents[2]\nBASE = Path('E:/codex-artifacts/sonic23_teleop_resume_20260911')","ROOT = Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof')\nBASE = Path(__file__).resolve().parent.parent")
    start=text.index('    qualification_path = ');end=text.index('    with np.load(traces[0])',start)
    text=text[:start]+'''    parser = argparse.ArgumentParser()
    parser.add_argument('--root-selected', action='store_true')
    args = parser.parse_args()
    if not args.root_selected:
        raise ValueError('Rendering is not selected by source preparation')
    paths, pins = paths_and_pins()
    traces = [paths['main_trace'], paths['hold_trace']]
    import mujoco
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
'''+text[end:]
    text=replace(text,'with np.load(traces[0]) as life, np.load(traces[1]) as hold:',"with np.load(traces[0], allow_pickle=False) as life, np.load(traces[1], allow_pickle=False) as hold:")
    text=replace(text,"        assert len(life['source_frame']) == 1569 and len(hold['source_frame']) == 250", "        np.testing.assert_array_equal(life['physics_qvel'][-1], hold['physics_qvel'][0])\n        assert life['physics_time'][-1] == hold['physics_time'][0]\n        assert len(life['source_frame']) == 1569 and len(hold['source_frame']) == 250\n        assert np.all(life['physics_substeps'] == 10) and np.all(hold['physics_substeps'] == 10)")
    text=replace(text,"    reference_path = Path('E:/codex-artifacts/sonic23_teleop_six_hour_20260910/mjbatch_intent_floor_inputs_v1/walk003/reference.npz')","    reference_path = paths['reference']")
    text=replace(text,'with np.load(reference_path) as reference:', 'with np.load(reference_path, allow_pickle=False) as reference:')
    text=replace(text,"with np.load(bundle / 'prepared_model_arrays.npz') as arrays:","with np.load(bundle / 'prepared_model_arrays.npz', allow_pickle=False) as arrays:")
    text=replace(text,"    out = BASE / 'expert_resumed_qualified_visual_v1'","    out = BASE / 'rendered_v1'")
    text=replace(text,'Native 23-DOF G1 | full walking expert branch + separate standing hold','Native 23-DOF G1 | offline expert recovery from actual student state251')
    text=replace(text,'Each pair: declared native reference LEFT / recorded physics RIGHT. Fixed world camera. Offline expert; fast student still unqualified.',
        'Reference LEFT / actual RIGHT. Fixed world camera. Full1569 + continuous250 hold. Offline expert recovery; fast student unqualified.')
    text=replace(text,"kind='qualified_expert_recorded_physics_visual_no_dynamics'","kind='offline_expert_recovery_recorded_physics_visual_no_dynamics'")
    text=replace(text,"                  hashes={str(p): sha(p) for p in [*traces, qualification_path, reference_path,\n                          bundle / 'native_prepared.xml', bundle / 'prepared_model_arrays.npz', Path(__file__)]},", "                  hashes=pins, new_physics_steps=0, model_inference_calls=0,\n                  learned_student_qualified=False, recovery_initial_control=251,\n                  actual_BFM_prefix_controls=250, actual_student_prefix_controls=1,")
    text=replace(text,"    (out / 'report.json').write_text", "    for path, digest in pins.items():\n        if sha(Path(path)) != digest:\n            raise ValueError('Render input changed: ' + path)\n    (out / 'report.json').write_text")
    put(source/'render_contact_sheet.py',text)
    put(BASE/'derivation.json',json.dumps(dict(source_only=True,template_path=TEMPLATE.as_posix(),
        template_sha256=hashlib.sha256(TEMPLATE.read_bytes()).hexdigest(),renderer_sha256=hashlib.sha256((source/'render_contact_sheet.py').read_bytes()).hexdigest(),
        actual_renderer_executed=False,new_native_steps=0,model_calls=0),indent=2)+'\n')
if __name__=='__main__':main()
