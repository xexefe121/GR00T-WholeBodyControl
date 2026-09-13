import ast
from pathlib import Path
import numpy as np
from qualified_inputs import visual_grid,CONTACT_STEPS

def test_complete_native_grid_and_endpoint_durations():
    steps,frames,durations=visual_grid()
    assert len(steps)==338 and steps[0]==0 and steps[-1]==16670
    assert set(CONTACT_STEPS)<=set(steps)
    np.testing.assert_allclose(np.r_[0,np.cumsum(durations[:-1])],steps*.002,atol=1e-12,rtol=0)
    assert durations[-1]==.002 and abs(durations.sum()-33.342)<1e-12

def test_exact_reference_frames_at_every_boundary():
    steps,frames,_=visual_grid();mapping=dict(zip(steps,frames))
    assert mapping[0]==10 and mapping[3500]==360 and mapping[10170]==1027
    assert mapping[11170]==1127 and mapping[14170]==mapping[16670]==1427
    assert np.all(frames[steps>14170]==1427)

def test_renderer_has_no_dynamics_or_root_alignment():
    text=Path(__file__).with_name('render_walk002_qualified.py').read_text()
    ast.parse(text)
    for forbidden in ('mj_step(', 'mj_forward(', 'mj_setState(', '.propose(', '.ilqr('):assert forbidden not in text
    assert "load_qualified(qualification_path, qualification_sha256)" in text
    assert "duration = float(durations[slot])" in text
