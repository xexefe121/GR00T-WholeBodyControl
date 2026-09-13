"""Original native bundle model preparation; imported only by selected runner."""
import hashlib,json
from pathlib import Path
import mujoco
import numpy as np

def sha256(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_model(directory, clip):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    assert sha256(directory / 'native_prepared.xml') == manifest['portable_xml_sha256']
    assert sha256(directory / 'prepared_model_arrays.npz') == manifest['prepared_arrays_sha256']
    for (name, digest) in manifest['meshes'].items():
        assert sha256(directory / 'meshes' / name) == digest, name
    motion_path = directory / clip / 'native_original.npz'
    assert sha256(motion_path) == manifest['cases'][clip]['native_original.npz']
    model = mujoco.MjModel.from_xml_path(str(directory / 'native_prepared.xml'))
    with np.load(directory / 'prepared_model_arrays.npz', allow_pickle=False) as archive:
        for name in archive.files:
            getattr(model, name)[:] = archive[name]
        mujoco.mj_setConst(model, mujoco.MjData(model))
        for name in archive.files:
            np.testing.assert_array_equal(getattr(model, name), archive[name], err_msg=name)
    contract = json.loads((directory / 'contract.json').read_text())
    assert (model.nq, model.nv, model.nu, model.nbody) == (30, 29, 23, 25)
    assert [model.joint(index).name for index in range(1, model.njnt)] == contract['joint_names']
    assert [model.body(index).name for index in range(1, model.nbody)] == contract['body_names']
    assert model.opt.timestep == contract['timestep'] == 0.002
    assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_EULER
    assert contract['decimation'] == 10
    return (model, contract)
