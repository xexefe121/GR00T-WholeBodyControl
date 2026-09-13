"""Preserve the first math witness and create a separately bound repeat."""
from pathlib import Path

base = Path(__file__).resolve().parent
text = (base / 'audit_mpc_math_independent.py').read_text()
old = 'from gear_sonic.utils.g1_true23_mjbatch_model import position_servo_copy'
new = '''SNAPSHOT_DIR=ROOT/'artifacts/teleop_six_hour_20260910/mpc_math_witness_v2_sources'
SNAPSHOT_DIR.mkdir(exist_ok=False)
BOUND_SOURCE_HASHES={}
for name in ('g1_true23_mjbatch_model.py','g1_true23_mjbatch_mpc.py','g1_true23_mjbatch_ilqr_core.py'):
    path=ROOT/'gear_sonic/utils'/name
    content=path.read_bytes()
    BOUND_SOURCE_HASHES[str(path)]=hashlib.sha256(content).hexdigest()
    (SNAPSHOT_DIR/name).write_bytes(content)
from gear_sonic.utils.g1_true23_mjbatch_model import position_servo_copy'''
assert text.count(old) == 1
text = text.replace(old, new)
text = text.replace("native,contract,motion,_,_=load_native_bundle", "native,contract,motion,_,_=load_native_bundle", 1)
old = "result=dict(kind='independent_mpc_math_witness',mujoco=mujoco.__version__,"
new = '''for path,digest in BOUND_SOURCE_HASHES.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    bundle=base/'mjbatch_native23_inputs_v1'
    input_files=[bundle/'contract.json',bundle/'walk002/native_original.npz',bundle/'native23.mjb']
    input_files=[path for path in input_files if path.exists()]
    result=dict(kind='independent_mpc_math_witness_v2',mujoco=mujoco.__version__,
                bound_source_hashes=BOUND_SOURCE_HASHES,source_snapshots=str(SNAPSHOT_DIR),
                input_hashes={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in input_files},'''
assert text.count(old) == 1
text = text.replace(old, new).replace("base/'mpc_independent_math_audit.json'", "base/'mpc_independent_math_audit_v2.json'")
destination = base / 'audit_mpc_math_independent_v2.py'
assert not destination.exists()
destination.write_text(text)
print(destination)
