"""Explicit source-level runtime dependency inventory; no model/native imports.

Package allowlists conservatively pin installed code for imported runtime packages.
They intentionally exclude trainer/CUDA libraries, training corpora and build tools.
"""
import hashlib,json
from pathlib import Path

BASE=Path(__file__).resolve().parent
NEW=BASE.parent
ROOT=Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof')
BUNDLE=ROOT/'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1'
BFM=ROOT/'artifacts/teleop_six_hour_20260910/bfm_onnx_v2'
TASK=Path('E:/codex-artifacts/sonic23_teleop_six_hour_20260910')
REFERENCE=TASK/'mjbatch_intent_floor_inputs_v1/walk003'
DEPS=Path('E:/codex_sonic_runtime/bfm_seed_20260910/onnx_deps')
VENV=Path('E:/codex_sonic_runtime/mjbatch323_20260910/venv')
SITE=VENV/'lib/python3.11/site-packages'
PACKAGES=('mujoco','mujoco.libs','numpy','numpy.libs','scipy','scipy.libs','mjbatch',
          'absl','etils','fsspec','glfw','OpenGL','packaging','zipp')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def code_file(p):return p.suffix in ('.py','.json','.pth') or '.so' in p.name

def collect():
    entries={}
    def add(p,reason):
        p=Path(p).resolve();assert p.is_file(),str(p)
        key=p.as_posix();entry=entries.setdefault(key,dict(path=key,sha256=sha(p),bytes=p.stat().st_size,reasons=[]))
        if reason not in entry['reasons']:entry['reasons'].append(reason)
    for p in (BASE/'source_draft_v1').rglob('*.py'):add(p,'frozen evaluator modules including preserved helper imports')
    for name in ('manifest.json','native_prepared.xml','prepared_model_arrays.npz','contract.json'):
        add(BUNDLE/name,'load_native_bundle explicit read')
    manifest=read(BUNDLE/'manifest.json')
    for name,digest in manifest['meshes'].items():
        p=BUNDLE/'meshes'/name;assert sha(p)==digest;add(p,'native model mesh and explicit loader hash')
    for name in ('native_original.npz','original29.npz','timeline.json'):add(BUNDLE/'walk003'/name,'selected canonical motion/root-task clock')
    for name in ('actor.onnx','backward.onnx','manifest.json'):add(BFM/name,'actual startup/terminal BFM sessions')
    for name in ('reference.npz','portable_receipt.json','report.json','before_floor_reference.npz','floor_transform_receipt.json','frame_lift.npz'):
        add(REFERENCE/name,'load_motion_override and floor validation actual read')
    add(TASK/'bfm_online_intent_v2/walk003_terminal_bfm_yaw4_hybrid_v1/recorded_source_audit_v2.json','original source-task convention for measured intent')
    for sub,names in (
        ('original_bfm_entry250_v1/entry250',('trace.npz','report.json')),
        ('bfm_entry250_labels_v1/labels',('labels.npz','report.json'))):
        for name in names:add(NEW/sub/name,'actual startup/query250 parity input')
    add(NEW/'velocity_chord_student_v1/generation/centers.npz','actual gate extracts query250 row2038 and span; full archive retained')
    for p in DEPS.rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts and code_file(p):add(p,'pinned ORT dependency directory used by actual sessions')
    for name in PACKAGES:
        directory=SITE/name;assert directory.is_dir(),str(directory)
        for p in directory.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and code_file(p):add(p,'conservative code closure for imported WSL package '+name)
    for name in ('typing_extensions.py','_virtualenv.py','_virtualenv.pth','coloredlogs.pth'):
        if (SITE/name).is_file():add(SITE/name,'existing WSL import initialization')
    add(VENV/'pyvenv.cfg','selected WSL interpreter environment configuration')
    add(ROOT/'artifacts/teleop_six_hour_20260910/RUN_PASSING_WALK_WSL.sh','actual fixed mount/bootstrap script')
    for p in (BASE/'prepare_sources.py',BASE/'source_derivation.json',Path(__file__)):add(p,'source-only provenance')
    # Completed receipts provide source/runtime lineage without recursive training hash traversal.
    for rel in ('direct_target_evaluation_source_review_v1/review.json',
                'direct_target_student_evaluation_v1/head_witness/report.json',
                'direct_target_student_evaluation_v1/evaluation_completion_verification.json'):
        add(NEW/rel,'immutable previously reviewed runtime lineage; prior connected trial failed')
    return entries

def main():
    output=BASE/'runtime_inventory.json';assert not output.exists()
    entries=collect()
    prohibited=('direct_target_gpu_20260911','/generation/features.npy','/generation/teacher_target.npy',
                '/one_step_policy_branch_collection_resume2969_v1/collection/data/',
                '/pico_walk002_labels_v1/collection/')
    assert not any(any(part in path for part in prohibited) for path in entries)
    report=dict(kind='explicit_direct55000_fixed_runtime_inventory',preparation_only=True,
        files=list(entries.values()),total_files=len(entries),total_bytes=sum(x['bytes'] for x in entries.values()),
        source_directory=(BASE/'source_draft_v1').as_posix(),onnx_dependencies=DEPS.as_posix(),
        future_final_export_and_receipts_bound=False,recursive_training_hashes=False,
        package_allowlist=list(PACKAGES),limitations=[
            'Conservative imported-package code closure includes package files that may remain lazy; it is not a dynamic syscall trace.',
            'System CPython and OS shared libraries remain part of the established WSL runtime trust boundary; no new install or runtime process was invoked.',
            'Full centers/query250 archives remain because unchanged runtime gates actually read them; extracting smaller capsules is not part of this change.'
        ],model_calls=0,native_steps=0,optimizer_updates=0)
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(inventory_sha256=sha(output),files=len(entries),MiB=report['total_bytes']/2**20)))

if __name__=='__main__':main()
