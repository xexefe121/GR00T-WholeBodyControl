"""Copy reviewed adapter and derive original model preparation; zero native calls."""
import ast,hashlib,json,shutil
from pathlib import Path
BASE=Path(__file__).resolve().parent;NEW=BASE.parent
ORIGINAL=NEW/'independent_native_stepper_v1/source_draft_v3'
SOURCE=BASE/'source_draft_v1'
MODEL_ORIGINAL=NEW/'direct_target_student_evaluation_v1/source_draft_v1/gear_sonic/utils/g1_true23_mjbatch_mpc.py'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    SOURCE.mkdir(exist_ok=False)
    review=NEW/'independent_native_stepper_root_review_v2/review_v3.json'
    r=json.loads(review.read_text());assert r['source_review_passed'] is True
    copies={}
    for name in ('native_stepper.py','model_identity.py','capture_schema.py','clock_core.py','history.py','mailbox.py','bfm_observations.py','oracle_source.py'):
        p=ORIGINAL/name;assert r['source_and_evidence_pins'][str(p).replace('/','\\')]==sha(p)
        shutil.copyfile(p,SOURCE/name);copies[name]=dict(original=p.as_posix(),sha256=sha(p))
    tree=ast.parse(MODEL_ORIGINAL.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='load_native_bundle')
    keep=[]
    for node in fn.body:
        if isinstance(node,ast.With) and 'np.load(motion_path' in ast.unparse(node):break
        keep.append(node)
    derived=ast.FunctionDef(name='load_model',args=fn.args,body=keep+[ast.Return(value=ast.Tuple(elts=[ast.Name(id='model',ctx=ast.Load()),ast.Name(id='contract',ctx=ast.Load())],ctx=ast.Load()))],decorator_list=[])
    ast.fix_missing_locations(derived)
    code='"""Original native bundle model preparation; imported only by selected runner."""\nimport hashlib,json\nfrom pathlib import Path\nimport mujoco\nimport numpy as np\n\ndef sha256(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()\n\n'+ast.unparse(derived)+'\n'
    (SOURCE/'native_loader.py').write_text(code)
    report=dict(preparation_only=True,adapter_review=dict(path=review.as_posix(),sha256=sha(review)),copies=copies,
        loader_original=dict(path=MODEL_ORIGINAL.as_posix(),sha256=sha(MODEL_ORIGINAL)),
        loader_derivation='exact load_native_bundle statements through contract/topology checks; omit unused motion decoding, return model and contract',
        loader_sha256=sha(SOURCE/'native_loader.py'),model_calls=0,native_steps=0)
    (BASE/'source_derivation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(copies=len(copies),derivation_sha256=sha(BASE/'source_derivation.json'))))

if __name__=='__main__':main()
