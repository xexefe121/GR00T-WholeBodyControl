"""Materialize root's selected single fit after completed data/source reviews."""
import argparse
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parent;NEW=ROOT.parent
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def local(p):
    text=str(p).replace('\\','/')
    if text.startswith('/mnt/') and len(text)>7:text=text[5].upper()+':'+text[6:]
    return Path(text).resolve()
parser=argparse.ArgumentParser();parser.add_argument('--source-review',type=Path,required=True)
parser.add_argument('--source-review-sha256',required=True);args=parser.parse_args()
source=args.source_review.resolve();assert sha(source)==args.source_review_sha256
source_data=read(source);assert source_data['source_review_pass'] is True
data_path=NEW/'one_step_branch_combined_data_review_v1/review.json'
assert sha(data_path)=='6faf42e89281ac155fdc35a6b38f5b5218c2d067318c46b7a883cafc1797d9ed'
data=read(data_path);assert data['data_review_pass'] is True
prior_path=NEW/'velocity_chord_final_export_review_v1/review.json'
assert sha(prior_path)=='dc0ba6b00a5daa073c0c62a24f4f443bfe24235881ea882bbc1e944adce55ac1'
prior=read(prior_path);assert prior['fit_review_passed'] is True and prior['export_review_passed'] is True
pins={}
def add(path,digest):
    p=local(path);assert sha(p)==digest,str(p)
    key=p.as_posix()
    if key in pins:assert pins[key]==digest
    pins[key]=digest
verification=local(data['verification']['path']);add(verification,data['verification']['sha256'])
verified=read(verification);assert verified['passed'] is True
for path,digest in verified['input_sha256'].items():add(path,digest)
for path,digest in data['direct_subject_sha256'].items():add(path,digest)
for path,digest in data['source_and_runtime_sha256'].items():add(path,digest)
for p in (data_path,source,prior_path,ROOT/'launcher_preparation_v1.json',Path(__file__)):
    add(p,sha(p))
for path,digest in read(ROOT/'launcher_preparation_v1.json')['source_sha256'].items():add(path,digest)
mapping=dict(collection_manifest='manifest',collection_report='producer_report',collection_request='producer_request',
    root_physics_report='root_physics_report',root_labels_report='root_labels_report',conflict_report='conflict_report')
paths={key:local(data['subjects'][role]['path']).as_posix() for key,role in mapping.items()}
reviews=dict(branch_data=dict(path=data_path.as_posix(),sha256=sha(data_path),pass_field='data_review_pass'),
    prior_fit=dict(path=prior_path.as_posix(),sha256=sha(prior_path),pass_field='fit_review_passed'),
    prior_export=dict(path=prior_path.as_posix(),sha256=sha(prior_path),pass_field='export_review_passed'),
    source=dict(path=source.as_posix(),sha256=sha(source),pass_field='source_review_pass'))
selection=dict(kind='one_physical_fit_selection',selected_by='root',model_fitting_authorized=True,
    authorization='Root selected one fixed ordinary70000 to ordinary75000 continuation after completed independent data checks; launch only after final frozen review.',
    additional_updates=5000,ordinary_final_step=75000,expected_valid_rows=3054,exact_conflict_groups=0,
    nominal_rows=3057,physical_requested_rows=3054,requested_cell_denominators=[99,819,100]*3,
    coefficients=dict(nominal=1.,velocity=1.,physical=1.),training_head_rows=36315000,head_ONNX_calls=1150,
    graph_BFM_calls=0,native_steps=0,automatic_optimizer_resume=False,checkpoint_selection=False,
    selected_source_directory=(ROOT/'source_draft_v2').as_posix(),
    paths=paths,reviews=reviews,input_sha256=pins,
    additional_data_subjects=[local(data['subjects'][key]['path']).as_posix() for key in (
        'root_physics_request','root_physics_segments','root_labels_request','unchanged_nominal_centers')],
    canonical_evaluation_selected=False)
out=ROOT/'selection_v1.json'
with out.open('x',encoding='utf-8') as f:json.dump(selection,f,indent=2);f.write('\n')
print(json.dumps(dict(selection_sha256=sha(out),input_pins=len(pins),fit_launched=False)))
