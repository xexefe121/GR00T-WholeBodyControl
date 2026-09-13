import hashlib,json
from pathlib import Path
B=Path('E:/codex-artifacts/sonic23_teleop_resume_20260911/direct_target_fp64_export_v1')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
p=B/'proposal.json';assert sha(p)=='a1e805e2598c94a1047e44af7f8fffd59321660c6be0867c6242eb7730e36826'
j=json.loads(p.read_text());sources={}
for n,h in j['source_sha256'].items():
 q=B/'source_draft_v1'/n;assert sha(q)==h;sources[str(q)]=h
for n,h in j['support_sha256'].items():assert sha(B/'support'/n)==h
assert sha(B/'tests/report.json')==j['synthetic_tests_sha256']
assert sha(B/'PROPOSAL.md')==j['proposal_document_sha256']
r={'kind':'same55000_fp64_export_source_preparation_review','passed':True,'source_review_pass':True,'preparation_only':True,
 'proposal':{'path':str(p),'sha256':sha(p)},'source_sha256':sources,
 'reviewed_findings':[],
 'findings':['All source f32 weights/biases/normalization are exact f64 promotions; public features cast before normalization; final output casts to f32.',
 'Double native Torch ELU alpha1 is the reference. ONNX Where(x>0,x,Exp(Min(x,0))-1) matches its mathematical function and avoids unused-positive-branch exponential overflow. Rounding near zero need not be bit-identical; the selected full-corpus unchanged1e-5 rad gate measures it.',
 'The execution arithmetic changes from the failed FP32 export; this is explicitly a new numerical execution variant, without weight, architecture, normalization, objective or limit changes.',
 '9904+140622+3054 rows and independent batch256 partitions give601 calls per backend; two Torch passes1202 calls/307160 rows plus601 ORT calls/153580 rows. Manual graph construction needs no traced task forward.',
 'Reviewed five owner synthetic tests without repeating their tiny model calls. Actual full runner/request/launcher, attempt/return preservation, final promotion identity and source rehash require final concrete review.'],
 'synthetic_tests':{'path':str(B/'tests/report.json'),'sha256':j['synthetic_tests_sha256']},
 'actual_validation_cleared':False,'export_review_pass':False,'canonical_evaluation_cleared':False,
 'reviewer_model_calls':0,'reviewer_native_steps':0,'reviewer_optimizer_updates':0}
out=Path(__file__).parent/'review.json'
with out.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
print(sha(out))
