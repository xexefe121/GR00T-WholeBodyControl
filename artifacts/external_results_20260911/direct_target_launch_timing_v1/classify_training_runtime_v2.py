"""Refine preserved conservative size report using its existing saved file list."""
import json
import hashlib
from pathlib import Path
P=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
report=json.loads((P/'report.json').read_text())
for mode in ('witness','evaluation'):
    stage=report[mode];groups={}
    for entry in stage['files']:
        if '/direct_target_gpu_20260911/' in entry['path'].replace('\\','/').lower():
            entry['category']='known_transitive_Windows_training_runtime'
        g=groups.setdefault(entry['category'],dict(files=0,bytes=0));g['files']+=1;g['bytes']+=entry['bytes']
    stage['hash_pass_groups']=groups
    stage['known_transitive_training_bytes']=sum(v['bytes'] for k,v in groups.items() if k.startswith('known_transitive_'))
    stage['known_transitive_fraction']=stage['known_transitive_training_bytes']/stage['launch_pin_bytes']
report['input_sha256'].update({str(P/'report.json'):sha(P/'report.json'),str(Path(__file__)):sha(__file__)})
report['conclusions'].append('Windows CUDA trainer libraries are not used by the selected WSL CPU ONNX evaluator. Their inherited pins dominate the conservative other category; this v2 classifies them explicitly, with all original timestamps unchanged.')
with (P/'report_v2.json').open('x',encoding='utf-8') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
r=report['evaluation']
print(json.dumps(dict(report_sha256=sha(P/'report_v2.json'),known_transitive_GiB=r['known_transitive_training_bytes']/2**30,
    known_fraction=r['known_transitive_fraction'],training_runtime_GiB=r['hash_pass_groups']['known_transitive_Windows_training_runtime']['bytes']/2**30)))
