"""Preparation only: minimal persistence/startup continuation of fixed collection."""
import json,hashlib
from pathlib import Path
BASE=Path(__file__).resolve().parent;SRC=BASE/'draft';OLD=BASE.parent/'one_step_policy_branch_collection_v1'
changes=[]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def edit(name,old,new):
    p=SRC/name;text=p.read_text();assert text.count(old)==1,(name,old[:80]);text=text.replace(old,new);p.write_text(text)
    changes.append(dict(file=name,old=old,new=new))
edit('collection_arrays.py','import hashlib,json,os','import hashlib,json,os,time,shutil')
edit('collection_arrays.py','def atomic(path,value):',"def replace_with_sharing_retry(temp,path):\n    # Filesystem metadata retry only: same fully written file, no graph/native call.\n    for attempt in range(41):\n        try:temp.replace(path);return\n        except PermissionError:\n            if attempt==40:raise\n            time.sleep(.05)\n\ndef atomic(path,value):")
edit('collection_arrays.py','    tmp.replace(path)','    replace_with_sharing_retry(tmp,path)')
edit('collection_arrays.py','    def __init__(self,folder):','    def __init__(self,folder,resume_directory=None):')
old='''        for key,spec in self.definition.items():
            a=np.lib.format.open_memmap(self.folder/spec['path'],mode='w+',dtype=spec['dtype'],shape=tuple(spec['shape']))
            a.fill(np.nan if a.dtype.kind=='f' else 0);self.arrays[key]=a'''
new='''        if resume_directory is not None:
            prior=Path(resume_directory);manifest=json.loads((prior/'manifest.json').read_text())
            assert manifest['complete'] is False
            for key,spec in self.definition.items():
                previous=manifest['arrays'][key]
                assert all(previous[k]==spec[k] for k in ('path','shape','dtype'))
                assert sha(prior/spec['path'])==previous['sha256']
                shutil.copyfile(prior/spec['path'],self.folder/spec['path'])
                assert sha(self.folder/spec['path'])==previous['sha256']
                self.arrays[key]=np.lib.format.open_memmap(self.folder/spec['path'],mode='r+')
            shutil.copyfile(prior/'schema.json',self.folder/'schema.json')
            return
        for key,spec in self.definition.items():
            a=np.lib.format.open_memmap(self.folder/spec['path'],mode='w+',dtype=spec['dtype'],shape=tuple(spec['shape']))
            a.fill(np.nan if a.dtype.kind=='f' else 0);self.arrays[key]=a'''
edit('collection_arrays.py',old,new)
edit('collect_branches.py','from collection_arrays import Store,ROWS,HISTORY_WIDTHS,sha,atomic','from collection_arrays import Store,ROWS,HISTORY_WIDTHS,sha,atomic,replace_with_sharing_retry\nfrom resume_prefix import restore_prefix,PREFIX_ROWS,PREFIX_STEPS')
edit('collect_branches.py',"    temp.replace(DEST/'active.npz')","    replace_with_sharing_retry(temp,DEST/'active.npz')")
text=(SRC/'collect_branches.py').read_text();start=text.index("        STORE=Store(DEST/'data')");stop=text.index('        from native_capture import CapturedForecast',start)
edit('collect_branches.py',text[start:stop],"        STORE=Store(DEST/'data',resume_directory=paths['prefix_data'])\n        restoration=restore_prefix(STORE,inputs,paths,DEST)\n        PROGRESS['nominal_verified']=PREFIX_ROWS\n        PROGRESS['native_attempted']=PROGRESS['native_returned']=PREFIX_STEPS\n        progress()\n")
edit('collect_branches.py','        ENGINE=CapturedForecast(model,contract)','        ENGINE=CapturedForecast(model,contract)\n        ENGINE.total_attempted=ENGINE.total_returned=PREFIX_STEPS')
edit('collect_branches.py',"        for row,(dataset,control,index) in enumerate(inputs.rows):\n            PROGRESS['active_row']=row;ACTIVE.clear();", "        for row,(dataset,control,index) in enumerate(inputs.rows):\n            if row<PREFIX_ROWS:continue\n            PROGRESS['active_row']=row;ACTIVE.clear();")
edit('collect_branches.py',"        assert request['graph_call_ceiling']==BUDGET and request['native_step_ceiling']==61080 and request['rows']==ROWS", "        assert request['graph_call_ceiling']==BUDGET and request['native_step_ceiling']==61080 and request['rows']==ROWS") if False else None
edit('collect_branches.py',"    assert request['graph_call_ceiling']==BUDGET and request['native_step_ceiling']==61080 and request['rows']==ROWS", "    assert request['graph_call_ceiling']==BUDGET and request['native_step_ceiling']==61080 and request['rows']==ROWS\n    assert request['prefix_nominal_rows']==PREFIX_ROWS and request['prefix_native_steps']==PREFIX_STEPS")
edit('collect_branches.py','            graph_calls=COUNTS,native_attempted=ENGINE.total_attempted,native_returned=ENGINE.total_returned,','            graph_calls=COUNTS,native_attempted=ENGINE.total_attempted,native_returned=ENGINE.total_returned,\n            continuation_attempt=2,prefix_native_steps=PREFIX_STEPS,current_attempt_native_steps=ENGINE.total_returned-PREFIX_STEPS,restoration=restoration,')
edit('collect_branches.py',"            request_sha256=request_sha,clearance_sha256=args.clearance_sha256,", "            request_sha256=request_sha,clearance_sha256=args.clearance_sha256,") if False else None
edit('collect_branches.py',"            query250_actual251_calibration_passed=True,all_frozen_inputs_unchanged=True,request_sha256=request_sha,clearance_sha256=args.clearance_sha256,", "            query250_actual251_calibration_passed=True,all_frozen_inputs_unchanged=True,request_sha256=request_sha,clearance_sha256=args.clearance_sha256,\n            continuation_attempt=2,prefix_nominal_rows=PREFIX_ROWS,prefix_native_steps=PREFIX_STEPS,current_attempt_native_steps=ENGINE.total_returned-PREFIX_STEPS,restoration=restoration,")
launcher=(OLD/'run_durable.ps1').read_text().replace('one_step_policy_branch_collection_v1','one_step_policy_branch_collection_resume2969_v1')
(BASE/'run_durable.ps1').write_text(launcher)
receipt=dict(original_source_directory=(OLD/'source_snapshot_v1').as_posix(),substitutions=changes,
    source_sha256={p.relative_to(SRC).as_posix():sha(p) for p in SRC.rglob('*.py')},
    prefix_rows=2969,prefix_native_steps=29690,remaining_native_ceiling=31390,remaining_graph_ceiling=12216,
    no_changed_math=True,no_repeated_committed_physics=True,filesystem_retry_only=True)
(BASE/'derivation.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(dict(substitutions=len(changes),launcher_sha256=sha(BASE/'run_durable.ps1'))))
