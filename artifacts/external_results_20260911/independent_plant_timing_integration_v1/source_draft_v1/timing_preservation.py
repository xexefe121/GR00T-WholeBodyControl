"""Outside-loop sidecar preservation; no native reads and no runtime imports."""
import hashlib,json
from pathlib import Path

def save_timing(output,hooks,adapter,foundation,lossless):
    output=Path(output);files={};errors=[]
    def put(name,data):
        path=output/name
        with path.open('xb') as f:f.write(data)
        files[name]={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
    def put_json(name,data):put(name,json.dumps(data,indent=2,allow_nan=False).encode())
    if hooks is None:return dict(enabled=False,complete=False,files={},errors=[])
    probe=hooks.probe
    if probe is None:return dict(enabled=False,complete=False,files={},errors=[])
    snapshot=None
    try:snapshot=probe.snapshot()
    except BaseException as exc:errors.append({'phase':'snapshot','type':type(exc).__name__,'detail':str(exc)})
    if snapshot is not None:
        for field,name in (('spans','timing_spans.bin'),('gc','timing_gc.bin')):
            try:put(name,snapshot.pop(field))
            except BaseException as exc:errors.append({'phase':field,'type':type(exc).__name__,'detail':str(exc)})
        try:put_json('timing_metadata.json',dict(snapshot,hook_status=hooks.status()))
        except BaseException as exc:errors.append({'phase':'metadata','type':type(exc).__name__,'detail':str(exc)})
    # Retain already-owned values if a timing interrupt occurred between original
    # call return and the foundation's later ownership/count assignments.
    owned={}
    for role,obj in (('adapter',adapter),('foundation',foundation)):
        if obj is not None:
            owned[role]={}
            for name in ('failure','last_capture','last_capture_return','last_verifier_return','last_assessment'):
                value=getattr(obj,name,None)
                if value is not None:
                    try:owned[role][name]=lossless(value)
                    except BaseException as exc:errors.append({'phase':'owned_partial_'+role+'_'+name,'type':type(exc).__name__,'detail':str(exc)})
    try:put_json('timing_owned_partial.json',owned)
    except BaseException as exc:errors.append({'phase':'owned_partial','type':type(exc).__name__,'detail':str(exc)})
    return dict(enabled=True,complete=bool(snapshot is not None and snapshot['instrumentation_complete'] and not hooks.failed and not errors),
        files=files,errors=errors,hook_status=hooks.status(),new_native_reads=0)
