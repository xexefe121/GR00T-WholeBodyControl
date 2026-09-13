"""Saved identity/completeness checks only; independent span math remains separate."""
import hashlib,json
from pathlib import Path

FILES=('timing_spans.bin','timing_gc.bin','timing_metadata.json','timing_owned_partial.json')

def check_sidecars(output,report,manifest):
    output=Path(output).resolve();info=report['timing_instrumentation']
    if type(info.get('enabled')) is not bool or type(info.get('complete')) is not bool:
        raise ValueError('Literal timing enablement and completion required')
    files=info.get('files');errors=info.get('errors')
    if type(files) is not dict or not set(files)<=set(FILES) or type(errors) is not list:
        raise ValueError('Exact sidecar file/error schema required')
    if info['enabled'] and (type(info.get('new_native_reads')) is not int or info['new_native_reads']!=0):
        raise ValueError('Sidecar preservation may not create native reads')
    owned={};unclaimed={}
    for name in FILES:
        path=output/name
        if name not in files and not path.exists():continue
        if not path.is_file() or not path.resolve().is_relative_to(output):
            raise ValueError('Declared sidecar missing or escaping output')
        entry={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}
        if manifest['files'].get(name)!=entry:raise ValueError('Sidecar differs from main output manifest')
        if name in files:
            if files[name]!=entry:raise ValueError('Sidecar differs from report identity')
            owned[name]=entry
        else:
            if not errors:raise ValueError('Unclaimed partial sidecar needs writer failure evidence')
            unclaimed[name]=entry
    if not info['enabled']:
        if files or unclaimed or info['complete']:raise ValueError('Disabled timing claimed evidence or completion')
    elif set(files)!=set(FILES) and not errors:
        raise ValueError('Missing sidecars need explicit preservation failure')
    metadata=None
    if 'timing_metadata.json' in files:
        metadata=json.loads((output/'timing_metadata.json').read_text(encoding='utf-8'))
        if type(metadata.get('instrumentation_complete')) is not bool:
            raise ValueError('Literal saved probe completion required')
        if metadata.get('hook_status')!=info.get('hook_status'):
            raise ValueError('Saved metadata/report hooks differ')
    hooks=info.get('hook_status')
    if info['enabled'] and (type(hooks) is not dict or type(hooks.get('failed')) is not bool):
        raise ValueError('Literal hook failure flag required')
    complete=bool(info['enabled'] and metadata is not None and metadata['instrumentation_complete'] and
        not hooks['failed'] and not errors and set(files)==set(FILES))
    if info['complete'] is not complete:raise ValueError('Reported sidecar completeness differs')
    if report.get('component_preliminary_pass') is True and not complete:
        raise ValueError('Incomplete sidecars cannot qualify preliminary timing')
    return dict(evidence_accounted=True,instrumentation_complete=complete,
        declared_sidecar_files=owned,unclaimed_partial_files=unclaimed,
        missing_sidecar_files=sorted(set(FILES)-set(files)),preservation_errors=errors,
        probe_complete=metadata['instrumentation_complete'] if metadata is not None else None,
        independent_sidecar_math_pending=True,new_native_reads=0,
        verification_return_or_native_credit_inferred=False)
