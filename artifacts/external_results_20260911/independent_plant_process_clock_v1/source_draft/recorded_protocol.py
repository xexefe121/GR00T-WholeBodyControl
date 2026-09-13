"""Literal recorded-command jobs. No inference, plant, native or file operations."""
from dataclasses import asdict
import base64
import json
import numpy as np
from clock_core import Binding, Command, Job, Result, encode, digest
from history import SIZES


def typed(a, dtype, shape, name):
    if type(a) is not np.ndarray or a.dtype != np.dtype(dtype) or a.shape != shape or not np.isfinite(a).all():
        raise ValueError(name+' shape/dtype/finite contract')
    return a


class CommandTable:
    """Owned immutable command bytes. Tiny tables are for stub tests only."""
    def __init__(self, targets, actions, source_frames, origins, limits):
        n=len(targets)
        if n<1:raise ValueError('nonempty fixed command table')
        typed(targets,np.float64,(n,23),'targets');typed(actions,np.float32,(n,23),'actions')
        typed(source_frames,np.int64,(n,),'source frames');typed(limits,np.float64,(23,2),'limits')
        if np.any(source_frames<0) or np.any(limits[:,0]>limits[:,1]) or np.any(targets<limits[:,0]) or np.any(targets>limits[:,1]):
            raise ValueError('unchanged native limits required; no clamp')
        if len(origins)!=n or any(type(s) is not str or len(s)!=64 or any(c not in '0123456789abcdef' for c in s) for s in origins):
            raise ValueError('one literal original trace SHA256 per command')
        self.commands=tuple(Command.make('saved:'+str(c)+':'+origins[c], 'recorded-query250:'+origins[c],targets[c],actions[c]) for c in range(n))
        self.frames=tuple(int(v) for v in source_frames)
        self.sha256=digest(encode(dict(commands=[c.wire() for c in self.commands],frames=self.frames)))
        self.windows=tuple('recorded:'+str(c)+':frame:'+str(self.frames[c])+':'+self.sha256 for c in range(n))


def canonical_table(full, hold, full_sha, hold_sha, limits, fixture):
    """Validate the original1569 + continuous250 contract before any future epoch.

    Callers must bind these already-qualified input archives and fixture by hash.
    This function never reconstructs, resets or replaces any simulated state.
    """
    for segment,n,start in [(full,1569,0),(hold,250,1569)]:
        for key,dtype,shape in [('target',np.float64,(n,23)),('action',np.float32,(n,23)),
            ('previous_action',np.float32,(n,23)),('control_integration_before',np.float64,(n,291)),
            ('control_history_before',np.float32,(n,300)),('initial_integration',np.float64,(291,)),
            ('final_integration',np.float64,(291,))]:typed(segment[key],dtype,shape,key)
        if segment['global_control'].dtype!=np.int64 or not np.array_equal(segment['global_control'],np.arange(start,start+n,dtype=np.int64)):
            raise ValueError('original complete global control coverage')
        if segment['physics_substeps'].dtype!=np.int64 or not np.array_equal(segment['physics_substeps'],np.full(n,10,np.int64)):
            raise ValueError('all selected controls own ten original steps')
        if int(segment['integration_state_spec'])!=8191:raise ValueError('full291 spec8191')
    typed(fixture,np.float64,(291,),'fixture')
    if fixture.tobytes()!=full['initial_integration'].tobytes() or fixture.tobytes()!=full['control_integration_before'][0].tobytes():
        raise ValueError('canonical full291 initial parity')
    if full['final_integration'].tobytes()!=hold['initial_integration'].tobytes() or full['final_integration'].tobytes()!=hold['control_integration_before'][0].tobytes():
        raise ValueError('original full291 main-to-hold continuity')
    if np.any(full['control_history_before'][0]) or full['previous_action'][0].tobytes()!=np.zeros(23,np.float32).tobytes():
        raise ValueError('unchanged foundation requires canonical zero history/prior')
    if full['action'][-1].tobytes()!=hold['previous_action'][0].tobytes():raise ValueError('hold prior continuity')
    full_history=np.concatenate([full['final_history_'+name].reshape(-1) for name in sorted(SIZES)])
    if full_history.tobytes()!=hold['control_history_before'][0].tobytes():raise ValueError('hold four-lag history continuity')
    if not np.array_equal(full['source_frame'],np.arange(11,1580,dtype=np.int64)) or not np.array_equal(hold['source_frame'],np.full(250,1579,np.int64)):
        raise ValueError('original source indices and held EOF required')
    return CommandTable(np.concatenate([full['target'],hold['target']]),np.concatenate([full['action'],hold['action']]),
                        np.concatenate([full['source_frame'],hold['source_frame']]),[full_sha]*1569+[hold_sha]*250,limits)


def _bytes(value,length):
    if type(value) is not str:raise ValueError('base64 string required')
    data=base64.b64decode(value,validate=True)
    if len(data)!=length:raise ValueError('typed byte size differs')
    return data


def decode_job(payload,binding,table):
    """Validate exact job identity and all snapshot byte schemas without a model."""
    if type(payload) is not bytes or not 0<len(payload)<=32768:raise ValueError('bounded immutable job')
    value=json.loads(payload)
    expected={'binding','sequence','snapshot_control','snapshot_physics','activation','window_id','input_digest',
              'history_digest','schedule_digest','created_ns','snapshot_payload'}
    if type(value) is not dict or set(value)!=expected:raise ValueError('literal job fields')
    if encode(value['binding'])!=encode(asdict(binding)):raise ValueError('wrong binding or epoch')
    for key in ['sequence','snapshot_control','snapshot_physics','activation','created_ns']:
        if type(value[key]) is not int or value[key]<0:raise ValueError('nonnegative literal job integer')
    c=value['snapshot_control'];a=value['activation']
    if not 0<=c<len(table.commands)-1 or a!=c+1 or value['sequence']!=c or value['snapshot_physics']!=c*10:
        raise ValueError('one-control future activation identity')
    if value['window_id']!=table.windows[a]:raise ValueError('wrong admitted source window')
    body=base64.b64decode(value['snapshot_payload'],validate=True)
    if digest(body)!=value['input_digest']:raise ValueError('snapshot digest mismatch')
    snapshot=json.loads(body)
    if set(snapshot)!={'state','incoming_raw','active_command','history_before','terms','history_after'}:
        raise ValueError('literal snapshot fields')
    state=_bytes(snapshot['state'],291*8)
    if not np.isfinite(np.frombuffer(state,np.float64)).all():raise ValueError('nonfinite full291 snapshot')
    incoming=_bytes(snapshot['incoming_raw'],23*4)
    if not np.isfinite(np.frombuffer(incoming,np.float32)).all():raise ValueError('nonfinite incoming prior')
    if snapshot['active_command']!=table.commands[c].wire():raise ValueError('active command differs from recorded benchmark source')
    if digest(encode(snapshot['active_command']))!=value['schedule_digest']:raise ValueError('active schedule digest')
    history_before=[];decoded={}
    for field,lags in [('history_before',4),('terms',1),('history_after',4)]:
        fields=snapshot[field]
        if type(fields) is not list or [x[0] for x in fields]!=sorted(SIZES):raise ValueError('history fields/order')
        decoded[field]={}
        for key,text in fields:
            raw=_bytes(text,lags*SIZES[key]*4)
            if not np.isfinite(np.frombuffer(raw,np.float32)).all():raise ValueError('nonfinite history')
            decoded[field][key]=raw
            if field=='history_before':history_before.append(raw)
    if digest(b''.join(history_before))!=value['history_digest']:raise ValueError('flat-history digest')
    if decoded['terms']['actions']!=incoming:raise ValueError('measured history must use actual incoming action')
    for key,size in SIZES.items():
        before=np.frombuffer(decoded['history_before'][key],np.float32).reshape(4,size)
        expected=np.concatenate([np.frombuffer(decoded['terms'][key],np.float32)[None],before[:3]])
        if expected.tobytes()!=decoded['history_after'][key]:raise ValueError('four-lag update order changed')
    job=Job(binding,c,c,c*10,a,value['window_id'],body,value['history_digest'],value['schedule_digest'],value['created_ns'])
    if job.to_bytes()!=payload:raise ValueError('canonical job serialization required')
    return job


def reply_to_job(payload,binding,table,completed_ns):
    job=decode_job(payload,binding,table)
    if type(completed_ns) is not int or completed_ns<job.created_ns:raise ValueError('worker completion before job')
    return Result.for_job(job,table.commands[job.activation],completed_ns).to_bytes()
