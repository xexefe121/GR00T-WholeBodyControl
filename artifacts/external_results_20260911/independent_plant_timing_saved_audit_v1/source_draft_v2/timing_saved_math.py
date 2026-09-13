"""Independent signed64 sidecar decoder. No producer, clocks or native imports."""
import hashlib
import struct
import numpy as np

SPAN_FIELDS=('phase','tick','control','thread','parent','return_kind','start_wall_before',
 'start_thread_cpu','start_process_cpu','start_wall_after','end_wall_before','end_thread_cpu',
 'end_process_cpu','end_wall_after','begin_complete','end_complete')
GC_FIELDS=('phase','generation','thread','active_span','inside_probe','collected','uncollectable',
 'wall_before','thread_cpu','process_cpu','wall_after','sample_complete')
PHASES=('TICK_ENVELOPE','RESULT_POLL_AND_ADMIT','FIXED_WAIT','BOUNDARY_SNAPSHOT',
 'BOUNDARY_HISTORY','BOUNDARY_SERIALIZATION','JOB_PUBLICATION','PD_AND_INPUT','MJ_STEP',
 'CAPTURE','CAPTURE_OWNERSHIP','VERIFY','VERIFY_OWNERSHIP','STEP_LEDGER','OUTER_LOG')
FAULTS=('NONE','CAPACITY','BAD_ARGUMENT','CLOCK_EXCEPTION','CLOCK_VALUE','CLOCK_REGRESSION',
 'STACK_ORDER','RETURN_MARK','GC_SCHEMA','GC_PAIR','GC_REGISTRY','FOREIGN_SPAN_THREAD')
MAX_SPANS=262144;MAX_GC=4096;I64=(1<<63)-1

def integer(value,minimum=0,maximum=I64):
    if type(value) is not int or not minimum<=value<=maximum:raise AssertionError('Literal bounded integer required')
    return value

def decode(raw,fields,count,metadata):
    if type(raw) is not bytes or len(raw)!=count*len(fields)*8:raise AssertionError('Sidecar byte count')
    if metadata['integer_dtype']!='q' or metadata['integer_itemsize']!=8 or metadata['integer_byteorder'] not in ('little','big'):
        raise AssertionError('Signed64 sidecar representation')
    codec=struct.Struct(('<' if metadata['integer_byteorder']=='little' else '>')+'q'*len(fields))
    return [dict(zip(fields,row)) for row in codec.iter_unpack(raw)]

def sample(row,keys,complete):
    for key in keys:integer(row[key],-1)
    if complete==1:
        if any(row[k]<0 for k in keys) or row[keys[3]]<row[keys[0]]:raise AssertionError('Local completed clock bracket')

def duration(row):
    return {'tick':row['tick'],'return_kind':row['return_kind'],
        'wall_lower_bound_ns':row['end_wall_before']-row['start_wall_after'],
        'wall_upper_bound_ns':row['end_wall_after']-row['start_wall_before'],
        'thread_cpu_delta_ns':row['end_thread_cpu']-row['start_thread_cpu'],
        'process_cpu_delta_ns':row['end_process_cpu']-row['start_process_cpu']}

def counter_gap_proof(report,rows,owned):
    """Allow only one interrupted terminal handoff, never create native credit."""
    if report['session'] is None:return {'partial_counter_proof':False,'gaps':{}}
    core=report['session']['foundation'];native=report['session']['stepper'];api=report['api_counters']
    for key in ('step_attempted','step_returned'):integer(api[key],0,18190)
    if api['step_returned']>api['step_attempted']:raise AssertionError('API return exceeds attempts')
    gaps={
      'api_attempt':native['attempted']-api['step_attempted'],
      'api_return':api['step_returned']-native['returned'],
      'foundation_return':native['returned']-core['returned'],
      'foundation_capture':native['captured']-core['captured'],
      'foundation_verify':native['verified']-core['verified']}
    if any(type(v) is not int or v not in (0,1) for v in gaps.values()):raise AssertionError('Unbounded or reversed terminal counter handoff')
    if not any(gaps.values()):return {'partial_counter_proof':False,'gaps':gaps}
    terminal=core['committed']
    if report['first_error'] is None or not rows:raise AssertionError('Counter gap lacks fatal terminal evidence')
    if any(row['tick']>terminal for row in rows):raise AssertionError('Continued after partial handoff')
    phases={row['phase'] for row in rows if row['tick']==terminal}
    terminal_rows={row['phase']:row for row in rows if row['tick']==terminal}
    root=terminal_rows.get(1)
    if root is None or root['return_kind']==1:raise AssertionError('Counter gap followed a normal complete tick')
    for key,phase in [('api_attempt',9),('api_return',9),('foundation_return',9),('foundation_capture',10),('foundation_verify',12)]:
        if gaps[key] and phase not in phases:raise AssertionError('Counter gap lacks corresponding started scope')
    for key,phase in [('api_attempt',9),('api_return',9),('foundation_capture',10),('foundation_verify',12)]:
        if gaps[key] and terminal_rows[phase]['return_kind']==1:raise AssertionError('Completed scope contradicts missing ownership counter')
    # Capture/verification credit lives in actual already-owned adapter evidence.
    if gaps['foundation_capture'] and owned.get('adapter',{}).get('last_capture') is None:
        raise AssertionError('Adapter capture gap lacks owned capture')
    if gaps['foundation_verify'] and owned.get('adapter',{}).get('last_assessment') is None:
        raise AssertionError('Adapter verification gap lacks owned assessment')
    return {'partial_counter_proof':True,'gaps':gaps,'native_credit_inferred':0,
        'partial_native_state_may_be_unavailable':bool(gaps['api_return'] or gaps['foundation_return'])}

def audit(metadata,span_bytes,gc_bytes,report,stages,arrays,owned):
    if type(metadata['schema_version']) is not int or metadata['schema_version']!=1 or tuple(metadata['span_fields'])!=SPAN_FIELDS or tuple(metadata['gc_fields'])!=GC_FIELDS or tuple(metadata['phases'])!=PHASES:
        raise AssertionError('Literal reviewed sidecar schema')
    ns=integer(metadata['spans_started'],0,MAX_SPANS);ng=integer(metadata['gc_recorded'],0,MAX_GC)
    for key in ('spans_attempted','spans_closed','spans_denied','gc_attempted','gc_denied'):integer(metadata[key])
    if metadata['spans_attempted']<ns+metadata['spans_denied'] or metadata['gc_attempted']!=ng+metadata['gc_denied']:
        raise AssertionError('Attempt/record/denial counts differ')
    rows=decode(span_bytes,SPAN_FIELDS,ns,metadata);gc=decode(gc_bytes,GC_FIELDS,ng,metadata)
    fault=metadata['fault']
    if fault not in FAULTS:raise AssertionError('Unknown probe fault')
    integer(metadata['fault_record'],-1,max(ns,ng)-1)
    if metadata['no_native_or_timing_qualification'] is not True:raise AssertionError('Probe incorrectly claims runtime qualification')
    if type(metadata['open_gc_generations']) not in (list,tuple) or len(metadata['open_gc_generations'])!=3:raise AssertionError('Open GC generation schema')
    for value in metadata['open_gc_generations']:integer(value,-1,ng-1)
    clean=fault=='NONE'
    hooks=metadata['hook_status'];report_hooks=report['timing_instrumentation']['hook_status']
    if hooks!=report_hooks or hooks['enabled'] is not True or hooks['probe_fault_code']!=FAULTS.index(fault):raise AssertionError('Hook/report fault identity differs')
    if hooks['spans_started']!=ns or hooks['spans_closed']!=metadata['spans_closed'] or hooks['gc_recorded']!=ng:raise AssertionError('Hook/row counts differ')
    integer(hooks['active_scopes'],0,16);integer(hooks['hook_faults'])
    first=hooks['first_hook_fault']
    if (first is None)!=(hooks['hook_faults']==0):raise AssertionError('Hook first-fault count differs')
    if first is not None:
        if type(first) not in (list,tuple) or len(first)!=2 or first[0] not in ('begin','fatal_begin','end','fatal_end','scope_capacity'):raise AssertionError('Unknown hook failure')
        integer(first[1],-1,max(-1,ns-1))
    if hooks['failed'] is not bool(first is not None or not clean):raise AssertionError('Hook failed flag differs')
    roots=[];by_tick={};closed=0;thread=None;last_root=None
    for i,row in enumerate(rows):
        integer(row['phase'],1,15);integer(row['tick'],0,18190);integer(row['control'],0,1819);integer(row['thread'],1)
        integer(row['parent'],-1,i-1);integer(row['return_kind'],0,2)
        for name in ('begin_complete','end_complete'):integer(row[name],-1,1)
        if row['control']!=row['tick']//10:raise AssertionError('Tick/control identity differs')
        if thread is None:thread=row['thread']
        if row['thread']!=thread:raise AssertionError('Span changes owner thread')
        if row['phase']==1:
            if row['parent']!=-1 or row['tick']!=len(roots):raise AssertionError('Root span skipped/repeated/reordered')
            roots.append(i)
            if last_root is not None and last_root['end_complete']==row['begin_complete']==1:
                if row['start_wall_before']<last_root['end_wall_after'] or row['start_thread_cpu']<last_root['end_thread_cpu'] or row['start_process_cpu']<last_root['end_process_cpu']:
                    raise AssertionError('Completed root clock regression')
            last_root=row
        elif row['parent']<0 or not roots:raise AssertionError('Non-root scope lacks parent')
        elif rows[row['parent']]['tick']!=row['tick'] or rows[row['parent']]['thread']!=row['thread']:raise AssertionError('Parent clock/thread identity differs')
        if row['phase']==11 and rows[row['parent']]['phase']!=10:raise AssertionError('Capture ownership parent')
        if row['phase']==13 and rows[row['parent']]['phase']!=12:raise AssertionError('Verify ownership parent')
        if row['phase'] not in (1,11,13) and rows[row['parent']]['phase']!=1:raise AssertionError('Phase has unexpected parent')
        phases=by_tick.setdefault(row['tick'],{})
        if row['phase'] in phases:raise AssertionError('Repeated scope in a tick')
        if phases and row['phase']<=max(phases):raise AssertionError('Scope operation order differs')
        phases[row['phase']]=row
        if row['phase'] in (4,5,6) and row['tick']%10:raise AssertionError('Boundary phase at non-boundary tick')
        sample(row,SPAN_FIELDS[6:10],row['begin_complete']);sample(row,SPAN_FIELDS[10:14],row['end_complete'])
        if row['end_complete']==1:
            closed+=1
            if row['return_kind']==0:raise AssertionError('Completed end lacks return marker')
            if row['begin_complete']==1 and (row['end_wall_before']<row['start_wall_after'] or row['end_thread_cpu']<row['start_thread_cpu'] or row['end_process_cpu']<row['start_process_cpu']):
                raise AssertionError('Paired span clock regression')
        if clean and (row['begin_complete']!=1 or row['end_complete']!=1 or row['return_kind']==0) and first is None:
            raise AssertionError('Partial span with no probe/hook failure')
    if closed!=metadata['spans_closed']:raise AssertionError('Closed span count differs')
    for row in rows:
        if row['parent']>=0:
            parent=rows[row['parent']]
            if row['begin_complete']==parent['begin_complete']==1 and row['start_wall_before']<parent['start_wall_after']:
                raise AssertionError('Child starts before parent body')
            if row['end_complete']==parent['end_complete']==1 and row['end_wall_after']>parent['end_wall_before']:
                raise AssertionError('Child ends after parent body')
            for clock in ('thread_cpu','process_cpu'):
                if row['begin_complete']==parent['begin_complete']==1 and row['start_'+clock]<parent['start_'+clock]:
                    raise AssertionError('Child CPU starts before parent sampled clock')
                if row['end_complete']==parent['end_complete']==1 and row['end_'+clock]>parent['end_'+clock]:
                    raise AssertionError('Child CPU ends after parent sampled clock')
    # Sequential scopes on this same thread/parent cannot overlap. This compares
    # explicit call ordering, never callback-updated global timestamps.
    previous_sibling={}
    for row in rows:
        parent=row['parent'];previous=previous_sibling.get(parent)
        if previous is not None and previous['end_complete']==row['begin_complete']==1:
            if previous['end_wall_after']>row['start_wall_before']:
                raise AssertionError('Sequential sibling wall overlap')
            for clock in ('thread_cpu','process_cpu'):
                if previous['end_'+clock]>row['start_'+clock]:
                    raise AssertionError('Sequential sibling CPU regression')
        previous_sibling[parent]=row
    depth=integer(metadata['active_depth'],0,16)
    # Counts are independent of parent/child duration sums.
    if report['session'] is not None:
        core=report['session']['foundation'];native=report['session']['stepper'];api=report['api_counters']
        for phase,counter in [(9,native['attempted']),(10,native['capture_attempts']),(12,native['verification_attempts'])]:
            actual=sum(phase in phases for phases in by_tick.values())
            if not counter<=actual<=counter+1:raise AssertionError('Scope/native attempt count mismatch')
        n=len(arrays['step_index'])
        if not n<=len(roots)<=n+1:raise AssertionError('Root span/native prefix scope')
        for index in range(n):
            phases=by_tick.get(index,{})
            if any(k not in phases for k in (1,2,3,8,9,10,11,12,13,14)):raise AssertionError('Committed sample lacks reached instrumentation scopes')
            if phases[9]['return_kind']!=1 or phases[11]['return_kind']!=1:raise AssertionError('Committed native/capture lacks normal scope return')
            if phases[3]['end_complete']==1 and phases[3]['end_wall_after']>int(arrays['step_actual_start'][index]):raise AssertionError('Wake recorded before wait end')
            if phases[13]['end_complete']==1 and phases[13]['end_wall_after']>int(arrays['step_actual_end'][index]):raise AssertionError('Finish recorded before verify ownership end')
            if phases[14]['begin_complete']==1 and int(arrays['step_actual_end'][index])>phases[14]['start_wall_before']:raise AssertionError('Step append before original finish timestamp')
            epoch=integer(report['epoch_ns'])
            if int(arrays['step_nominal_start'][index])!=epoch+index*2000000 or int(arrays['step_nominal_end'][index])!=epoch+(index+1)*2000000:raise AssertionError('Original absolute step deadline changed')
        # Extra uncommitted native work never receives complete trace credit.
        if api['step_returned']>n+1:raise AssertionError('Unpreserved multiple native returns')
    open_gc=[-1,-1,-1];gc_partial=False;gc_pairs=[]
    for i,row in enumerate(gc):
        integer(row['sample_complete'],-1,1)
        sample(row,GC_FIELDS[7:11],row['sample_complete'])
        if row['sample_complete']!=1 or row['phase'] not in (1,2) or row['generation'] not in (0,1,2):
            if clean:raise AssertionError('Invalid GC row without fault')
            gc_partial=True;continue
        integer(row['thread'],1);integer(row['inside_probe'],0,1);integer(row['collected']);integer(row['uncollectable']);integer(row['active_span'],-1,ns-1)
        if row['active_span']>=0 and rows[row['active_span']]['thread']!=row['thread']:raise AssertionError('Foreign GC claims active sampled-thread span')
        gen=row['generation']
        if row['phase']==1:
            if open_gc[gen]!=-1:
                if clean:raise AssertionError('Duplicate GC generation start')
                gc_partial=True
            else:open_gc[gen]=i
        elif open_gc[gen]<0:
            if clean:raise AssertionError('GC stop lacks start')
            gc_partial=True
        else:
            start=gc[open_gc[gen]]
            if row['thread']!=start['thread'] or row['wall_before']<start['wall_after'] or row['thread_cpu']<start['thread_cpu'] or row['process_cpu']<start['process_cpu']:
                if clean:raise AssertionError('GC pair clock/thread regression')
                gc_partial=True
            else:gc_pairs.append({'start_row':open_gc[gen],'stop_row':i,'generation':gen,'thread':row['thread'],
                'wall_lower_bound_ns':row['wall_before']-start['wall_after'],
                'wall_upper_bound_ns':row['wall_after']-start['wall_before'],
                'thread_cpu_delta_ns':row['thread_cpu']-start['thread_cpu'],
                'process_cpu_delta_ns':row['process_cpu']-start['process_cpu']})
            open_gc[gen]=-1
    if not gc_partial and list(metadata['open_gc_generations'])!=open_gc:raise AssertionError('Open GC pair accounting differs')
    if type(metadata['callback_detached']) is not bool or type(metadata['instrumentation_complete']) is not bool:raise AssertionError('Literal completion flags')
    reconstructed=clean and depth==0 and ns==closed and all(v==-1 for v in metadata['open_gc_generations']) and metadata['callback_detached']
    if metadata['instrumentation_complete'] is not reconstructed:raise AssertionError('Probe completion differs')
    complete=bool(reconstructed and not hooks['failed'] and not report['timing_instrumentation']['errors'])
    if report['timing_instrumentation']['complete'] is not complete:raise AssertionError('Sidecar report completion differs')
    by_stage={r['stage']:r for r in stages};armed=by_stage.get('epoch_armed');exited=by_stage.get('plant_or_setup_exit');finished=by_stage.get('preservation_complete')
    if roots:
        if armed is None or exited is None or report['setup_finished_ns'] is None:raise AssertionError('Span execution lacks fixed epoch/stage receipts')
        for row in rows:
            if row['begin_complete']==1 and row['start_wall_before']<report['setup_finished_ns']:raise AssertionError('Tick probe before setup admission')
            if row['end_complete']==1 and row['end_wall_after']>exited['record_created_ns']:raise AssertionError('Span extends beyond saved exit receipt')
    if armed is not None:
        initial=armed['timing_probe']
        if initial['enabled'] is not True or initial['spans_started']!=0 or initial['spans_closed']!=0 or initial['gc_recorded']>ng or armed['gc_enabled_unchanged'] is not True:
            raise AssertionError('Epoch timing allocation/GC evidence differs')
    if exited is not None and exited['timing_probe'] is not None:
        closing=exited['timing_probe']
        for key in hooks.keys()-{'gc_recorded','probe_fault_code','failed'}:
            if closing[key]!=hooks[key]:raise AssertionError('Final span/hook counts differ from exit stage')
        if not 0<=closing['gc_recorded']<=ng:raise AssertionError('Final GC count lost exit prefix')
        if closing['probe_fault_code']!=hooks['probe_fault_code']:
            if closing['probe_fault_code']!=0 or hooks['probe_fault_code']==0 or (fault!='GC_REGISTRY' and not closing['gc_recorded']<=metadata['fault_record']<ng):
                raise AssertionError('Post-exit probe fault lacks later GC/registry evidence')
        if closing['failed'] and not hooks['failed']:raise AssertionError('Probe failure erased after exit')
    earliest=by_stage.get('native_restored',stages[0])['record_created_ns'] if stages else 0
    latest=finished['record_created_ns'] if finished else None
    for row in gc:
        if row['sample_complete']==1 and (row['wall_before']<earliest or (latest is not None and row['wall_after']>latest)):raise AssertionError('GC sample outside setup/preservation timeline')
    if complete and report['gc_enabled_before'] is not report['gc_enabled_after']:raise AssertionError('GC policy changed')
    gaps=counter_gap_proof(report,rows,owned)
    phase_timing={}
    for phase,name in enumerate(PHASES,1):
        selected=[duration(row) for row in rows if row['phase']==phase and row['begin_complete']==row['end_complete']==1]
        phase_timing[name]={'complete_spans':len(selected),'incomplete_spans':sum(row['phase']==phase for row in rows)-len(selected),
            'normal_scope_returns':sum(v['return_kind']==1 for v in selected),'raised_scope_returns':sum(v['return_kind']==2 for v in selected),
            'largest_complete_spans':sorted(selected,key=lambda v:v['wall_upper_bound_ns'],reverse=True)[:10]}
    return {'sidecar_evidence_integrity_passed':True,'instrumentation_complete':complete,
        'spans':ns,'gc_records':ng,'fault':fault,'hook_faults':hooks['hook_faults'],
        'pending_spans':ns-closed,'active_depth':depth,'scope_parentage_checked':True,
        'absolute_epoch_unchanged':True,'counter_handoff':gaps,
        'phase_timing':phase_timing,'complete_gc_pairs':gc_pairs,
        'foreign_thread_gc_overlap_rows':[i for i,r in enumerate(gc) if r['active_span']==-1 and r['inside_probe']==1],
        'limitations':['Nested/inclusive durations must not be summed as disjoint work.',
          'Process CPU includes other threads; wall/CPU differences alone do not prove scheduler or GC causation.',
          'inside_probe is process-shared; foreign-thread overlap does not prove sampled-thread interruption.',
          'Partial spans or counter handoffs receive no full timing/physical qualification.']}
