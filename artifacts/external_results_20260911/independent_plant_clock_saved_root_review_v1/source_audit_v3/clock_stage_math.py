"""Independent saved-only stage/deadline audit; never imports producer helpers."""
import ast
import hashlib

NS=1_000_000_000
BUDGETS={'setup_ns':240*NS,'plant_ns':120*NS,'preservation_ns':180*NS,
         'outer_timeout_seconds':555,'outer_kill_grace_seconds':5}
SOURCE_SHA={'original_runner':'ed431d371197f9becaae6858676b04ba169937d1e6a8f7cb4b448cfb36c465b1',
    'clock_runner':'24138c0c8bc4030549e0e9700eb2b8653785f86f66ed02356830a543aa80d7f6',
    'clock_watchdog':'a6613da97b1163031b6b7641af62f4c9ec63662bc3c6ff25deda3ee09b0b8089'}
COUNTERS=('attempted','returned','captured','verified')
NATIVE=('attempted','returned','capture_attempts','captured','verification_attempts','verified')
API=('step_attempted','step_returned','serialization_attempted','serialization_returned')


def integer(value,name):
    if type(value) is not int or value<0:raise AssertionError('Nonnegative integer '+name)
    return value


def source_contract(paths):
    texts={}
    for name,expected in SOURCE_SHA.items():
        raw=paths[name].read_bytes()
        if hashlib.sha256(raw).hexdigest()!=expected:raise AssertionError('Reviewed clock source differs: '+name)
        texts[name]=raw.decode()
    def loops(text):return [ast.dump(n,include_attributes=False) for n in ast.walk(ast.parse(text)) if isinstance(n,ast.While)]
    if loops(texts['clock_runner'])!=loops(texts['original_runner']):raise AssertionError('Original stepping loop changed')
    if len(loops(texts['clock_runner']))!=1:raise AssertionError('Expected one unchanged plant loop')
    return {'literal_sources_exact':True,'original_hot_loop_AST_exact':True,'producer_imports':0}


def request_contract(request,arguments):
    expected={'epoch_lead_ns':200000000,'debt_abort_steps':100,'elapsed_abort_ns':60*NS,
        'native_step_budget':18190,'serialization_budget':4,'outer_process_timeout_seconds':555,
        'requested_controls':1819,'main_controls':1569,'hold_controls':250}
    if request.get('watchdog_budgets')!=BUDGETS or any(type(v) is not int for v in request['watchdog_budgets'].values()):
        raise AssertionError('Stage budget differs from reviewed source')
    for key,value in expected.items():
        if type(request.get(key)) is not int or request[key]!=value:raise AssertionError('Original scope/deadline differs: '+key)
    if request.get('epoch_rebase_allowed') is not False:raise AssertionError('Epoch rebasing allowed')
    if arguments.count('timeout')!=1:raise AssertionError('Single outer timeout required')
    i=arguments.index('timeout')
    if arguments[i:i+4]!=['timeout','--signal=TERM','--kill-after=5s','555s']:
        raise AssertionError('Outer timeout/grace differs')


def watchdog(value,record_time,epoch):
    if value['budgets']!=BUDGETS or value['native_signal_delivery_may_be_delayed'] is not True:raise AssertionError('Watchdog contract changed')
    transitions=value['transitions'];names=[t['stage'] for t in transitions]
    if names not in (['setup'],['setup','plant'],['setup','preservation'],['setup','plant','preservation']):
        raise AssertionError('Duplicate, reordered or rearmed stage')
    deadlines={};last=-1
    for t in transitions:
        entered=integer(t['entered_ns'],'stage entry');deadline=integer(t['deadline_ns'],'deadline')
        if entered<last or entered>record_time:raise AssertionError('Stage transition clock order')
        if t['stage']=='setup':expected=entered+240*NS
        elif t['stage']=='plant':
            if type(epoch) is not int or not entered<epoch:raise AssertionError('Plant armed after fixed epoch')
            expected=epoch+120*NS
            if entered>=deadlines['setup']:raise AssertionError('Plant armed after setup timeout')
        else:expected=entered+180*NS
        if deadline!=expected:raise AssertionError('Stage deadline rebased')
        deadlines[t['stage']]=deadline;last=entered
    if value['phase']!=names[-1] or value['deadline_ns']!=transitions[-1]['deadline_ns']:raise AssertionError('Active stage differs')
    for timeout in value['timeouts']:
        observed=integer(timeout['observed_ns'],'timeout observation')
        if timeout['stage'] not in deadlines or timeout['deadline_ns']!=deadlines[timeout['stage']] or not deadlines[timeout['stage']]<=observed<=record_time:
            raise AssertionError('Timeout observation has no matching fixed deadline')
    return transitions


def counter_block(value,report,serialization=None):
    if value['extra_capture_or_native_read'] is not False:raise AssertionError('Stage reporter performed native read')
    core=None if report['session'] is None else report['session']['foundation']
    native=None if report['session'] is None else report['session']['stepper'];api=report['api_counters']
    for role,expected,keys in [('foundation',core,COUNTERS),('adapter',native,NATIVE),('api',api,API)]:
        block=value[role]
        if expected is None:
            if role!='adapter' and block is not None:raise AssertionError('Stage counter has no actual runtime object')
            continue
        if block is None:raise AssertionError('Missing stage counter block')
        for key in keys:
            actual=integer(block[key],role+'/'+key)
            wanted=serialization if role=='api' and key.startswith('serialization_') and serialization is not None else expected[key]
            if wanted=='prefix':
                attempts=integer(block['serialization_attempted'],'entry serialization attempts')
                if attempts>api['serialization_attempted']:raise AssertionError('Entry serialization exceeds final count')
                wanted=attempts if key=='serialization_attempted' else sum(r['native_returned'] is True for r in api['serialization_records'][:attempts])
            if actual!=wanted:raise AssertionError('Stage counter differs from final actual ledger: '+role+'/'+key)
    uncertain=bool(api is not None and api['step_attempted']!=api['step_returned'])
    if value['native_return_may_be_uncertain'] is not uncertain:raise AssertionError('Native return uncertainty erased')


def stages(records,report,raw,request_sha,report_sha,manifest_sha,arrays=None):
    if not records or records[0]['stage']!='setup_started':raise AssertionError('Missing durable setup start')
    names=[r['stage'] for r in records]
    allowed=['setup_started','native_setup_started','native_restored','epoch_armed','plant_or_setup_exit','preservation_complete','incomplete_process_exit']
    if len(set(names))!=len(names) or any(n not in allowed for n in names) or [allowed.index(n) for n in names]!=sorted(allowed.index(n) for n in names):
        raise AssertionError('Stage records skipped backwards, duplicated or unknown')
    pid=integer(records[0]['pid'],'native PID')
    if pid==0:raise AssertionError('Invalid native PID')
    if report.get('runtime',{}).get('pid',pid)!=pid:raise AssertionError('Stage/native process mismatch')
    epoch=report['epoch_ns'];previous_time=-1;previous_transitions=[];previous_timeouts=[]
    for record in records:
        now=integer(record['record_created_ns'],'record clock')
        if record['request_sha256']!=request_sha or record['pid']!=pid or now<previous_time:raise AssertionError('Stage request/PID/clock differs')
        transitions=watchdog(record['watchdog'],now,epoch)
        expected_phase={'setup_started':'setup','native_setup_started':'setup','native_restored':'setup',
            'epoch_armed':'plant','plant_or_setup_exit':'preservation','preservation_complete':'preservation'}.get(record['stage'])
        if expected_phase is not None and record['watchdog']['phase']!=expected_phase:raise AssertionError('Stage receipt phase differs')
        if transitions[:len(previous_transitions)]!=previous_transitions:raise AssertionError('Earlier absolute stage transition changed')
        if record['watchdog']['timeouts'][:len(previous_timeouts)]!=previous_timeouts:raise AssertionError('Earlier timeout erased')
        previous_time=now;previous_transitions=transitions;previous_timeouts=record['watchdog']['timeouts']
    by_name=dict(zip(names,records));armed=by_name.get('epoch_armed');exited=by_name.get('plant_or_setup_exit');complete=by_name.get('preservation_complete')
    native_rows=0 if arrays is None else len(arrays['step_index'])
    if 'native_restored' in by_name:
        restored=by_name['native_restored']['counters']
        if restored['foundation'] is not None or restored['extra_capture_or_native_read'] is not False or restored['native_return_may_be_uncertain'] is not False:
            raise AssertionError('Unexpected native restoration bookkeeping')
        if restored['adapter'] is None or any(type(restored['adapter'][key]) is not int or restored['adapter'][key]!=0 for key in NATIVE):raise AssertionError('Native steps during restoration')
        if restored['api']!={'step_attempted':0,'step_returned':0,'serialization_attempted':2,'serialization_returned':2}:raise AssertionError('Initial model serialization scope differs')
    if armed is not None:
        chosen=integer(armed['epoch_chosen_ns'],'chosen epoch');allocation=integer(armed['allocation_finished_ns'],'allocation finish')
        if armed['epoch_ns']!=epoch or epoch-chosen!=200000000 or not chosen<=allocation<=armed['record_created_ns']:
            raise AssertionError('Single chosen epoch/allocation differs')
        if report['epoch_chosen_ns']!=chosen or report['epoch_rebased'] is not False:raise AssertionError('Final epoch differs')
        finished=report['setup_finished_ns']
        if finished is None:
            if native_rows or report['first_error'] is None:raise AssertionError('Missing setup gate with native execution or no failure')
        else:
            integer(finished,'setup finish')
            if finished<armed['record_created_ns'] or (native_rows and not finished<epoch):raise AssertionError('Native steps after slow epoch fsync/setup')
        for role in ('foundation','adapter'):
            block=armed['counters'][role]
            if block is None or any(block[key]!=0 or type(block[key]) is not int for key in COUNTERS):raise AssertionError('Native steps before epoch admission')
        if armed['counters']['api']!={'step_attempted':0,'step_returned':0,'serialization_attempted':2,'serialization_returned':2}:
            raise AssertionError('Epoch entry native/serialization counts differ')
    elif native_rows:raise AssertionError('Native trace lacks durable epoch')
    if exited is not None:
        if exited['first_error']!=report['first_error'] or exited['epoch_ns']!=epoch or exited['epoch_chosen_ns']!=report['epoch_chosen_ns'] or exited['setup_finished_ns']!=report['setup_finished_ns']:
            raise AssertionError('Loop-exit failure/epoch evidence changed')
        if exited['watchdog']['phase']!='preservation':raise AssertionError('Cleanup began without preservation arm')
        counter_block(exited['counters'],report,serialization=2 if armed is not None else 'prefix')
        if arrays is not None:
            captured=exited['counters']['foundation']['captured'];verified=exited['counters']['foundation']['verified']
            if not native_rows<=captured<=native_rows+1 or not int(arrays['step_verified'].sum())<=verified<=int(arrays['step_verified'].sum())+1:
                raise AssertionError('Stage count exceeds saved/reserved native coverage')
    elif native_rows:raise AssertionError('Native trace lacks durable exit counters')
    if complete is not None:
        if exited is None or complete['watchdog']['phase']!='preservation':raise AssertionError('Preservation completed without loop/setup exit')
        if complete['report_sha256']!=report_sha or complete['output_manifest_sha256']!=manifest_sha:raise AssertionError('Completed stage output subject differs')
        counter_block(complete['counters'],report)
        if complete['watchdog']!=report['stage_watchdog']:raise AssertionError('Final report watchdog history differs')
    known_zero=raw.get('known') is True and type(raw.get('raw_python_exit_code')) is int and raw['raw_python_exit_code']==0 and raw.get('raw_error') is None
    deadline_clear=bool(complete and complete['record_created_ns']<complete['watchdog']['deadline_ns'] and not previous_timeouts)
    core=None if report['session'] is None else report['session']['foundation']
    native_complete=bool(core and all(type(core[k]) is int and core[k]==18190 for k in COUNTERS))
    full_stage=bool(names==allowed[:-1] and armed and exited and complete and deadline_clear and known_zero and native_complete)
    return {'stage_evidence_integrity_passed':True,'stage_watchdog_qualified':full_stage,
        'stage_names':names,'same_request_and_pid':True,'fixed_deadlines_reconstructed':True,
        'epoch_record_present':armed is not None,'loop_exit_counters_present':exited is not None,
        'preservation_complete':complete is not None,'known_raw_zero':known_zero,'timeouts':previous_timeouts,
        'counter_blocks_equal_final_native_ledger':exited is not None and complete is not None,
        'limitations':['Record timestamps precede fsync completion; raw-zero and pinned finish() source jointly establish the final deadline check.',
            'A hard timeout or incomplete final record never receives stage qualification. Native integrity still requires the independent saved-array math.']}
