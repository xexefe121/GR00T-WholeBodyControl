"""Keep the inactive standing policy warm without changing control outputs."""
from pathlib import Path
BASE=Path(__file__).resolve().parent
s=(BASE/'run_paced_controller.py').read_text()
s=s.replace('import numpy as np','import numpy as np\nimport threading',1)
s=s.replace('    base.pending=None;base.current_mode=None;base.context={}', '''    # CountedSession captured this dry startup call's immutable inference inputs.
    warm_feeds={name:{key[len(name+'_input_'):]:np.asarray(value).copy()
        for key,value in base.context.items() if key.startswith(name+'_input_')}
        for name in ('backward','actor')}
    assert all(warm_feeds.values())
    warm_stop=threading.Event();warm_done=threading.Event();warm_thread=None
    warm_log={'calls_attempted':0,'calls_returned':0,'cycles_ms':[],'error':None}
    def keep_standing_policy_warm():
        try:
            while not warm_stop.is_set():
                tick=time.perf_counter()
                for name in ('backward','actor'):
                    warm_log['calls_attempted']+=1
                    base.seed.sessions[name].inner.run(None,warm_feeds[name])
                    warm_log['calls_returned']+=1
                warm_log['cycles_ms'].append((time.perf_counter()-tick)*1000)
                warm_stop.wait(.5)
        except BaseException as error:warm_log['error']=repr(error)
        finally:warm_done.set()
    base.pending=None;base.current_mode=None;base.context={}''')
s=s.replace('        nonlocal expected','        nonlocal expected,warm_thread')
s=s.replace('            tick=time.perf_counter()\n            arrays[\'start_lateness_ms\']', '''            tick=time.perf_counter()
            if control==300:
                warm_thread=threading.Thread(target=keep_standing_policy_warm,name='standing-policy-keepalive')
                warm_thread.start()
            if control==1200:warm_stop.set()
            if control==1269 and (not warm_done.is_set() or warm_log['error'] is not None):
                failure=dict(kind='standing_policy_warmup_not_ready',control=control,error=warm_log['error']);break
            arrays['start_lateness_ms']''')
s=s.replace("        state_fields={'physics_qpos'", "        warm_stop.set()\n        if warm_thread is not None:warm_thread.join()\n        state_fields={'physics_qpos'")
s=s.replace("            startup_uncommitted_policy_warmup_ms=warmup_ms,startup_policy_calls=2,", "            startup_uncommitted_policy_warmup_ms=warmup_ms,startup_policy_calls=2,\n            background_noncontrol_inference=warm_log,background_calls_affect_control_output=False,")
out=BASE/'run_paced_hot_controller.py';assert not out.exists();out.write_text(s,encoding='utf-8')
launcher=(BASE/'run_paced_trial_once.py').read_text().replace('paced_process_v1','paced_hot_process_v1').replace("BASE/'paced_v1'","BASE/'paced_hot_v1'").replace('run_paced_controller.py','run_paced_hot_controller.py')
out=BASE/'run_paced_hot_trial_once.py';assert not out.exists();out.write_text(launcher,encoding='utf-8')
print('Prepared one 50 Hz full-motion + 30 s hold trial with noncontrol standing-policy keepalive.')
