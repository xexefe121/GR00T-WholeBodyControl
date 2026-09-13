"""Run the unchanged reviewed suite once; observe child IDs and module closure."""
from pathlib import Path
import hashlib
import importlib.abc
import json
import multiprocessing as mp
import multiprocessing.process
import os
import platform
import sys
import time
import traceback
import unittest

BASE=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def linux(path):
    path=path.replace('\\','/')
    return Path('/mnt/'+path[0].lower()+path[2:]) if len(path)>2 and path[1]==':' else Path(path)
def write(path,value):
    with Path(path).open('x') as f:f.write(json.dumps(value,indent=2)+'\n')

class NoTaskEngines(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'mujoco','onnxruntime','torch','mjbatch'}:
            raise RuntimeError('Native/model imports forbidden in selected synthetic test: '+fullname)

def main():
    request=json.loads((BASE/'request.json').read_text())
    pins=request['input_sha256'];initial={p:sha(linux(p)) for p in pins}
    assert initial==pins
    assert list(sys.version_info[:2])==request['expected_python_major_minor']
    sys.meta_path.insert(0,NoTaskEngines())
    import numpy as np
    assert np.__version__==request['expected_numpy']
    sys.path.insert(0,str(BASE/'source_snapshot_v1'))
    import test_transport
    import clock_core,history,mailbox,shared_mailbox
    for module in [test_transport,clock_core,history,mailbox,shared_mailbox]:
        assert Path(module.__file__).resolve().parent==BASE/'source_snapshot_v1'
    suite=unittest.defaultTestLoader.loadTestsFromModule(test_transport)
    assert suite.countTestCases()==11
    spawned=[];original_start=multiprocessing.process.BaseProcess.start
    def observed_start(child,*args,**kwargs):
        value=original_start(child,*args,**kwargs)
        spawned.append(dict(pid=child.pid,name=child.name))
        return value
    multiprocessing.process.BaseProcess.start=observed_start
    started=time.monotonic();result=None;failure=None
    try:
        with (BASE/'tests.log').open('x') as log:
            result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    except BaseException:
        failure=traceback.format_exc()
    finally:
        multiprocessing.process.BaseProcess.start=original_start
    alive=mp.active_children()
    cleanup=[]
    for child in alive:
        child.terminate();child.join(5);cleanup.append(dict(pid=child.pid,exitcode=child.exitcode));child.close()
    runtime_files=[Path(sys.executable).resolve(),Path(np.__file__),Path(np.core._multiarray_umath.__file__),Path(mp.__file__)]
    final={p:sha(linux(p)) for p in pins}
    report=dict(passed=result is not None and result.wasSuccessful() and result.testsRun==11 and not result.skipped and failure is None and not alive and final==pins,
        tests_run=None if result is None else result.testsRun,failures=[] if result is None else [(str(t),s) for t,s in result.failures],
        errors=[] if result is None else [(str(t),s) for t,s in result.errors],skipped=[] if result is None else [(str(t),s) for t,s in result.skipped],
        exception=failure,spawned_processes=spawned,active_children_after_suite=[dict(pid=c['pid']) for c in cleanup],cleanup=cleanup,
        resource_tracker_pid=getattr(mp.resource_tracker._resource_tracker,'_pid',None),
        pid=os.getpid(),elapsed_seconds=time.monotonic()-started,request_sha256=sha(BASE/'request.json'),
        input_sha256=pins,all_initial_and_final_pins_exact=initial==pins==final,
        python=sys.version,executable=sys.executable,numpy=np.__version__,platform=platform.platform(),
        runtime_sha256={str(p):sha(p) for p in runtime_files},test_log_sha256=sha(BASE/'tests.log'),
        imported_task_engines=[m for m in sys.modules if m.split('.')[0] in {'mujoco','onnxruntime','torch','mjbatch'}],
        native_steps=0,model_calls=0,optimizer_updates=0,plant_ticks=0,real_time_qualified=False)
    assert not report['imported_task_engines']
    write(BASE/'test_report.json',report)
    return 0 if report['passed'] else 1

if __name__=='__main__':sys.exit(main())
