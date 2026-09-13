"""Concurrent publication stress test for the simulation clock's C ABI."""
import argparse,ctypes as ct,json,multiprocessing as mp,sys,time
from pathlib import Path
import numpy as np,mujoco
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'artifacts/teleop_resume_20260911'))
from run_causal_native_clock import Bridge,ptr


def main():
    p=argparse.ArgumentParser();p.add_argument('--library',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
    bridge=Bridge(a.library);assert bridge.lib.clock_snapshot_protocol_version()==1
    context=mp.get_context('fork')
    count=20000;start=context.Event();done=context.Event();result=context.Queue();base=np.arange(382)*.125
    target_base=np.arange(23)*.25;reads=[0,0,0];last=[-1,-1,-1]
    def writer():
        start.wait()
        try:
            for ident in range(count):
                values=base+ident;target=target_base+ident
                assert bridge.lib.clock_publish_observation(bridge.address,ident,ident%2,ident*17,ptr(values))==1
                for channel in (0,1):
                    assert bridge.lib.clock_publish_result(bridge.address,channel,ident,ident*19,ident*23,ptr(target))==1
        finally:done.set()
    def reader():
        start.wait();deadline=time.monotonic()+20
        while time.monotonic()<deadline:
            obs=bridge.observation()
            if obs is not None and obs[0]>=0:
                ident,fault,stamp,values=obs
                assert ident>=last[0] and fault==ident%2 and stamp==ident*17
                assert np.array_equal(values,base+ident);last[0]=ident;reads[0]+=1
            for channel in (0,1):
                ident,begin,finish=ct.c_int(),ct.c_int64(),ct.c_int64();target=np.empty(23)
                if bridge.lib.clock_read_result(bridge.address,channel,ct.byref(ident),ct.byref(begin),ct.byref(finish),ptr(target)) and ident.value>=0:
                    assert ident.value>=last[channel+1] and begin.value==ident.value*19 and finish.value>begin.value
                    assert np.array_equal(target,target_base+ident.value)
                    last[channel+1]=ident.value;reads[channel+1]+=1
            if done.is_set() and min(last)==count-1:
                result.put((reads,last));return
        raise RuntimeError('Snapshot reader did not receive each final publication')
    started=time.monotonic()
    processes=[context.Process(target=writer),context.Process(target=reader)]
    try:
        for process in processes:process.start()
        start.set()
        for process in processes:process.join(20)
        assert all(process.exitcode==0 for process in processes),[p.exitcode for p in processes]
        reads,last=result.get(timeout=2)
    finally:
        for process in processes:
            if process.is_alive():process.terminate();process.join(2)
        bridge.close(owner=True);result.close()
    report=dict(passed=True,publications_per_channel=count,channels=3,publication_rejections=0,
        coherent_reads=reads,last_received_ids=last,elapsed_seconds=time.monotonic()-started,
        separate_producer_consumer_processes=True,
        consumer_may_skip_overwritten_samples=True,real_time_deadlines_tested=False,simulation_ready=False)
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
