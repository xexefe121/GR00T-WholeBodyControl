"""Synthetic spawn-process transport/admission tests. No plant ticks or native APIs."""
import copy
import hashlib
import json
import multiprocessing as mp
import os
import time
import unittest
import numpy as np
from shared_mailbox import SharedMailbox,epoch_token,HEADER,FULL,MAX_U64
from clock_core import Binding,Command,Job,Result,PlantFoundation,FixedLedger,encode

TOKEN=hashlib.sha256(b'synthetic run/model/reference/epoch3').digest()

def publish_worker(mailbox,key,payload,pipe):
    pipe.send(mailbox.try_publish(key,payload));pipe.close()

def take_worker(mailbox,pipe):
    pipe.send(mailbox.poll_once());pipe.close()

class SlowCopy(SharedMailbox):
    def _copy_payload(self,index,payload):
        view=self._view();start=self._offset(index)+HEADER.size;half=len(payload)//2
        try:
            view[start:start+half]=payload[:half];self.ready.set()
            if not self.release_copy.wait(10):raise RuntimeError('test watchdog')
            view[start+half:start+len(payload)]=payload[half:]
        finally:view.release()

def slow_worker(mailbox,key,payload,ready,release,pipe):
    slow=SlowCopy(mailbox._layout,TOKEN);slow.ready=ready;slow.release_copy=release
    pipe.send(slow.try_publish(key,payload));pipe.close()

class InterruptedCopy(SharedMailbox):
    def _copy_payload(self,index,payload):
        view=self._view();start=self._offset(index)+HEADER.size
        view[start:start+1]=payload[:1];view.release();self.ready.set()
        os._exit(73)  # Abrupt process death while owning the shared semaphore.

def interrupted_worker(mailbox,ready):
    broken=InterruptedCopy(mailbox._layout,TOKEN);broken.ready=ready;broken.try_publish(0,b'partial packet')

class ThrowingCopy(SharedMailbox):
    def _copy_payload(self,index,payload):
        super()._copy_payload(index,payload[:1]);raise RuntimeError('injected copy exception')

def exception_worker(mailbox,pipe):
    broken=ThrowingCopy(mailbox._layout,TOKEN)
    try:broken.try_publish(0,b'not fully committed')
    except RuntimeError:pipe.send('EXCEPTION')
    finally:pipe.close()

def contention_worker(mailbox,ready,release):
    lock=mailbox._layout.locks[0];lock.acquire()
    try:ready.set();release.wait(10)
    finally:lock.release()

def stream_worker(mailbox,count,pipe):
    statuses=[]
    for key in range(count):
        payload=key.to_bytes(8,'little')+bytes([key%251])*(1000+key*17)
        deadline=time.monotonic()+10
        while True:  # Harness-only retries; transport itself never retries.
            status=mailbox.try_publish(key,payload)
            if status=='PUBLISHED':break
            if status not in ('FULL','BUSY') or time.monotonic()>deadline:raise RuntimeError(status)
            time.sleep(.001)
        statuses.append(status)
    pipe.send(statuses);pipe.close()

class Now:
    def __init__(self,ns=115_000_000):self.ns=ns
    def now_ns(self):return self.ns

def admission_harness(mailbox):
    # Only existing pure result-admission logic is exercised; no tick/stepper/controller.
    binding=Binding('synthetic','0'*64,'1'*64,3)
    job=Job(binding,0,0,0,1,'window1',b'immutable synthetic snapshot','2'*64,'3'*64,105_000_000)
    plant=object.__new__(PlantFoundation)
    plant.binding=binding;plant.results=mailbox;plant.clock=Now();plant._last_clock_ns=None;plant.clock_fault=None;plant.failure=None
    plant._epoch_ns=100_000_000;plant.input_fault=None;plant.returned=0;plant.issued={1:job};plant.published_jobs={1};plant.sealed={};plant.command_ids={'initial'}
    plant.limits=np.tile(np.array([-2.,2.]),(23,1));plant.events=FixedLedger(1000)
    command=Command.make('synthetic proposed','synthetic',np.zeros(23,np.float64),np.zeros(23,np.float32))
    return plant,job,command

class TransportTests(unittest.TestCase):
    def setUp(self):self.ctx=mp.get_context('spawn');self.children=[]
    def tearDown(self):
        for child in self.children:
            child.join(2)
            if child.is_alive():child.terminate();child.join(5)
            child.close()
    def mailbox(self,slots=2,capacity=32768):return SharedMailbox.create(self.ctx,slots,capacity,TOKEN)
    def start(self,target,*args):
        child=self.ctx.Process(target=target,args=args);child.start();self.children.append(child);return child
    def recv(self,pipe):
        self.assertTrue(pipe.poll(10),'spawn harness watchdog');return pipe.recv()
    def publish_spawn(self,mailbox,key,payload):
        receive,send=self.ctx.Pipe(False);child=self.start(publish_worker,mailbox,key,payload,send);send.close();result=self.recv(receive);receive.close();child.join(10);self.assertEqual(child.exitcode,0);return result
    def test_round_trip_two_spawned_peers(self):
        mailbox=self.mailbox();payload=b'complete immutable packet'*100
        self.assertEqual(self.publish_spawn(mailbox,3,payload),'PUBLISHED')
        receive,send=self.ctx.Pipe(False);child=self.start(take_worker,mailbox,send);send.close();items=self.recv(receive);receive.close();child.join(10);self.assertEqual(child.exitcode,0)
        taken=[p for _,status,p in items if status=='TAKEN'];self.assertEqual(len(taken),1);self.assertEqual(taken[0].payload,payload);self.assertEqual(type(taken[0].payload),bytes);self.assertEqual(taken[0].version,1)
        self.assertEqual(mailbox.poll_once()[1][1],'EMPTY')
    def test_full_slot_never_overwrites_newer_or_older(self):
        mailbox=self.mailbox(1)
        self.assertEqual(self.publish_spawn(mailbox,4,b'first'),'PUBLISHED')
        self.assertEqual(self.publish_spawn(mailbox,7,b'newer'),'FULL');self.assertEqual(self.publish_spawn(mailbox,1,b'older'),'FULL')
        first=mailbox.poll_once()[0][2];self.assertEqual((first.key,first.payload),(4,b'first'))
        self.assertEqual(self.publish_spawn(mailbox,7,b'newer'),'PUBLISHED');second=mailbox.poll_once()[0][2]
        self.assertEqual(second.version,2);self.assertEqual(first.payload,b'first')
    def test_partial_write_not_visible(self):
        mailbox=self.mailbox(1,524288);ready=self.ctx.Event();release=self.ctx.Event();receive,send=self.ctx.Pipe(False)
        payload=bytes(range(256))*2048;child=self.start(slow_worker,mailbox,0,payload,ready,release,send);send.close();self.assertTrue(ready.wait(10))
        for _ in range(50):self.assertEqual(mailbox.poll_once(),((0,'BUSY',None),))
        release.set();self.assertEqual(self.recv(receive),'PUBLISHED');receive.close();child.join(10)
        self.assertEqual(mailbox.poll_once()[0][2].payload,payload)
    def test_process_death_stays_nonblocking_busy(self):
        mailbox=self.mailbox(1);ready=self.ctx.Event();child=self.start(interrupted_worker,mailbox,ready);self.assertTrue(ready.wait(10));child.join(10);self.assertEqual(child.exitcode,73)
        started=time.perf_counter()
        for _ in range(100):self.assertEqual(mailbox.try_publish(1,b'new'),'BUSY');self.assertEqual(mailbox.poll_once(),((0,'BUSY',None),))
        self.assertLess(time.perf_counter()-started,5,'generous harness watchdog, not a real-time qualification')
    def test_copy_exception_poisoned_without_partial_return(self):
        mailbox=self.mailbox(1);receive,send=self.ctx.Pipe(False);child=self.start(exception_worker,mailbox,send);send.close();self.assertEqual(self.recv(receive),'EXCEPTION');receive.close();child.join(10)
        self.assertEqual(mailbox.poll_once(),((0,'UNCOMMITTED',None),));self.assertEqual(mailbox.try_publish(2,b'new'),'UNCOMMITTED')
    def test_lock_contention_does_not_wait_for_worker(self):
        mailbox=self.mailbox(1);ready=self.ctx.Event();release=self.ctx.Event();child=self.start(contention_worker,mailbox,ready,release);self.assertTrue(ready.wait(10))
        started=time.perf_counter()
        for _ in range(100):self.assertEqual(mailbox.try_publish(0,b'x'),'BUSY');self.assertEqual(mailbox.poll_once(),((0,'BUSY',None),))
        self.assertLess(time.perf_counter()-started,5);self.assertFalse(release.is_set());release.set();child.join(10);self.assertEqual(child.exitcode,0)
    def test_wrong_endpoint_epoch_cannot_publish_or_consume(self):
        mailbox=self.mailbox(1);wrong=mailbox.endpoint(hashlib.sha256(b'other epoch').digest())
        self.assertEqual(self.publish_spawn(wrong,0,b'wrong'),'WRONG_EPOCH');self.assertEqual(mailbox.poll_once()[0][1],'EMPTY')
        self.assertEqual(self.publish_spawn(mailbox,0,b'right'),'PUBLISHED');self.assertEqual(wrong.poll_once(),((0,'WRONG_EPOCH',None),));self.assertEqual(mailbox.poll_once()[0][2].payload,b'right')
    def test_fixed_stream_whole_packets_and_immutable_copies(self):
        mailbox=self.mailbox(3,8192);receive,send=self.ctx.Pipe(False);child=self.start(stream_worker,mailbox,48,send);send.close();seen={};deadline=time.monotonic()+20
        while len(seen)<48 and time.monotonic()<deadline:
            for _,status,p in mailbox.poll_once():
                if status=='TAKEN':
                    self.assertNotIn(p.key,seen);expected=p.key.to_bytes(8,'little')+bytes([p.key%251])*(1000+p.key*17);self.assertEqual(p.payload,expected);seen[p.key]=p
                else:self.assertIn(status,('EMPTY','BUSY'))
            time.sleep(.001)
        self.assertEqual(len(seen),48);self.assertEqual(self.recv(receive),['PUBLISHED']*48);receive.close();child.join(10);self.assertEqual(child.exitcode,0)
        self.assertEqual(seen[0].payload,b'\0'*1008)
    def test_original_binding_sequence_duplicate_and_late_rejections(self):
        for mutation,reason in [('epoch','WRONG_BINDING_OR_EPOCH'),('sequence','JOB_OR_INPUT_MISMATCH'),('late','LATE'),('duplicate','DUPLICATE')]:
            mailbox=self.mailbox();plant,job,command=admission_harness(mailbox);result=Result.for_job(job,command,110_000_000)
            if mutation=='epoch':result.identity['binding']['epoch']=4
            if mutation=='sequence':result.identity['sequence']=99
            if mutation=='late':plant.clock.ns=120_000_000
            if mutation=='duplicate':
                self.assertEqual(self.publish_spawn(mailbox,1,result.to_bytes()),'PUBLISHED');plant.poll_results();self.assertIn(1,plant.sealed)
            self.assertEqual(self.publish_spawn(mailbox,1,result.to_bytes()),'PUBLISHED');plant.poll_results()
            events=[json.loads(v) for v in plant.events.records()];self.assertEqual(events[-1]['rejection'],reason)
    def test_capacity_bytes_only_corruption_and_close(self):
        mailbox=self.mailbox(1,8)
        for value in (bytearray(b'abc'),memoryview(b'abc'),b'123456789'):self.assertEqual(mailbox.try_publish(0,value),'OVERSIZE_OR_MUTABLE')
        for key in (-1,True,1.5,MAX_U64+1):self.assertEqual(mailbox.try_publish(key,b'x'),'BAD_KEY')
        self.assertEqual(mailbox.try_publish(0,b'123'),'PUBLISHED')
        with mailbox._layout.locks[0]:
            view=mailbox._view();view[HEADER.size]=99;view.release()
        self.assertEqual(mailbox.poll_once(),((0,'CORRUPT',None),));mailbox.close();self.assertEqual(mailbox.try_publish(1,b'x'),'CLOSED');self.assertEqual(mailbox.poll_once(),((0,'CLOSED',None),))
    def test_constructor_bounds_and_binding_token(self):
        for slots,cap in [(0,1),(True,1),(1,0),(1025,1),(1,1048577),(1024,1048576)]:
            with self.assertRaises(ValueError):self.mailbox(slots,cap)
        self.assertEqual(epoch_token(b'canonical'),hashlib.sha256(b'canonical').digest())
        for bad in (b'',bytearray(b'a'),b'x'*4097):
            with self.assertRaises(ValueError):epoch_token(bad)

if __name__=='__main__':unittest.main()
