"""Local ZMQ startup and captured-packet replay; no headset or robot connection."""
import copy, json, queue, sys, threading, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np, mujoco
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'artifacts/teleop_resume_20260911'))
import run_causal_native_clock as runtime
import zmq
from gear_sonic.utils.g1_true23_pico_body_adapter import Native23PicoBodyAdapter


def main():
    cycle=Path('/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/onboard_factory_firmware_v1/controller_state_ab_v1')
    out=cycle/'packet_transport_check_v2';out.mkdir(exist_ok=False)
    live=out/'live';live.mkdir();replay=out/'replay';replay.mkdir()
    fixture=json.loads((cycle/'pico_body_fk_check/wire_fixture.json').read_text())
    context=zmq.Context();publisher=context.socket(zmq.XPUB)
    port=publisher.bind_to_random_port('tcp://127.0.0.1');url=f'tcp://127.0.0.1:{port}'
    packets=queue.Queue();start=threading.Event();stop=threading.Event()
    thread=threading.Thread(target=runtime.live_producer,args=(packets,start,stop,url,live))
    thread.start()
    try:
        assert not publisher.poll(200,zmq.POLLIN),'Subscribed before controller warmup finished'
        epoch=time.monotonic_ns();start.set()
        assert publisher.poll(3000,zmq.POLLIN),'Missing local subscription'
        assert publisher.recv()==b'\x01'
        origin=time.monotonic_ns();sent=[]
        for index in range(3):
            stamp=origin+index*20_000_000
            while time.monotonic_ns()<stamp:time.sleep(.001)
            value=copy.deepcopy(fixture);body=value['native23_body_pose']
            value['control_monotonic_ns']=stamp;value['control_source_frame_index']=index
            body.update(reference_monotonic_ns=stamp,capture_monotonic_ns=stamp,source_frame_index=index)
            sent.append(value);publisher.send_json(value)
        received=[packets.get(timeout=3) for _ in sent]
        assert [item['payload'] for item in received]==sent
    finally:
        stop.set();start.set();thread.join(3);publisher.close(linger=0);context.term()
    assert not thread.is_alive() and not (live/'producer_error.txt').exists()
    captured=[json.loads(line) for line in (live/'live_packets.jsonl').read_text().splitlines()]
    assert captured==received
    np.savez(live/'trace.npz',epoch_ns=np.asarray(epoch))
    replay_epoch=SimpleNamespace(value=time.monotonic_ns()+20_000_000)
    packets=queue.Queue();start=threading.Event();stop=threading.Event();start.set()
    runtime.pico_replay_producer(packets,replay_epoch,start,stop,live/'live_packets.jsonl',replay)
    replayed=[packets.get(timeout=1) for _ in sent]
    assert [item['payload'] for item in replayed]==sent
    assert not (replay/'producer_error.txt').exists()
    _,contract,*_=runtime.load_case('walk002')
    original_adapter=Native23PicoBodyAdapter(contract);replayed_adapter=Native23PicoBodyAdapter(contract)
    for original,item in zip(received,replayed):
        a=original_adapter.adapt(original['payload'],received_monotonic_ns=original['received_monotonic_ns'])
        # A replay arrival is explicitly rebased, while payload timestamps stay original.
        rebased=original['received_monotonic_ns']-item['source_clock_offset_ns']
        b=replayed_adapter.adapt(item['payload'],received_monotonic_ns=rebased+item['source_clock_offset_ns'])
        assert a.source==b.source and a.packet.sequence==b.packet.sequence
        for key in a.packet.fields:np.testing.assert_array_equal(a.packet.fields[key],b.packet.fields[key])
    report=dict(passed=True,packets=len(sent),subscription_after_controller_ready=True,
        captured_payloads_unchanged=True,replayed_payloads_unchanged=True,explicit_clock_rebasing=True,
        adapter_fields_identical=True,headset_connected=False,robot_connected=False,live_qualified=False)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
