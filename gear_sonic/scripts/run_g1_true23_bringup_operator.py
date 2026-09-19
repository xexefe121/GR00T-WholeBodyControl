"""Independent operator heartbeat and stage-control client for the native G1 ladder.

This process has no policy, DDS, Unitree SDK, model, or LowCmd dependency.  It
connects only to the native loop's TCP control endpoint.  Keeping it separate
means that a sick or stopped BFM policy cannot prevent a supervised software
abort; if this client itself stops, the native loop detects the missing
heartbeat and aborts locally within one second.
"""

from __future__ import annotations

import argparse
import queue
import struct
import sys
import threading
import time

import zmq


CONTROL_MAGIC = 0x34434742  # BGC4, matching g1_true23_bfm_lowcmd_loop.cpp
# The native wire structs are inside #pragma pack(1), so ControlWire is 28
# bytes with no trailing padding, as asserted natively.
CONTROL_WIRE = struct.Struct("<IIQQI")
HEARTBEAT, ADVANCE, MANUAL_ABORT = range(3)


def payload(sequence: int, operation: int, sent_monotonic_ns: int = 0) -> bytes:
    return CONTROL_WIRE.pack(CONTROL_MAGIC, 2, sequence, sent_monotonic_ns, operation)


def run(args: argparse.Namespace) -> None:
    if not args.endpoint.startswith("tcp://"):
        raise ValueError("--endpoint must be a TCP endpoint")
    actions: queue.SimpleQueue[int] = queue.SimpleQueue()
    stop = threading.Event()
    abort_sent = threading.Event()

    def sender() -> None:
        context = zmq.Context.instance()
        socket = context.socket(zmq.PUSH)
        # A PUSH socket with no connected peer refuses to send.  Blocking with a
        # bounded timeout makes that refusal visible and retryable instead of
        # silently dropping the frame, and a non-zero linger lets a final abort
        # flush on close.  With DONTWAIT the very first send raised Again before
        # the asynchronous connect completed, which killed this thread and left
        # the operator channel permanently silent, heartbeats included.
        socket.linger = 500
        socket.sndtimeo = 200
        socket.connect(args.endpoint)
        sequence = 0
        delivered = 0
        try:
            # PUSH has no connection acknowledgement.  Retain the independent
            # heartbeat client long enough for the TCP/ZMQ handshake before an
            # operator command can be consumed from a piped or interactive
            # terminal.
            stop.wait(0.25)
            while not stop.is_set():
                operation = HEARTBEAT
                try:
                    operation = actions.get_nowait()
                except queue.Empty:
                    pass
                # `CLOCK_MONOTONIC` values are meaningful only when this
                # verifier and the native loop share one host.  The normal PC
                # client deliberately sends zero: it must not report a false
                # one-way latency across independently clocked hosts.
                sent_monotonic_ns = time.monotonic_ns() if args.same_host_loopback_clock else 0
                try:
                    socket.send(payload(sequence, operation, sent_monotonic_ns))
                except zmq.Again:
                    # No peer yet, or the queue is full.  Keep the command and
                    # retry on the next beat rather than losing it.
                    if operation != HEARTBEAT:
                        actions.put(operation)
                    if delivered == 0:
                        print("operator channel has not delivered a frame yet", file=sys.stderr, flush=True)
                    stop.wait(args.heartbeat_seconds)
                    continue
                delivered += 1
                if operation == MANUAL_ABORT:
                    abort_sent.set()
                sequence += 1
                stop.wait(args.heartbeat_seconds)
        finally:
            socket.close()

    thread = threading.Thread(target=sender, name="g1-bringup-operator-heartbeat", daemon=True)
    thread.start()
    print("Native operator control is live. Type advance, abort, or quit.", flush=True)
    try:
        for line in sys.stdin:
            command = line.strip().lower()
            if command == "advance":
                actions.put(ADVANCE)
                print("advance requested", flush=True)
            elif command == "abort":
                actions.put(MANUAL_ABORT)
                print("manual abort requested", flush=True)
            elif command in {"quit", "exit"}:
                actions.put(MANUAL_ABORT)
                break
            elif command:
                print("expected: advance, abort, or quit", file=sys.stderr, flush=True)
    finally:
        # Give the sender one heartbeat period to deliver the requested abort;
        # if delivery is impossible, the native deadman still aborts locally.
        abort_sent.wait(timeout=args.heartbeat_seconds * 2)
        time.sleep(args.heartbeat_seconds * 2)
        stop.set()
        thread.join(timeout=args.heartbeat_seconds * 2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--heartbeat-seconds", type=float, default=0.05)
    parser.add_argument("--same-host-loopback-clock", action="store_true",
                        help="record native-monotonic send timestamps for a same-host loopback verifier only")
    args = parser.parse_args()
    if not (0.0 < args.heartbeat_seconds <= 0.25):
        raise ValueError("--heartbeat-seconds must be in (0, 0.25]")
    run(args)


if __name__ == "__main__":
    main()
