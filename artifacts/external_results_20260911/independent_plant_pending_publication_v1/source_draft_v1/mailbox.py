"""Bounded copied byte slots; supplied locks are always tried exactly once.

This foundation is an in-process wrapper. A real spawned-process implementation
must replace these bytearrays with shared memory and validate publication timing.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Publication:
    key: int
    version: int
    payload: bytes


class ByteSlot:
    def __init__(self, capacity, lock):
        if capacity <= 0:
            raise ValueError('positive fixed payload capacity required')
        self.buffer = bytearray(capacity)
        self.lock = lock
        self.length = 0
        self.key = None
        self.version = 0

    def try_publish(self, key, payload):
        if not isinstance(payload, bytes) or len(payload) > len(self.buffer):
            return 'OVERSIZE_OR_MUTABLE'
        if not self.lock.acquire(False):
            return 'BUSY'
        try:
            if self.key is not None:
                return 'FULL'
            self.buffer[:len(payload)] = payload
            self.length = len(payload)
            self.version += 1
            self.key = key  # Published last, while still owning the lock.
            return 'PUBLISHED'
        finally:
            self.lock.release()

    def try_take(self):
        if not self.lock.acquire(False):
            return 'BUSY', None
        try:
            if self.key is None:
                return 'EMPTY', None
            owned = bytes(memoryview(self.buffer)[:self.length])
            result = Publication(self.key, self.version, owned)
            self.key = None
            self.length = 0
            return 'TAKEN', result
        finally:
            self.lock.release()


class Mailbox:
    """Fixed ring; an occupied older slot cannot overwrite a newer publication."""
    def __init__(self, slots, payload_capacity, lock_factory):
        if slots <= 0:
            raise ValueError('positive slot count required')
        self.slots = tuple(ByteSlot(payload_capacity, lock_factory()) for _ in range(slots))

    def try_publish(self, key, payload):
        if type(key) is not int or key < 0:
            return 'BAD_KEY'
        return self.slots[key % len(self.slots)].try_publish(key, payload)

    def poll_once(self):
        # Fixed finite work; no retry, timeout wait, queue put, or disk write.
        return tuple((index, *slot.try_take()) for index, slot in enumerate(self.slots))
