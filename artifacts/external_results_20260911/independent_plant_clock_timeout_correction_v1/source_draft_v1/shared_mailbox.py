"""Fixed shared-memory byte slots with one nonblocking process-lock attempt.

Only trusted same-host spawned peers are in scope. No owner-death recovery,
network transport, serialization of application payloads, or real-time claim.
"""
import ctypes
from dataclasses import dataclass
import hashlib
import struct
from mailbox import Publication

MAGIC=b'S23MBOX1';SCHEMA=1
EMPTY,WRITING,FULL=0,1,2
HEADER=struct.Struct('<8sIIQQI32s32s')
ZERO_DIGEST=bytes(32);MAX_U64=(1<<64)-1
MAX_SLOTS=1024;MAX_PAYLOAD=1024*1024;MAX_TOTAL=64*1024*1024

def epoch_token(binding_bytes):
    """Bind a preadmitted canonical full run/model/reference/epoch identity."""
    if type(binding_bytes) is not bytes or not binding_bytes or len(binding_bytes)>4096:raise ValueError('bounded immutable canonical binding required')
    return hashlib.sha256(binding_bytes).digest()

@dataclass(frozen=True)
class SharedLayout:
    slots:int
    payload_capacity:int
    epoch_token:bytes
    memory:object
    locks:tuple

class SharedMailbox:
    def __init__(self,layout,expected_epoch_token):
        if type(expected_epoch_token) is not bytes or len(expected_epoch_token)!=32:raise ValueError('literal 32-byte epoch token required')
        self._layout=layout;self._expected_epoch_token=expected_epoch_token;self._closed=False
    @classmethod
    def create(cls,context,slots,payload_capacity,token):
        if context.get_start_method()!='spawn':raise ValueError('explicit spawn context required')
        if type(slots) is not int or not 0<slots<=MAX_SLOTS or type(payload_capacity) is not int or not 0<payload_capacity<=MAX_PAYLOAD:raise ValueError('positive bounded integer slots/capacity required')
        if type(token) is not bytes or len(token)!=32:raise ValueError('literal 32-byte epoch token required')
        total=slots*(HEADER.size+payload_capacity)
        if total>MAX_TOTAL:raise ValueError('fixed transport allocation exceeds64MiB')
        layout=SharedLayout(slots,payload_capacity,token,context.RawArray(ctypes.c_ubyte,total),tuple(context.Lock() for _ in range(slots)))
        instance=cls(layout,token)
        for i in range(slots):instance._store_header(i,EMPTY,0,0,0,ZERO_DIGEST)
        return instance
    @property
    def slots_count(self):return self._layout.slots
    @property
    def payload_capacity(self):return self._layout.payload_capacity
    def endpoint(self,token):
        """A spawned peer endpoint; changing the expected epoch cannot alter storage."""
        return type(self)(self._layout,token)
    def close(self):
        """Close this process-local endpoint only; no shared unlock/reset/unlink."""
        self._closed=True
    def _offset(self,index):return index*(HEADER.size+self.payload_capacity)
    def _view(self):return memoryview(self._layout.memory).cast('B')
    def _load_header(self,index):
        view=self._view()
        try:return HEADER.unpack_from(view,self._offset(index))
        finally:view.release()
    def _store_header(self,index,state,key,version,length,digest):
        view=self._view()
        try:HEADER.pack_into(view,self._offset(index),MAGIC,SCHEMA,state,key,version,length,self._layout.epoch_token,digest)
        finally:view.release()
    def _copy_payload(self,index,payload):
        view=self._view();start=self._offset(index)+HEADER.size
        try:view[start:start+len(payload)]=payload
        finally:view.release()
    def _header_valid(self,index,header):
        magic,schema,state,key,version,length,token,digest=header
        return magic==MAGIC and schema==SCHEMA and state in (EMPTY,WRITING,FULL) and token==self._layout.epoch_token and length<=self.payload_capacity and (state!=FULL or (version>0 and key%self.slots_count==index))
    def _entry_status(self):
        if self._closed:return 'CLOSED'
        if self._expected_epoch_token!=self._layout.epoch_token:return 'WRONG_EPOCH'
        return None
    def try_publish(self,key,payload):
        status=self._entry_status()
        if status:return status
        if type(key) is not int or not 0<=key<=MAX_U64:return 'BAD_KEY'
        if type(payload) is not bytes or len(payload)>self.payload_capacity:return 'OVERSIZE_OR_MUTABLE'
        index=key%self.slots_count;lock=self._layout.locks[index]
        if not lock.acquire(False):return 'BUSY'
        try:
            header=self._load_header(index)
            if not self._header_valid(index,header):return 'CORRUPT'
            _,_,state,_,version,_,_,_=header
            if state==WRITING:return 'UNCOMMITTED'
            if state==FULL:return 'FULL'
            if version==MAX_U64:return 'VERSION_EXHAUSTED'
            # Any exception before the final FULL header leaves a non-readable slot.
            self._store_header(index,WRITING,key,version,len(payload),ZERO_DIGEST)
            self._copy_payload(index,payload)
            digest=hashlib.sha256(payload).digest()
            self._store_header(index,FULL,key,version+1,len(payload),digest)
            return 'PUBLISHED'
        finally:lock.release()
    def _try_take(self,index):
        status=self._entry_status()
        if status:return status,None
        lock=self._layout.locks[index]
        if not lock.acquire(False):return 'BUSY',None
        try:
            header=self._load_header(index)
            if not self._header_valid(index,header):return 'CORRUPT',None
            _,_,state,key,version,length,_,digest=header
            if state==EMPTY:return 'EMPTY',None
            if state==WRITING:return 'UNCOMMITTED',None
            view=self._view();start=self._offset(index)+HEADER.size
            try:owned=bytes(view[start:start+length])
            finally:view.release()
            if hashlib.sha256(owned).digest()!=digest:return 'CORRUPT',None
            publication=Publication(key,version,owned)
            self._store_header(index,EMPTY,0,version,0,ZERO_DIGEST)
            return 'TAKEN',publication
        finally:lock.release()
    def poll_once(self):
        # Exactly one attempt per fixed slot. No waiting/retry/yield/queue/disk path.
        return tuple((index,*self._try_take(index)) for index in range(self.slots_count))
