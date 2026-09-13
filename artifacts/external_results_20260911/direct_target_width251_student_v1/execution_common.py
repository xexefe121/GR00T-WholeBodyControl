"""Saved-only warm-fit accounting; numerical completion remains separate."""
import ctypes
from ctypes import wintypes
import hashlib
import json
import math
from pathlib import Path
BACKENDS=('initial_GPU32','final_GPU32','CPU64','GPU64','ORT64')
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def canonical_pins(pins):
    result={}
    for path,digest in pins.items():
        key=Path(path).resolve().as_posix().casefold()
        if key in result:raise ValueError('Duplicate physical pin path.')
        result[key]=digest
    return result
def validate_process_pin_report(record,expected_pins):
    if record['all_exact'] is not True or record['count']!=len(expected_pins):raise ValueError('Process pin count/pass differs.')
    actual={path:item['expected'] for path,item in record['files'].items()}
    if canonical_pins(actual)!=canonical_pins(expected_pins):raise ValueError('Process pin membership differs.')
    for path,item in record['files'].items():
        if item['matched'] is not True or item['actual']!=item['expected'] or sha(path)!=item['expected']:raise ValueError('Process pin record differs: '+path)
def absent(pid):
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=(wintypes.DWORD,wintypes.BOOL,wintypes.DWORD);kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes=(wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD));kernel.GetExitCodeProcess.restype=wintypes.BOOL
    kernel.CloseHandle.argtypes=(wintypes.HANDLE,);kernel.CloseHandle.restype=wintypes.BOOL
    handle=kernel.OpenProcess(0x1000,False,int(pid))
    if not handle:
        if ctypes.get_last_error()==87:return True
        raise OSError(ctypes.get_last_error(),'Cannot prove process absence')
    try:
        code=wintypes.DWORD()
        if not kernel.GetExitCodeProcess(handle,ctypes.byref(code)):raise OSError(ctypes.get_last_error(),'Cannot inspect process')
        return code.value!=259
    finally:kernel.CloseHandle(handle)
def expected(calls,rows):return dict(calls_attempted=calls,calls_returned=calls,calls_synchronized=calls,calls_verified=calls,rows_attempted=rows,rows_returned=rows,rows_verified=rows)
def validate_counter(value,calls,rows,complete):
    if set(value)!=set(expected(0,0)):raise ValueError('Counter keys differ.')
    if any(type(v)!=int or v<0 for v in value.values()):raise ValueError('Counter must be nonnegative integers.')
    c=[value['calls_'+k] for k in ('attempted','returned','synchronized','verified')]
    r=[value['rows_'+k] for k in ('attempted','returned','verified')]
    if c!=sorted(c,reverse=True) or r!=sorted(r,reverse=True) or c[0]>calls or r[0]>rows:raise ValueError('Counter prefix/budget differs.')
    if complete and value!=expected(calls,rows):raise ValueError('Incomplete counter on completed condition.')
