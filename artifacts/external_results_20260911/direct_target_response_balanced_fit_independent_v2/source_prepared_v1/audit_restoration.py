"""Compare saved model/optimizer/RNG trees without executing a model."""
import struct
import numpy as np
import torch

def differences(actual, expected, path='root'):
    problems=[]
    if torch.is_tensor(expected):
        if not torch.is_tensor(actual):return [path+': tensor type']
        if actual.dtype != expected.dtype or actual.shape != expected.shape:
            return [path+': tensor dtype/shape']
        a=actual.detach().cpu().numpy();b=expected.detach().cpu().numpy()
        if a.tobytes()!=b.tobytes():problems.append(path+': tensor bytes')
    elif isinstance(expected,np.ndarray):
        if not isinstance(actual,np.ndarray) or actual.dtype!=expected.dtype or actual.shape!=expected.shape or actual.tobytes()!=expected.tobytes():
            problems.append(path+': array bytes/type/shape')
    elif isinstance(expected,dict):
        if not isinstance(actual,dict) or set(actual)!=set(expected):return [path+': dictionary keys']
        for key,value in expected.items():problems.extend(differences(actual[key],value,path+'.'+str(key)))
    elif isinstance(expected,(tuple,list)):
        if type(actual) is not type(expected) or len(actual)!=len(expected):return [path+': sequence type/length']
        for i,value in enumerate(expected):problems.extend(differences(actual[i],value,path+f'[{i}]'))
    elif isinstance(expected,float):
        if type(actual) is not type(expected) or struct.pack('>d',actual)!=struct.pack('>d',expected):problems.append(path+': float bytes/type')
    elif type(actual) is not type(expected) or actual!=expected:
        problems.append(path+': scalar value/type')
    return problems
