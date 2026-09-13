"""Read dense float weights from the owned factory23 MNN FlatBuffer.

Schema sources: Alibaba MNN schema/default/MNN.fbs and CaffeOp.fbs. This reads
serialized data only. It does not execute converter tools or robot binaries.
"""
import struct
import json
import hashlib
import argparse
from pathlib import Path
import numpy as np
from run_factory_mimic_sim import FIRMWARE,ASSETS


class Table:
    def __init__(self,data,pos):self.data,self.pos=data,pos
    def scalar(self,pos,fmt):return struct.unpack_from('<'+fmt,self.data,pos)[0]
    def field(self,index):
        vt=self.pos-self.scalar(self.pos,'i');size=self.scalar(vt,'H');offset=4+2*index
        if offset>=size:return 0
        value=self.scalar(vt+offset,'H')
        return self.pos+value if value else 0
    def integer(self,index,default=0):
        p=self.field(index);return self.scalar(p,'i') if p else default
    def target(self,index):
        p=self.field(index);return p+self.scalar(p,'I') if p else 0
    def child(self,index):return Table(self.data,self.target(index))
    def text(self,index):
        p=self.target(index);n=self.scalar(p,'I');return self.data[p+4:p+4+n].decode('utf8')
    def vector(self,index,dtype):
        p=self.target(index)
        return np.frombuffer(self.data,dtype,count=self.scalar(p,'I'),offset=p+4).copy() if p else np.empty(0,dtype)
    def tables(self,index):
        p=self.target(index);n=self.scalar(p,'I')
        return [Table(self.data,p+4+i*4+self.scalar(p+4+i*4,'I')) for i in range(n)]


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',type=Path,default=ASSETS/'policies/mimic_test/yangliu_g1_mimic_cxk_new_38600_scuffing.mnn')
    ap.add_argument('--output',type=Path,default=FIRMWARE/'native23_trainable_v1')
    args=ap.parse_args();path=args.model
    data=path.read_bytes();root=Table(data,struct.unpack_from('<I',data)[0])
    weights={};layers=[]
    for op in root.tables(3):
        if op.integer(5)!=12:continue
        name=op.text(3)
        conv=op.child(2);common=conv.child(0)
        out_count,in_count=common.integer(10),common.integer(11)
        if common.integer(2,1)!=1 or common.integer(3,1)!=1:raise ValueError('not a dense1x1 convolution')
        w,b=conv.vector(1,'<f4'),conv.vector(2,'<f4')
        if len(w)!=out_count*in_count or len(b)!=out_count:raise ValueError('unsupported compressed or external weights')
        prefix=name.removeprefix('/actor/').split('/Gemm')[0].replace('/','.')
        weights[prefix+'.weight']=w.reshape(out_count,in_count);weights[prefix+'.bias']=b
        layers.append({'name':prefix,'input':in_count,'output':out_count})
    if not layers:raise ValueError('no dense factory layers found')
    folder=args.output;folder.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(folder/'factory_weights.npz',**weights)
    report={'source_sha256':hashlib.sha256(data).hexdigest(),'layers':layers,
        'schema_sources':['https://github.com/alibaba/MNN/blob/master/schema/default/MNN.fbs',
            'https://github.com/alibaba/MNN/blob/master/schema/default/CaffeOp.fbs']}
    (folder/'extraction.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
