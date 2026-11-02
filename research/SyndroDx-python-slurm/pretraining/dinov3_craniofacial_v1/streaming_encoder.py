"""Write an identical single safetensors file one tensor at a time.

Keeps native parameter precision while bounding rank-zero host memory during
large encoder export. Temporary output becomes visible only after caller rename.
"""
import json,math,os,struct
import torch

DTYPES={torch.bool:'BOOL',torch.uint8:'U8',torch.int8:'I8',torch.int16:'I16',torch.int32:'I32',torch.int64:'I64',torch.float16:'F16',torch.bfloat16:'BF16',torch.float32:'F32',torch.float64:'F64'}
class StreamingEncoderWriter:
 def __init__(self,path,state):
  self.order=list(state);self.next=0;self.offset=0;self.spec={};header={'__metadata__':{'format':'pt'}}
  for name,value in state.items():
   if value.dtype not in DTYPES:raise ValueError('unsupported_encoder_export_dtype')
   size=math.prod(value.shape)*value.element_size();shape=list(value.shape)
   header[name]={'dtype':DTYPES[value.dtype],'shape':shape,'data_offsets':[self.offset,self.offset+size]};self.spec[name]=(value.dtype,tuple(shape),size);self.offset+=size
  encoded=json.dumps(header,separators=(',',':'),ensure_ascii=False).encode('utf-8');encoded+=b' '*((-len(encoded))%8)
  self.file=open(path,'wb');self.file.write(struct.pack('<Q',len(encoded)));self.file.write(encoded)
 def write(self,name,value):
  if self.next>=len(self.order) or self.order[self.next]!=name:raise ValueError('encoder_export_tensor_order_changed')
  dtype,shape,size=self.spec[name]
  if value.dtype!=dtype or tuple(value.shape)!=shape:raise ValueError('encoder_export_tensor_schema_changed')
  cpu=value.detach().to('cpu').contiguous();raw=memoryview(cpu.reshape(-1).view(torch.uint8).numpy())
  if raw.nbytes!=size:raise ValueError('encoder_export_tensor_size_changed')
  self.file.write(raw);self.next+=1
 def close(self):
  try:
   if self.next!=len(self.order):raise ValueError('encoder_export_incomplete')
   self.file.flush();os.fsync(self.file.fileno())
  finally:self.file.close()
 def abort(self):self.file.close()
