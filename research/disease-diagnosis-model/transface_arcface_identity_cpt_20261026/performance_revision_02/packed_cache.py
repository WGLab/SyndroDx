"""Read verified RGB pixels from a local immutable packed cache."""
import hashlib
import mmap
from pathlib import Path
from PIL import Image
from control import load,digest

class PackedRGB:
    def __init__(self,folder,source_index_sha256):
        self.folder=Path(folder)
        receipt=load(self.folder/'receipt.json')
        if receipt.get('state')!='complete' or receipt.get('source_index_sha256')!=source_index_sha256:
            raise RuntimeError('verified_packed_cache_required')
        index_path=self.folder/'index.private.json'
        if digest(index_path)!=receipt.get('index_sha256'):
            raise RuntimeError('packed_cache_index_mismatch')
        self.records=load(index_path)['images']
        self.path=self.folder/'train_rgb.private.bin'
        self.size=self.path.stat().st_size
        if self.size!=receipt['bytes'] or len(self.records)!=receipt['images']:
            raise RuntimeError('packed_cache_incomplete')
        self.handle=None;self.mapping=None;self.checked=set()

    def image(self,sha):
        if self.mapping is None:
            self.handle=self.path.open('rb')
            self.mapping=mmap.mmap(self.handle.fileno(),0,access=mmap.ACCESS_READ)
        record=self.records[sha]
        offset=record['offset'];width=record['width'];height=record['height'];length=record['bytes']
        if min(width,height)<1 or max(width,height)>256 or length!=width*height*3 or offset<0 or offset+length>self.size:
            raise RuntimeError('packed_cache_record_bounds')
        raw=memoryview(self.mapping)[offset:offset+length]
        try:
            if sha not in self.checked:
                if hashlib.sha256(raw).hexdigest()!=record['pixel_sha256']:
                    raise RuntimeError('packed_cache_pixel_checksum_mismatch')
                self.checked.add(sha)
            return Image.frombytes('RGB',(width,height),raw)
        finally:raw.release()

    def close(self):
        if self.mapping is not None:self.mapping.close();self.mapping=None
        if self.handle is not None:self.handle.close();self.handle=None

    def __del__(self):
        try:self.close()
        except Exception:pass
