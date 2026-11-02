"""Synthetic pixel equality and corruption tests for packed RGB cache."""
import hashlib
import io
import tempfile
import unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from PIL import Image
from packed_cache import PackedRGB
from control import save,digest
from data import OneView

class PackedCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.pixels=np.random.default_rng(7).integers(0,256,size=(256,192,3),dtype=np.uint8)
        raw=self.pixels.tobytes();(self.root/'train_rgb.private.bin').write_bytes(raw)
        save(self.root/'index.private.json',dict(images={'synthetic':dict(offset=0,width=192,height=256,
            bytes=len(raw),pixel_sha256=hashlib.sha256(raw).hexdigest())}))
        save(self.root/'receipt.json',dict(state='complete',source_index_sha256='source-cache',
            index_sha256=digest(self.root/'index.private.json'),images=1,bytes=len(raw)))
        self.reader=None

    def tearDown(self):
        if self.reader:self.reader.close()
        self.temp.cleanup()

    def test_pixels_equal_original_png_decode(self):
        self.reader=PackedRGB(self.root,'source-cache')
        image=self.reader.image('synthetic')
        np.testing.assert_array_equal(np.asarray(image),self.pixels)
        image.close()

    def test_augmentation_is_identical_with_same_rng_state(self):
        self.reader=PackedRGB(self.root,'source-cache')
        raw=io.BytesIO();Image.fromarray(self.pixels).save(raw,format='PNG');raw.seek(0)
        with Image.open(raw) as source:
            original=source.convert('RGB')
        packed=self.reader.image('synthetic');transform=OneView()
        for seed in range(5):
            torch.manual_seed(seed);a=transform(original)
            torch.manual_seed(seed);b=transform(packed)
            torch.testing.assert_close(a,b,rtol=0.,atol=0.)
        original.close();packed.close()

    def test_wrong_source_index_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'verified_packed_cache'):
            PackedRGB(self.root,'different-training-cache')

    def test_changed_index_rejected(self):
        (self.root/'index.private.json').write_text('{}')
        with self.assertRaisesRegex(RuntimeError,'index_mismatch'):
            PackedRGB(self.root,'source-cache')

    def test_changed_pixel_rejected(self):
        path=self.root/'train_rgb.private.bin'
        with path.open('r+b') as stream:stream.write(b'changed')
        self.reader=PackedRGB(self.root,'source-cache')
        with self.assertRaisesRegex(RuntimeError,'pixel_checksum_mismatch'):
            self.reader.image('synthetic')

    def test_truncated_pack_rejected(self):
        (self.root/'train_rgb.private.bin').write_bytes(b'short')
        with self.assertRaisesRegex(RuntimeError,'packed_cache_incomplete'):
            PackedRGB(self.root,'source-cache')

    def test_record_bounds_rejected(self):
        save(self.root/'index.private.json',dict(images={'synthetic':dict(offset=999999,width=192,height=256,
            bytes=self.pixels.nbytes,pixel_sha256=hashlib.sha256(self.pixels.tobytes()).hexdigest())}))
        save(self.root/'receipt.json',dict(state='complete',source_index_sha256='source-cache',
            index_sha256=digest(self.root/'index.private.json'),images=1,bytes=self.pixels.nbytes))
        self.reader=PackedRGB(self.root,'source-cache')
        with self.assertRaisesRegex(RuntimeError,'record_bounds'):
            self.reader.image('synthetic')

if __name__=='__main__':unittest.main()
