"""Synthetic cache equivalence, corruption recovery, and source integrity."""
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from PIL import Image
import cache_inputs

class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.override=patch.object(cache_inputs,'WORK',self.root);self.override.start()
        self.source=self.root/'synthetic.png'
        value=np.random.default_rng(2).integers(0,256,size=(360,270,3),dtype=np.uint8)
        Image.fromarray(value).save(self.source)
        self.sha=hashlib.sha256(self.source.read_bytes()).hexdigest()
    def tearDown(self):
        self.override.stop();self.temp.cleanup()
    def test_evaluation_cache_exactly_preserves_reference_preprocessing(self):
        result=cache_inputs.materialize((str(self.source),self.sha,'eval'))
        self.assertIsNotNone(result)
        actual=np.load(cache_inputs.destination(self.sha,'eval'),allow_pickle=False)
        image=cv2.imread(str(self.source),cv2.IMREAD_COLOR)
        image=cv2.cvtColor(cv2.resize(image,(112,112)),cv2.COLOR_BGR2RGB)
        expected=np.ascontiguousarray(image.transpose(2,0,1),dtype=np.float32)
        expected/=255.;expected-=.5;expected/=.5
        np.testing.assert_array_equal(actual,expected)
    def test_training_cache_resizes_and_repairs_corruption(self):
        first=cache_inputs.materialize((str(self.source),self.sha,'train'))
        path=cache_inputs.destination(self.sha,'train')
        with Image.open(path) as image:self.assertEqual(image.size,(192,256))
        path.write_bytes(b'corrupt')
        second=cache_inputs.materialize((str(self.source),self.sha,'train'))
        self.assertEqual(first,second)
    def test_changed_source_rejected_before_first_derivation(self):
        self.source.write_bytes(b'changed')
        self.assertIsNone(cache_inputs.materialize((str(self.source),self.sha,'train')))

if __name__=='__main__':unittest.main()
