import unittest
import numpy as np
import cv2
from benchmark import preprocess,metric_driver

class BaselineTests(unittest.TestCase):
    def test_official_rgb_order_and_normalization(self):
        image=np.zeros((112,112,3),dtype=np.uint8);image[:]=[0,128,255]
        value=preprocess(image)
        self.assertEqual(value.shape,(3,112,112));self.assertEqual(value.dtype,np.float32)
        np.testing.assert_allclose(value[:,0,0],[1,128/255*2-1,-1],atol=1e-7)
    def test_official_resize_handles_input_shapes(self):
        image=np.full((56,200,3),255,dtype=np.uint8)
        self.assertTrue(np.all(preprocess(image)==1))
    def test_shared_metric_driver_is_loaded(self):
        self.assertTrue(callable(metric_driver()))

if __name__=='__main__':unittest.main()
