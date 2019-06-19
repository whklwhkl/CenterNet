import unittest
import pickle

from lib.datasets.dataset.wider2019pd import WIDER2019


class Test_eval(unittest.TestCase):

    def setUp(self) -> None:
        class Opt:
            def __init__(self):
                self.data_dir = '/home/wanghao/datasets/WIDER_pd2019'
        opt = Opt()
        self.w2019pd = WIDER2019(opt, 'val')
        dts = pickle.load(open('/home/wanghao/PycharmProjects/reid/CenterNet/unit_test/dts.pkl', 'rb'))
        self.result = {}
        for d in dts:
            boxes = self.result.setdefault(d['image_id'], [])
            boxes.append(d['bbox']+[d['score']])
        for k in self.result.keys():
            self.result[k] = {1:self.result[k]}

        self.example_ap = 0.31149255196675985

    def test_eval(self):
        ap = self.w2019pd.run_eval(self.result, '.')
        assert abs(ap-self.example_ap)<1e-8, f'{ap}!={self.example_ap}'