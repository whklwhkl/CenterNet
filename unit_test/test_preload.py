import unittest

from lib.datasets.dataset.wider2019pd_preload import ValidationSet, remove_ignored_det, ltrb2ltwh, ltwh2ltrb


class Tester(unittest.TestCase):

    def setUp(self) -> None:
        self.val = ValidationSet('/home/wanghao/datasets/WIDER_pd2019')
        import pickle
        self.gt = pickle.load(open('/home/wanghao/PycharmProjects/reid/CenterNet/unit_test/gt.pkl', 'rb'))
        self.k1 = list(self.gt.keys())
        self.k2 = list(self.val.gt_map.keys())
        self.l1 = len(self.k1)
        self.l2 = len(self.k2)

    def test_keys(self):
        assert set(self.k1) == set(self.k2), 'image ids are not the same'

    def test_len(self):
        assert self.l1==self.l2, f'{self.l1}!={self.l2}'

    def test_box_convert(self):
        cbox = [[1,2,3,4]]
        bbox = [[1,2,4,6]]
        assert cbox==ltrb2ltwh(bbox), f'ltrb2ltwh, {ltrb2ltwh(bbox)}'
        cbox = [[1, 2, 3, 4]]
        bbox = [[1, 2, 4, 6]]
        assert bbox==ltwh2ltrb(cbox), f'ltwh2ltrb, {cbox}=>{ltwh2ltrb(cbox)}'

    def test_box_len(self):
        for im, bb in self.gt.items():
            bb = bb['bbox']
            read_bb = self.val.gt_map[im]
            assert len(bb) == len(read_bb), f'{bb}\n\n{read_bb}'

    def test_read_txt(self):
        for im, bb in self.gt.items():
            bb = bb['bbox']
            read_bb = self.val.gt_map[im]
            for b1, b2 in zip(bb, read_bb):
                assert b1 == b2, f'{b1}!={b2}'

