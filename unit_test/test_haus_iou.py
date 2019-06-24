import torch
import unittest
import importlib


from importlib.machinery import SourceFileLoader
loader = SourceFileLoader('hao_iou',
                          '/home/wanghao/PycharmProjects/reid/CenterNet/src/lib/trains/hau_iou.py')
hau_iou = loader.load_module()


class Hau(unittest.TestCase):
    def setUp(self) -> None:
        self.hau = hau_iou.WeightedHausdorffDistance([4,8])
        self.hm = torch.zeros(4,8)
        x = torch.randint(0, 8, [3])
        y = torch.randint(0, 4, [3])
        self.points = torch.stack([y, x], -1)
        # self.points = torch.stack([x, y], -1)
        for y,x in self.points:
            self.hm[y.item(), x.item()]=1

    def test_setup(self):
        print(self.hm, self.points)
        assert True

    def test_loss(self):
        loss1 = self.hau(self.hm, self.points.float())
        assert 0==loss1, f'{loss1} is not 0'
        chaos = torch.rand(4,8)
        loss2 = self.hau(chaos, self.points.float())
        print(loss2)
        assert loss2 > 0, f'{loss2} is not greater than 0'

    def test_regime(self):
        losses = []
        for i in range(1000):
            losses += [self.hau(torch.rand(4, 8), self.points.float())]
        print(min(losses), max(losses))
        assert True

class IOU(unittest.TestCase):
    def setUp(self) -> None:
        self.box1 = torch.tensor([[0,0,1,2]]).float()
        self.box2 = torch.tensor([[0,1,2,1]]).float()
        self.box3 = torch.tensor([[2,2,1,1]]).float()
        print(self.box1);print(self.box2);print(self.box3)
        self.iou_loss = hau_iou.bounded_iou_loss_box

    def test_loss(self):
        ## make sure the loss is semi-positively definite
        same_loss = self.iou_loss(self.box1, self.box1)
        print(same_loss)
        assert torch.allclose(same_loss, torch.FloatTensor(0), atol=1e-4, rtol=1e-4), 'failed same loss check'
        diff_loss = self.iou_loss(self.box1, self.box2)
        print(diff_loss)
        assert diff_loss > 0, f'{diff_loss} is not greater than 0'
        disj_loss = self.iou_loss(self.box1, self.box3)
        print(disj_loss)
        assert disj_loss > 0, f'disjoint loss failed @ {disj_loss}'

    def test_regime(self):
        losses = []
        for i in range(1000):
            losses += [self.iou_loss(torch.rand(1,4), torch.rand(1,4))]
        print(min(losses), max(losses))
        assert True
