import unittest
from torch.utils.tensorboard import SummaryWriter
import random
from lib.datasets.dataset.wider2019pd_preload import TrainingSet
import sys


class Board(unittest.TestCase):
    def setUp(self) -> None:
        def exp(name):
            loss = 10
            lr =1e-1
            for e in range(10):
                writer = SummaryWriter(f'/home/wanghao/PycharmProjects/reid/CenterNet/boarding/{name}')
                for i in range(16):
                    loss *= 1 + lr * (random.random() - 1)
                    writer.add_scalar('loss', loss, e*16+i+100)
                writer.close()
        self.exp = exp

    def test_disp(self):
        self.exp('foo2')
        self.exp('bar2')
        assert True
