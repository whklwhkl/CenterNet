import torch
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
from time import time


TIMEWALL = 5

class WeightedHausdorffDistance(torch.nn.Module):
    def __init__(self, heat_map_shape, eps=1e-5, alpha=-1, trainer=None):
        """
        This is a pytorch implementation
        of the [Locating Objects Without Bounding Boxes](https://arxiv.org/pdf/1806.07564.pdf)
        note: this distance metric takes effect when the input heatmap values are within [0,1]
        SINGLE_IMAGE
        """
        super().__init__()
        self.grid = torch.stack(
            torch.meshgrid(
                *map(lambda x:torch.arange(x, dtype=torch.float),
                     heat_map_shape)), -1)[:, :, None].cuda()     # 2D grid shape: height, width, 2
        self.soft_min_fn = lambda x: x.pow(alpha).mean(0).mean(0).pow(1 / alpha)  # alpha --> -infi ==> min_func
        self.heat_map_shape = torch.FloatTensor([[heat_map_shape]]).cuda()
        self.eps = eps
        self.trainer = trainer
        if trainer is not None:
            self.summary_writer = trainer.summary_writer    # type: SummaryWriter
        else:
            self.summary_writer = None
        self.last = time()

    def forward(self, heat_map, points):
        """ heat_map N*M:[0,1], points [(y,x), ...]
            `points` SHOULD NOT be empty
        """
        # print(self.grid.shape, points.shape)
        delta = self.grid - points[None, None]
        delta = delta / self.heat_map_shape     # norm distance w.r.t x-y axis
        dist = delta.norm(2, -1)
        # breakpoint()
        weighted_min = torch.sum(heat_map * dist.min(-1)[0]) / (self.eps + heat_map.norm(1)) # term 1
        heat_map = heat_map[..., None]
        weighted_dist = heat_map * dist + (1 - heat_map) * dist.max()
        mean_soft_min = self.soft_min_fn(weighted_dist).mean()                  # term 2
        if self.summary_writer is not None:
            if time() - self.last > TIMEWALL:
                self.summary_writer.add_scalar('WHD_avg_x_min_y', weighted_min.item(), self.trainer.global_steps, TIMEWALL)
                self.summary_writer.add_scalar('WHD_avg_y_min_x', mean_soft_min.item(), self.trainer.global_steps, TIMEWALL)
                fig = plt.Figure(frameon=False)
                ax = fig.add_subplot(111)
                ax.imshow(heat_map.detach().cpu().numpy()[..., 0])
                points = points.detach().cpu().numpy()
                ax.scatter(points[:, 1], points[:, 0], c='r', marker='X')
                self.summary_writer.add_figure('heat_map', fig, self.trainer.global_steps, TIMEWALL)
                self.summary_writer.add_histogram('dist', dist.detach().cpu().numpy(),
                                                  self.trainer.global_steps, TIMEWALL)
                self.summary_writer.add_histogram('weighted_dist', weighted_dist.detach().cpu().numpy(),
                                                  self.trainer.global_steps, TIMEWALL)
                plt.close(fig)
                self.last = time()
        return weighted_min + mean_soft_min     # todo: weighting


def bounded_iou_loss(loc1, size1, loc2, size_):
    """
    loc1/2:[(l,t),...]
    size1/2:[(w,h),...]
    2 |= target
    :return 0 if same, >0 if diff
    SINGLE_IMAGE
    """
    loc_delta = torch.abs(loc1 - loc2)
    iou_loc = torch.div(size_ - loc_delta, size_ + loc_delta)
    iou_size = torch.min(torch.div(size1, size_ + 1e-5), torch.div(size_, size1 + 1e-5))
    bounded_iou = torch.mean(iou_loc + iou_size) / 2    # average over localization and size iou
    return 1 - bounded_iou


def bounded_iou_loss_box(box1, box2):
    """
    box:[(l,t,w,h),...]
    box2 |= target
    :return 0 if same, >0 if diff
    SINGLE_IMAGE
    """
    loc1, size1 = torch.chunk(box1, 2, -1)
    loc2, size_ = torch.chunk(box2, 2, -1)
    return bounded_iou_loss(loc1, size1, loc2, size_)
