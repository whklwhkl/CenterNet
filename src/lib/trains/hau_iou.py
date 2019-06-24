import torch


class WeightedHausdorffDistance(torch.nn.Module):
    def __init__(self, heat_map_shape, alpha=-5):
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

    def forward(self, heat_map, points):
        """ heat_map N*M:[0,1], points [(y,x), ...]
            `points` SHOULD NOT be empty
        """
        # print(self.grid.shape, points.shape)
        delta = self.grid - points[None, None]
        dist = delta.norm(2, -1)
        # breakpoint()
        weighted_min = torch.sum(heat_map * dist.min(-1)[0]) / heat_map.norm(1)
        heat_map = heat_map[..., None]
        weighted_dist = heat_map * dist + (1 - heat_map) * dist.max()
        mean_soft_min = self.soft_min_fn(weighted_dist).mean()
        return weighted_min + mean_soft_min


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
    iou_size = torch.min(torch.div(size1, size_), torch.div(size_, size1))
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
