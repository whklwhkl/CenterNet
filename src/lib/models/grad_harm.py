import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientHarmonization(nn.Module):
    def __init__(self, loss_N_grad_fn, disabled=False, momentum=.9, number_of_bins=30):
        super().__init__()
        self.exponential_moving_average = None
        self.loss_N_grad_fn = loss_N_grad_fn
        self.number_of_bins = number_of_bins
        self.momentum = momentum
        self.momentum_ = 1-momentum
        self.disabled = disabled

    def forward(self, pred, true):
        loss, gradient_norm = self.loss_N_grad_fn(pred, true)

        ind = (gradient_norm.clamp_max(.999) * self.number_of_bins).cpu().long()
        freq = torch.bincount(ind, minlength=self.number_of_bins).float()
        if self.exponential_moving_average is None:
            self.exponential_moving_average = freq
        else:
            self.exponential_moving_average = self.momentum * self.exponential_moving_average + self.momentum_ * freq
        density = self.number_of_bins * (self.exponential_moving_average + 1)  # Laplace's correction, preventing 0-div

        if self.disabled:
            return loss.mean(), gradient_norm.norm(2).item()/len(gradient_norm)
        denominator = density[ind].cuda()
        reformulated = loss / denominator
        gradient_norm_of_a_batch = gradient_norm.div(density[ind].cuda()).norm(2).item()
        return reformulated.sum(), gradient_norm_of_a_batch


def log_sum_exp(x):
    return x.exp().sum(-1).log()


def cross_entropy(logit, ground):
    """
    :param logit: from net
    :param ground: Ground label, with the last dimension being 0 ~ Classes-1
    :return: loss and the corresponding gradient norm
    """
    # loss = -torch.log(probability.clamp(1e-5)).gather(-1, ground).view(-1)
    loss = log_sum_exp(logit) - logit.gather(-1, ground).view(-1)
    # l = - \Sigma y_i ln p_i
    prob = F.softmax(logit, -1).detach()
    residule = torch.gather(1-2*prob, -1, ground).view(-1)
    quad_prob = prob.pow_(2).sum(-1)
    # (p-g)^2 = | p^2           , g==0
    #           | p^2 - 2p + 1  , g==1
    gradient_norm = residule.add_(quad_prob).sqrt_()  # ([698368, 1])
    return loss, gradient_norm

mu = .5
# mu = 0.682397
# mu = 1
# mu = 2
mu2 = mu**2
def authentic_smooth_l1(pred, true):
    """
    replace smooth L1,
    when mu==0.682397, they integrate to the same value within [0,2]
    :param pred: from net
    :param true: ground truth, has the same shape as pred
    :return: loss and gradient in accordance
    """
    d = pred.sub(true).view(-1)         # [N * 4]
    radius = d.pow(2).add(mu2).sqrt()
    loss = radius.sub(mu)
    gradient = abs(d) / radius
    gradient_norm = gradient.detach()   # make sure the norm is within [0, 1)
    return loss, gradient_norm


def smooth_l1(pred, true):
    d = pred.sub(true).view(-1).abs()
    cond = d.lt(1)
    loss = torch.where(cond, d.pow(2)/2, d-.5)
    gradient_norm = torch.where(cond, d, torch.ones_like(d))
    return loss, gradient_norm.detach()


class IoU_Loss(nn.Module):
    def __init__(self, x_var, w_var):
        super().__init__()

        def iou_loss_x(x_norm, xt_norm, wt_log_norm):
            """
            gradient are ensure to be within (0, 1]
            :param x_norm:
            :param xt_norm:
            :param wt_log_norm:
            :return:
            """
            delta = torch.abs(x_norm - xt_norm) * x_var
            wt = torch.exp(wt_log_norm * w_var)
            w_plus_dx = wt + delta
            loss = wt * torch.log(w_plus_dx) - wt * torch.log(wt)
            grad = wt / w_plus_dx
            return loss, grad.detach()

        def iou_loss_w(w_log_norm, wt_log_norm):
            """
            gradient are ensure to be within (0, 1]
            loss = [IoU(w) - 1 - log(IoU(w))]/var_w, where IoU(w)== exp[- var_w * abs(w_log_norm - wt_log_norm)]
            :param w_log_norm: log(w)/var_w
            :param wt_log_norm: log(w_t)/var_w
            :return: loss and the corresponding abs(gradient)
            """
            delta_log = abs(w_log_norm - wt_log_norm) * w_var
            iou = delta_log.neg().exp()
            loss = (delta_log + iou - 1) / w_var
            return loss, 1 - iou.detach()

        self.loss_x = iou_loss_x
        self.loss_w = iou_loss_w

    def forward(self, pred, true):
        x_n, y_n, w_ln, h_ln = pred.split(1, -1)
        xt_n, yt_n, wt_ln, ht_ln = true.split(1, -1)
        xl, xg = self.loss_x(x_n, xt_n, wt_ln)
        yl, yg = self.loss_x(y_n, yt_n, ht_ln)
        wl, wg = self.loss_w(w_ln, wt_ln)
        hl, hg = self.loss_w(h_ln, ht_ln)
        loss = torch.cat([xl, yl, wl, hl])
        grad = torch.cat([xg, yg, wg, hg])
        # todo: divide the localization task into two tasks: regress point (x,y), estimate scale (w,h)
        return loss.view(-1), grad.view(-1)