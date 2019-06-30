import torch
import torch.nn as nn
import numpy as np

from models.losses import FocalLoss
from models.losses import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss
from models.decode import ctdet_decode
from models.utils import _sigmoid
from utils.debugger import Debugger
from utils.post_process import ctdet_post_process
from .base_trainer import BaseTrainer
from models.utils import _tranpose_and_gather_feat

import imp
imp.load_module('tools', *imp.find_module('tools', ['/home/wanghao/datasets/WIDER_pd2019/']))
from lib.datasets.dataset.wider2019pd import save_results
from tools import evaluate
from termcolor import colored

import os.path as osp


# LOG_W_STD = 0.8467545257117721  # std(log(w))
# LOG_H_STD = 0.8007598947555256  # std(log(h))


def diff(output, mask, ind, target):
    pred = _tranpose_and_gather_feat(output, ind)
    mask = mask.unsqueeze(2).expand_as(pred).float()
    return torch.abs(pred - target) * mask


class Loss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.crit = FocalLoss()
        self.crit_reg = RegL1Loss()
        # todo : bounded iou loss
        self.opt = opt

    def forward(self, outputs, batch):
        opt = self.opt
        hm_loss, wh_loss, off_loss = 0, 0, 0
        for s in range(opt.num_stacks):
            output = outputs[s]
            output['hm'] = _sigmoid(output['hm'])
            hm_loss += self.crit(output['hm'], batch['hm'])
            wh_loss += self.crit_reg(output['wh'], batch['reg_mask'], batch['ind'], batch['wh'])      # std
            off_loss += self.crit_reg(output['reg'], batch['reg_mask'], batch['ind'], batch['reg'])   # std

        wh_loss /= opt.num_stacks
        hm_loss /= opt.num_stacks
        off_loss /= opt.num_stacks
        loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + opt.off_weight * off_loss
        loss_stats = {'loss': loss, 'hm_loss': hm_loss, 'wh_loss': wh_loss, 'off_loss': off_loss}
        return loss, loss_stats


class Trainer2(BaseTrainer):

    def __init__(self, opt, model, optimizer=None):
        super().__init__(opt, model, optimizer=optimizer)
        self.best = None

    def _get_losses(self, opt):
        loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss']
        loss = Loss(opt)
        return loss_states, loss

    def val(self, epoch, data_loader):
        ret, results = super().val(epoch, data_loader)
        detections = {}
        for i, img_id in enumerate(results['img_id']):
            detections[img_id] = results['detections'][i]
        file_name = f'val_{epoch}.txt'
        # breakpoint()
        save_results(detections, self.log_dir, file_name)
        ap = evaluate.get_score(osp.join(self.log_dir, file_name))
        ret['ap'] = -ap  # the main training loop assumes the best model to be of the minimal value
        ap_str = 'AP {:.4f}'.format(ap)
        if self.best is None:
            self.best = ap
            color = 'blue'
        elif self.best < ap:
            self.best = ap
            color = 'green'
        else:
            color = 'red'
        print(f"\033[F\033[{200}G", end='|')
        print(colored(ap_str, color))
        self.val_writer.add_scalar('AP', ap, self.global_steps)
        return ret, results

    def debug(self, batch, output, iter_id):
        opt = self.opt
        reg = output['reg'] if opt.reg_offset else None
        dets = ctdet_decode(
            output['hm'], output['wh'], reg=reg,
            cat_spec_wh=opt.cat_spec_wh, K=opt.K)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets[:, :, :4] *= opt.down_ratio
        dets_gt = batch['meta']['gt_det'].numpy().reshape(1, -1, dets.shape[2])
        dets_gt[:, :, :4] *= opt.down_ratio
        for i in range(1):
            debugger = Debugger(
                dataset=opt.dataset, ipynb=(opt.debug == 3), theme=opt.debugger_theme)
            img = batch['input'][i].detach().cpu().numpy().transpose(1, 2, 0)
            img = np.clip(((
                                   img * opt.std + opt.mean) * 255.), 0, 255).astype(np.uint8)
            pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
            gt = debugger.gen_colormap(batch['hm'][i].detach().cpu().numpy())
            debugger.add_blend_img(img, pred, 'pred_hm')
            debugger.add_blend_img(img, gt, 'gt_hm')
            debugger.add_img(img, img_id='out_pred')
            for k in range(len(dets[i])):
                if dets[i, k, 4] > opt.center_thresh:
                    debugger.add_coco_bbox(dets[i, k, :4], dets[i, k, -1],
                                           dets[i, k, 4], img_id='out_pred')

            debugger.add_img(img, img_id='out_gt')
            for k in range(len(dets_gt[i])):
                if dets_gt[i, k, 4] > opt.center_thresh:
                    debugger.add_coco_bbox(dets_gt[i, k, :4], dets_gt[i, k, -1],
                                           dets_gt[i, k, 4], img_id='out_gt')

            if opt.debug == 4:
                debugger.save_all_imgs(opt.debug_dir, prefix='{}'.format(iter_id))
            else:
                debugger.show_all_imgs(pause=True)

    def save_result(self, output, batch, results):
        reg = output['reg'] if self.opt.reg_offset else None
        dets = ctdet_decode(
            output['hm'], output['wh'], reg=reg,
            cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets_out = ctdet_post_process(
            dets.copy(), batch['meta']['c'].cpu().numpy(),
            batch['meta']['s'].cpu().numpy(),
            output['hm'].shape[2], output['hm'].shape[3], output['hm'].shape[1])
        results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]
