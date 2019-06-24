import torch
import numpy as np

from models.decode import ctdet_decode
from models.utils import _sigmoid, _tranpose_and_gather_feat
from utils.debugger import Debugger
from utils.post_process import ctdet_post_process
from .base_trainer import BaseTrainer

from importlib.machinery import SourceFileLoader

loader = SourceFileLoader('hao_iou', '/home/wanghao/PycharmProjects/reid/CenterNet/src/lib/trains/hau_iou.py')
hau_iou = loader.load_module()


class Loss(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.hau = hau_iou.WeightedHausdorffDistance([x//opt.down_ratio for x in (opt.input_h, opt.input_w)])
        self.iou = hau_iou.bounded_iou_loss
        self.opt = opt

    def forward(self, outputs, batch):
        opt = self.opt
        hm_loss, iou_loss = [torch.autograd.Variable(torch.tensor(0.)).cuda() for _ in [1,1]]
        hm_loss.require_grad = True
        iou_loss.require_grad = True
        for s in range(opt.num_stacks):
            output = outputs[s]
            output['hm'] = _sigmoid(output['hm'])
            WH = _tranpose_and_gather_feat(output['wh'], batch['ind'])
            REG = _tranpose_and_gather_feat(output['reg'], batch['ind'])
            for hm, ct_int, mask, reg, wh, reg_, wh_ in zip(output['hm'], batch['ctr'], batch['reg_mask'],
                                                            batch['reg'], batch['wh'],
                                                            REG, WH):
                if mask.sum():
                    iou_loss += self.iou(reg[mask], wh[mask], reg_[mask], wh_[mask])
                    ct_int = ct_int[mask]
                    xs = ct_int[:, 0]
                    ys = ct_int[:, 1]
                    hm_loss += self.hau(hm[0], torch.stack([ys, xs], -1))
            # breakpoint()
        hm_loss /= opt.batch_size * opt.num_stacks
        iou_loss /= opt.batch_size * opt.num_stacks
        loss = opt.hm_weight * hm_loss + opt.wh_weight * iou_loss
        loss_stats = {'loss': loss, 'hm_loss': hm_loss, 'iou_loss': iou_loss}
        return loss, loss_stats


class Trainer(BaseTrainer):
    def __init__(self, opt, model, optimizer=None):
        super().__init__(opt, model, optimizer=optimizer)

    def _get_losses(self, opt):
        loss_states = ['loss', 'hm_loss', 'iou_loss']
        loss = Loss(opt)
        return loss_states, loss

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
            debugger = Debugger(dataset=opt.dataset, ipynb=(opt.debug == 3), theme=opt.debugger_theme)
            img = batch['input'][i].detach().cpu().numpy().transpose(1, 2, 0)
            img = np.clip(((img * opt.std + opt.mean) * 255.), 0, 255).astype(np.uint8)
            pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
            debugger.add_blend_img(img, pred, 'pred_hm')
            debugger.add_img(img, img_id='out_pred')
            for k in range(len(dets[i])):
                if dets[i, k, 4] > opt.center_thresh:
                    debugger.add_coco_bbox(dets[i, k, :4], dets[i, k, -1], dets[i, k, 4], img_id='out_pred')

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
