from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
import torch
import cv2
from torchvision.ops import roi_align, roi_pool, RoIAlign

from external.nms import soft_nms
from models.decode import ctdet_decode
from models.utils import flip_tensor
from utils.post_process import ctdet_post_process
from .base_detector import BaseDetector

from importlib.machinery import SourceFileLoader

DHN = SourceFileLoader('dhn', '/home/wanghao/github/deepmot/models/DHN.py').load_module()
tracker = SourceFileLoader('tracker', '/home/wanghao/github/deepmot/tracker.py').load_module()

KCF = SourceFileLoader('kcf', '/home/wanghao/github/KCFpy/kcftracker.py').load_module()


def roi(fea_map, box):
    return


class PersonDetector(BaseDetector):
    INTEVAL = 3

    def __init__(self, opt):
        super(PersonDetector, self).__init__(opt)
        self.frame_counter = 0
        self.feature_map = None

        def get(module, input, output):
            self.feature_map = input[0]  # input is tuple, dono why

        self.model.hm.register_forward_hook(get)
        mod = DHN.Munkrs(1, 256, 1, True, 1, False, False)
        weights = torch.load('/home/wanghao/github/deepmot/model_weights/DHN.pth')
        mod.load_state_dict(weights)
        self.tracker = tracker.Tracker(mod)
        self.video_out = cv2.VideoWriter('centernet.mp4', cv2.VideoWriter_fourcc('M','J','P','G'), 25, (960,540))
        self.roi = RoIAlign([1,1], opt.down_ratio, 6)
        self.kcfs = None

    def run(self, image_or_path_or_tensor, meta=None):
        img = cv2.imread(image_or_path_or_tensor)
        if self.frame_counter % self.INTEVAL:
            tic = time.time()
            boxes = []
            for kcf,c in self.kcfs:
                l,t,w,h = kcf.update(img)
                boxes += [(l, t, l+w, t+h, c)]
            track_time = time() - tic
            ret = {'results': {1:np.array(boxes)}, 'tot': track_time, 'load': 0,
                  'pre': 0, 'net': 0, 'dec': track_time,
                  'post': 0, 'merge': 0}
        else:
            ret = super().run(img, meta)
            self.kcfs = []
            for bbox in ret['results'][1]:
                if bbox[4] > self.opt.vis_thresh:
                    kcf = KCF.KCFTracker(False, True, True)
                    l,t,r,b,c = bbox
                    kcf.init([int(l), int(t), int(r-l), int(b-t)], img)
                    self.kcfs += [(kcf,c)]
        self.frame_counter += 1
        return ret
        # if self.frame_counter % self.INTEVAL:
        #     return super().run(image_or_path_or_tensor, meta)
        #     # pass    # todo fill transmitting frames using tracker
        #     # return {'results': results, 'tot': tot_time, 'load': load_time,
        #     #       'pre': pre_time, 'net': net_time, 'dec': dec_time,
        #     #       'post': post_time, 'merge': merge_time}
        # else:
        #     return super().run(image_or_path_or_tensor, meta)

    def process(self, images, return_time=False):
        with torch.no_grad():
            output = self.model(images)[-1]
            hm = output['hm'].sigmoid_()
            wh = output['wh']
            reg = output['reg'] if self.opt.reg_offset else None
            if self.opt.flip_test:
                hm = (hm[0:1] + flip_tensor(hm[1:2])) / 2
                wh = (wh[0:1] + flip_tensor(wh[1:2])) / 2
                reg = reg[0:1] if reg is not None else None
            torch.cuda.synchronize()
            forward_time = time.time()
            dets = ctdet_decode(hm, wh, reg=reg, K=self.opt.K)

        if return_time:
            return output, dets, forward_time
        else:
            return output, dets

    def post_process(self, dets, meta, scale=1):
        dets = dets.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])
        # breakpoint()
        dets = ctdet_post_process(
            dets.copy(), [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'], self.opt.num_classes)
        for j in range(1, self.num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
            dets[0][j][:, :4] /= scale
        return dets[0]

    def merge_outputs(self, detections):
        results = {}
        for j in range(1, self.num_classes + 1):
            results[j] = np.concatenate(
                [detection[j] for detection in detections], axis=0).astype(np.float32)
            if len(self.scales) > 1 or self.opt.nms:
                soft_nms(results[j], Nt=0.5, method=2)
        scores = np.hstack(
            [results[j][:, 4] for j in range(1, self.num_classes + 1)])
        if len(scores) > self.max_per_image:
            kth = len(scores) - self.max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.num_classes + 1):
                keep_inds = (results[j][:, 4] >= thresh)
                results[j] = results[j][keep_inds]
        return results

    def debug(self, debugger, images, dets, output, scale=1):
        detection = dets.detach().cpu().numpy().copy()
        detection[:, :, :4] *= self.opt.down_ratio
        for i in range(1):
            img = images[i].detach().cpu().numpy().transpose(1, 2, 0)
            img = ((img * self.std + self.mean) * 255).astype(np.uint8)
            pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
            debugger.add_blend_img(img, pred, 'pred_hm_{:.1f}'.format(scale))
            debugger.add_img(img, img_id='out_pred_{:.1f}'.format(scale))
            for k in range(len(dets[i])):
                if detection[i, k, 4] > self.opt.center_thresh:
                    debugger.add_coco_bbox(detection[i, k, :4], detection[i, k, -1],
                                           detection[i, k, 4],
                                           img_id='out_pred_{:.1f}'.format(scale))

    def show_results(self, debugger, image, results):
        debugger.add_img(image, img_id='ctdet')
        boxes = []
        for bbox in results[1]:
            if bbox[4] > self.opt.vis_thresh:
                boxes += [torch.FloatTensor(bbox[:5])]
        boxes = torch.stack(boxes)
        # print(self.feature_map.device, boxes.cuda().device)
        # features = roi_align(self.feature_map.cpu(), [boxes], (9, 3), self.opt.down_ratio)        # todo: use fea
        # features = self.roi(self.feature_map.cpu(), [boxes])        # todo: use fea
        # breakpoint()
        h, w = image.shape[:2]
        diag = (h**2 + w**2)**.5
        self.tracker.update(boxes/diag)
        for id, track in self.tracker.tracks.items():
            if not track.is_candidate and track.age > 10:
                # print(track.box[4])
                debugger.add_coco_bbox(track.box[:4]*diag, id%80, track.box[4]*diag, img_id='ctdet', box_name=f'#{id}|')
        debugger.show_all_imgs(pause=self.pause)
        self.video_out.write(cv2.resize(debugger.imgs['ctdet'], (960, 540)))
