from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch.utils.data as data
from .wider2019pd_preload import TrainingSet, ValidationSet, TestSet, remove_ignored_det


class WIDER2019(data.Dataset):
  default_resolution = [512, 512]
  mean = np.array([0.4102250051589073, 0.41284485423898254, 0.40470640518037193],
                  dtype=np.float32) # shape 1×1×3
  std = np.array([0.3106054621264184, 0.30618797320431057, 0.30613979423706167],
                 dtype=np.float32)
  _eig_val = np.array([0.27909881412713583, 0.003764150758693364, 0.0010854367671571876],
                      dtype=np.float32)
  _eig_vec = np.array([[0.5814744492684015, 0.5773210261771315, 0.5732258696027186],
                       [0.7204828581950962, -0.03815557189484072, -0.6924222724468088],
                       [-0.3778781759972839, 0.8156252724549191, -0.4381363931902992]],
                      dtype=np.float32)
  class_name = ['__background__', 'person']
  num_classes = 1
  _valid_ids = [1]
  cat_ids = {v: i for i, v in enumerate(_valid_ids)}
  voc_color = [(64, 0, 32)]
  max_objs = 128
  _data_rng = np.random.RandomState(123)

  def __init__(self, opt, split):
    super(WIDER2019, self).__init__()
    self.data_dir = opt.data_dir
    self.split = split
    self.opt = opt
    print(f'==> initializing WIDER 2019 Pedestrian Detection {split} data.')
    split_map = {'train':TrainingSet, 'val':ValidationSet, 'test':TestSet}
    self.w2019pd = split_map[split](self.data_dir)
    self.images = self.w2019pd.image_ids
    print('Loaded {} {} samples'.format(split, len(self.w2019pd)))

  def __len__(self):
    return len(self.w2019pd)

  def run_eval(self, results, save_dir):
    # result_json = osp.join(save_dir, "results.json")
    # detections  = self.convert_eval_format(results)
    # json.dump(detections, open(result_json, "w"))
    save_results(results, save_dir)  # submission

    for ii in self.w2019pd.image_ids:
      ibox = self.w2019pd.ignore_box.get(ii)
      if ibox is not None:
        results[ii] = remove_ignored_det(results[ii], ibox)

    mAP = pedestrian_eval(results, self.w2019pd.gt_map)
    print('mAP of the submission is', mAP)


def save_results(detection, save_path):

  def f3(x): return float("{:.3f}".format(x))
  def f1(x): return float("{:.1f}".format(x))

  with open(save_path, 'w') as fw:
    for image_id, boxes in detection.items():
      for bbox in boxes:
        bbox[2] -= bbox[0]
        bbox[3] -= bbox[1]
        score = bbox[4]
        print(image_id, f3(score), *map(f1, bbox[:4]), file=fw)


def pedestrian_eval(dts, gt):
  """
  :param dts: detection dict {img_id:bbox[[x,y,w,h],...]}
  :param gt: ground truth, same format as above
  :return: mAP ref COCO
  """

  def compute_ap(rec, prec):
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))
    for i in range(mpre.size - 1, 0, -1):
      mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

  aap = []
  nd = len(dts)
  ovethr = np.arange(0.5, 1.0, 0.05)

  npos = 0    # number of positives among the dataset
  _det = {}
  for image_id in list(gt.keys()):
    im_pos = len(gt[image_id])
    npos += im_pos
    _det[image_id] = [False] * im_pos  # assume there's no box detected

  for ove in ovethr:
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    det = _det.copy()  # whether a box in an image is detected or not

    for i, (image_id, bb) in enumerate(dts.items()):
      BBGT = np.array(gt[image_id])
      bb = np.array(bb).T
      iou_max = -np.inf
      if BBGT.size > 0:
        ixmin = np.maximum(BBGT[:, 0], bb[0])
        iymin = np.maximum(BBGT[:, 1], bb[1])
        ixmax = np.minimum(BBGT[:, 2], bb[2])
        iymax = np.minimum(BBGT[:, 3], bb[3])
        iw = np.maximum(ixmax - ixmin + 1., 0.)
        ih = np.maximum(iymax - iymin + 1., 0.)
        inters = iw * ih
        bba = (bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.)
        BBa = (BBGT[:, 2] - BBGT[:, 0] + 1.) * (BBGT[:, 3] - BBGT[:, 1] + 1.)
        uni = bba + BBa - inters
        iou = inters / uni
        iou_max = np.max(iou)
        iou_argmax = np.argmax(iou)
      if iou_max > ove:
        if not det[image_id][iou_argmax]:
          tp[i] = 1.
          det[image_id][iou_argmax] = True
        else:
          fp[i] = 1.
      else:
        fp[i] = 1.
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = compute_ap(rec, prec)
    aap.append(ap)
  mAP = np.mean(aap)
  return mAP