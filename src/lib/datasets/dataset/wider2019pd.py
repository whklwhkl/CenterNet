from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch.utils.data as data
from .wider2019pd_preload import TrainingSet, ValidationSet, TestSet
import os.path as osp

import imp
wider_tools_spec = imp.find_module('tools', ['/home/wanghao/datasets/WIDER_pd2019/'])
imp.load_module('tools', *wider_tools_spec)
from tools import evaluate

from termcolor import colored


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
    save_results(results, save_dir)  # submission
    mAP = evaluate.get_score(osp.join(save_dir, 'submission.txt'))
    print(colored(f'mAP of the submission is {mAP}', 'green'))
    return mAP


def save_results(detection, save_dir):
  """no side effect"""
  def f3(x): return float("{:.3f}".format(x))
  def f1(x): return float("{:.1f}".format(x))
  path = osp.join(save_dir, 'submission.txt')
  path = osp.abspath(path)
  print(colored(path, 'magenta'))
  with open(path, 'w') as fw:
    for image_id, boxes in detection.items():
      for bbox in boxes[1]:
        l = bbox[0]
        t = bbox[1]
        w = max(bbox[2] - bbox[0], .1)
        h = max(bbox[3] - bbox[1], .1)
        score = bbox[4]
        print(image_id, f3(score), *map(f1, [l,t,w,h]), file=fw)