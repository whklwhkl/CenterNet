from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import cv2
from progress.bar import Bar
import torch

from external.nms import soft_nms
from opts import opts
from logger import Logger
from utils.utils import AverageMeter
from datasets.dataset_factory import dataset_factory
from detectors.detector_factory import detector_factory
from termcolor import colored
import pickle


class PrefetchDataset(torch.utils.data.Dataset):
  def __init__(self, opt, dataset, pre_process_func):
    self.dataset = dataset
    self.images = dataset.images
    # print(colored(dataset.data_dir, 'red'))
    self.pre_process_func = pre_process_func
    self.opt = opt
  
  def __getitem__(self, index):
    img_id = self.dataset.images[index]
    img_path = self.dataset.w2019pd[index][0]
    image = cv2.imread(img_path)
    images, meta = {}, {}
    for scale in opt.test_scales:
      images[scale], meta[scale] = self.pre_process_func(image, scale)
    return img_id, {'images': images, 'image': image, 'meta': meta}

  def __len__(self):
    return len(self.images)

def test(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

  Dataset = dataset_factory[opt.dataset]
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)
  Logger(opt)
  Detector = detector_factory[opt.task]
  
  split = 'val' if not opt.trainval else 'test'
  dataset = Dataset(opt, split)
  # if os.path.exists('results.pkl'):
  #   dataset.run_eval(pickle.load(open('results.pkl', 'rb')), opt.save_dir)
  #   return
  detector = Detector(opt)
  
  data_loader = torch.utils.data.DataLoader(
    PrefetchDataset(opt, dataset, detector.pre_process), 
    batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

  results = {}
  num_iters = len(dataset)
  bar = Bar('{}'.format(opt.exp_id), max=num_iters)
  time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
  avg_time_stats = {t: AverageMeter() for t in time_stats}
  for ind, (img_id, pre_processed_images) in enumerate(data_loader):
    ret = detector.run(pre_processed_images)
    # breakpoint()
    results[img_id[0]] = ret['results']
    Bar.suffix = '[{0}/{1}]|Tot:{total:}|ETA:{eta:}'.format(
                   ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
    for t in avg_time_stats:
      avg_time_stats[t].update(ret[t])
      Bar.suffix = Bar.suffix + '|{}{tm.val:.3f}s({tm.avg:.2f}s)'.format(
        t, tm = avg_time_stats[t])
    bar.next()

  pickle.dump(results, open('results.pkl', 'wb'))
  bar.finish()
  dataset.run_eval(results, opt.save_dir)


if __name__ == '__main__':
  opt = opts().parse()
  test(opt)