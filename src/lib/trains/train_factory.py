from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .ctdet import CtdetTrainer
from .ddd import DddTrainer
from .exdet import ExdetTrainer
from .multi_pose import MultiPoseTrainer
from .pdet import PdetTrainer
from .no_gt_hm import Trainer
from .logwh import Trainer2

train_factory = {
  'exdet': ExdetTrainer, 
  'ddd': DddTrainer,
  'ctdet': CtdetTrainer,
  'multi_pose': MultiPoseTrainer,
  'pdet': PdetTrainer,
  'pdet_logwh': Trainer2,
  'haus': Trainer
}
