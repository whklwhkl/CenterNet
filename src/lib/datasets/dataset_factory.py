from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# from .sample.ddd import DddDataset
# from .sample.exdet import EXDetDataset
# from .sample.ctdet import CTDetDataset
# from .sample.multi_pose import MultiPoseDataset
from .sample.pdet import PedestrianDet
from .sample.hausdorff import Hausdorff_BoundedIOU

# from .dataset.coco import COCO
# from .dataset.pascal import PascalVOC
# from .dataset.kitti import KITTI
# from .dataset.coco_hp import COCOHP
from .dataset.wider2019pd import WIDER2019


dataset_factory = {
  # 'coco': COCO,
  # 'pascal': PascalVOC,
  # 'kitti': KITTI,
  # 'coco_hp': COCOHP,
  'wider': WIDER2019
}

_sample_factory = {
  # 'exdet': EXDetDataset,
  # 'ctdet': CTDetDataset,
  # 'ddd': DddDataset,
  # 'multi_pose': MultiPoseDataset,
  'pdet': PedestrianDet,
  'pdet_logwh': PedestrianDet,
  'haus': Hausdorff_BoundedIOU
}


def get_dataset(dataset, task):
  class Dataset(dataset_factory[dataset], _sample_factory[task]):
    pass
  return Dataset
  
