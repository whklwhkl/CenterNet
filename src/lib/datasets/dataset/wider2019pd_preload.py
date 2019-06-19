import os.path as osp
from copy import deepcopy


SPLIT = ['train', 'val', 'test']
TYPE = ['list', 'ignore', 'bbox']
CATEGORIES = ['pedestrian', 'ignore']


class _Dataset:
    def __init__(self, data_dir=''):
        self.anno_pattern = osp.join(data_dir, 'Annotations', '{split}_{type}.txt')
        self.image_paths = []
        self.ignore_box = []
        self.image_ids = []
        self.gt_map = None      # ground truth
        self.boxes = []
        self.length = 0
        self.ignore_map = None

    def __getitem__(self, index):
        return self.image_paths[index], self.boxes[index]#, self.ignore_box[index]

    def __len__(self):
        return self.length

    def load(self, load_all=False):
        """read in and convert data in dict to list,
        so that any column with the same index belongs to the same object
        """
        self.gt_map = self._read_txt(TYPE[2])[1]
        self.ignore_map = self._read_txt(TYPE[1])[1]
        path_map =self._read_txt(TYPE[0])[0]
        # print(len(path_map))
        for ii in path_map.keys():
            if ii == 'sur03954.jpg': continue  # Premature end of JPEG
            box = self.gt_map.get(ii) or []
            ignore_box = self.ignore_map.get(ii) or []
            if load_all:  # special settings for validation sets
                self.image_paths += [path_map[ii]]
                self.ignore_box += [ignore_box]
                self.image_ids += [ii]
                ltwh2ltrb(box)
                if len(ignore_box):
                    ltwh2ltrb(ignore_box)  # ltrb
                    box = remove_ignored_det(box, ignore_box)
                self.gt_map[ii] = deepcopy(box)           # ltrb format, used when eval
                self.boxes += [ltrb2ltwh(box)]  # coco format
            elif len(box):
                self.image_ids += [ii]
                if len(box):
                    self.image_paths += [path_map[ii]]
                    self.ignore_box += [ignore_box]
                    self.image_ids += [ii]
                    self.boxes += [box]
                    self.gt_map[ii] = box
        self.length = len(self.boxes)

    def _read_txt(self, type):
        """
        :param type: 'list', 'bbox' or 'ignore'
        :return: a path map of each image name, and a box map with the same image names
        """
        image_paths = []
        image_names = []
        bboxes = []
        txt_path = self.anno_pattern.format(**{'type': type})
        with open(txt_path) as f:
            for line in f:
                tokens = line.strip().split()
                name = tokens[0]
                image_paths += [self._extend_path(name)]
                image_names += [name]
                bbox = []
                for i in range(len(tokens) // 4):
                    start = i * 4 + 1
                    bbox += [list(float(x) for x in tokens[start:start + 4])]
                bboxes += [bbox]
        path_map = dict(zip(image_names, image_paths))
        bbox_map = dict(zip(image_names, bboxes))
        return path_map, bbox_map

    def _extend_path(self, image_name):
        raise NotImplementedError('define name extending logic')


class TrainingSet(_Dataset):
    def __init__(self, data_dir=''):
        super().__init__(data_dir)
        self.anno_pattern = self.anno_pattern.format(split=SPLIT[0], type='{type}')
        self.sur = osp.join(data_dir, 'sur_train', '{}')
        self.ad = osp.join(data_dir, 'ad_train', 'ad_0{}', '{}')
        self.load()

    def _extend_path(self, image_name):
        if image_name.startswith('sur'):
            return self.sur.format(image_name)
        else:
            for i in range(1, 4):
                guess_path = self.ad.format(i, image_name)
                if osp.exists(guess_path):
                    return guess_path
        raise KeyError('cannot locate image')


class ValidationSet(_Dataset):
    def __init__(self, data_dir=''):
        super().__init__(data_dir)
        self.anno_pattern = self.anno_pattern.format(split=SPLIT[1], type='{type}')
        self.val = osp.join(data_dir, 'val_data', '{}')
        self.load(load_all=True)

    def _extend_path(self, image_name):
        return self.val.format(image_name)


class TestSet(_Dataset):
    def __init__(self, data_dir=''):
        super().__init__(data_dir)
        self.load()

    def load(self):
        self.image_paths = list(self._read_txt(TYPE[0]).keys())

    def __getitem__(self, item):
        return self.image_paths[item]

    def _extend_path(self, image_name):
        raise NotImplementedError


def remove_ignored_det(dt_box, ig_box):
    """ltrb!!!!format"""
    remain_box = []
    for p in dt_box:
        if len(p)>4:
            _,pl,pt,pr,pb = p
        else:
            pl,pt,pr,pb = p
        p_area = float((pr-pl)*(pb-pt)) + 1e-5  # numerical stability
        overlap = -0.01
        for c in ig_box:
            cl,ct,cr,cb = c
            if (cr>pl) and (cl<pr) and (ct<pb) and (cb>pt):
                overlap += (min(cr,pr)-max(cl,pl)+1.0)*(min(cb,pb)-max(ct,pt)+1.0)
        if overlap/p_area <= 0.5:
            remain_box.append(p)
    return remain_box


def ltrb2ltwh(boxes):
    for i in range(len(boxes)):
        boxes[i][2] -= boxes[i][0]
        boxes[i][3] -= boxes[i][1]
    return boxes


def ltwh2ltrb(boxes):
    for i in range(len(boxes)):
        boxes[i][2] += boxes[i][0]
        boxes[i][3] += boxes[i][1]
    return boxes


if __name__ == '__main__':
    val = ValidationSet('/home/wanghao/datasets/WIDER_pd2019')
    for i in range(len(val)):
        x = val[i]