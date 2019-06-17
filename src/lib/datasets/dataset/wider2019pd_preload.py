import os.path as osp


SPLIT = ['train', 'val', 'test']
TYPE = ['list', 'ignore', 'bbox']
CATEGORIES = ['pedestrian', 'ignore']


class _Dataset:
    def __init__(self, data_dir=''):
        self.anno_pattern = osp.join(data_dir, 'Annotations', '{split}_{type}.txt')
        self.image_paths = None
        self.ignore_box = None
        self.image_ids = None
        self.gt_map = None      # ground truth
        self.length = 0
        self.box = None

    def __getitem__(self, index):
        return self.image_paths[index], self.box[index]#, self.ignore_box[index]

    def __len__(self):
        return self.length

    def load(self):
        """read in and convert data in dict to list,
        so that any column with the same index belongs to the same object
        """
        path_map =self._read_txt(TYPE[0])[0]
        self.image_ids = list(path_map.keys())
        self.image_paths = [path_map[k] for k in self.image_ids]
        ignore = self._read_txt(TYPE[1])[1]
        self.gt_map = self._read_txt(TYPE[2])[1]
        boxes, ignore_boxes = [], []
        for ii in self.image_ids:
            box = self.gt_map.get(ii)
            if box is not None:
                boxes += [self.gt_map.get(ii)]
                ignore_box = ignore.get(ii)
                if ignore_box is not None:
                    self.gt_map[ii] = remove_ignored_det(box, ignore_box)
            else:
                boxes += [[]]
        self.box = boxes
        self.length = len(self.image_paths)

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
        self.load()

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
    remain_box = []
    for p in dt_box:
        if len(p)>4:
            _,pl,pt,pr,pb = p
        else:
            pl,pt,pr,pb = p
        p_area = float((pr-pl)*(pb-pt))
        overlap = -0.01
        for c in ig_box:
            cl,ct,cr,cb = c
            if (cr>pl) and (cl<pr) and (ct<pb) and (cb>pt):
                overlap += (min(cr,pr)-max(cl,pl)+1.0)*(min(cb,pb)-max(ct,pt)+1.0)
        if overlap/p_area <= 0.5:
            remain_box.append(p)
    return remain_box