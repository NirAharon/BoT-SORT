# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import os.path as osp
import re
import warnings

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class MOT20(ImageDataset):
    """MOT20.

    Reference:
        Dendorfer, P., Rezatofighi, H., Milan, A., Shi, J., Cremers, D., Reid, I., Roth, S., Schindler, K. & Leal-Taixé, L. MOT20: A benchmark for multi object tracking in crowded scenes. arXiv:2003.09003[cs], 2020., (arXiv: 2003.09003).

    URL: `<https://motchallenge.net/data/MOT20/>`_

    Dataset statistics:
        - identities: ?
        - images: ?
    """
    _junk_pids = [0, -1]
    dataset_dir = 'MOT20'
    dataset_url = ''  # 'https://motchallenge.net/data/MOT20.zip'
    dataset_name = "MOT20"

    def __init__(self, root='datasets', **kwargs):
        # self.root = osp.abspath(osp.expanduser(root))
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # allow alternative directory structure
        self.data_dir = self.dataset_dir
        data_dir = osp.join(self.data_dir, 'MOT20-ReID')
        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn('The current data structure is deprecated. Please '
                          'put data folders such as "bounding_box_train" under '
                          '"MOT20-ReID".')

        self.train_dir = osp.join(self.data_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.data_dir, 'query')
        self.gallery_dir = osp.join(self.data_dir, 'bounding_box_test')
        self.extra_gallery_dir = osp.join(self.data_dir, 'images')
        self.extra_gallery = False

        required_files = [
            self.data_dir,
            self.train_dir,
            # self.query_dir,
            # self.gallery_dir,
        ]

        self.check_before_run(required_files)

        train = lambda: self.process_dir(self.train_dir)
        query = lambda: self.process_dir(self.query_dir, is_train=False)
        gallery = lambda: self.process_dir(self.gallery_dir, is_train=False) + \
                          (self.process_dir(self.extra_gallery_dir, is_train=False) if self.extra_gallery else [])

        super(MOT20, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, is_train=True):

        img_paths = glob.glob(osp.join(dir_path, '*.bmp'))
        pattern = re.compile(r'([-\d]+)_MOT20-0(\d)')

        data = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            # assert 0 <= pid   # pid == 0 means background
            # assert 1 <= camid <= 5
            camid -= 1  # index starts from 0
            if is_train:
                pid = self.dataset_name + "_" + str(pid)
                camid = self.dataset_name + "_" + str(camid)
            data.append((img_path, pid, camid))

        return data
