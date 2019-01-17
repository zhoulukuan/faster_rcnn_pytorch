import os.path as osp
import random
import numpy as np
import math
import random

import torch
from torch.utils.data import Dataset
from data_layer.minibatch import get_minibatch

class CocoDataset(Dataset):
    def __init__(self, roidb, num_classes, training=False):
        self.roidb = roidb
        self.num_classes = num_classes
        self.training = training

    def __len__(self):
        return len(self.roidb)

    def __getitem__(self, item):
        minibatch_db = [self.roidb[item]]
        blobs = get_minibatch(minibatch_db, self.num_classes)
        data = blobs['data']
        im_info = blobs['im_info']
        if self.training:
            gt_boxes = blobs['gt_boxes']
        else:
            gt_boxes = torch.FloatTensor([0, 0, 0, 0, 0])

        return data[0].transpose(2, 0, 1), im_info.reshape([3]), gt_boxes