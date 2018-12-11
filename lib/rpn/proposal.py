from __future__ import absolute_import

import torch
import torch.nn as nn
import numpy as np

from utils.config import cfg
from rpn.generate_anchors import generate_anchors


class Proposal(nn.Module):
    """
    Get region proposals from base model outputs
    """
    def __init__(self, feat_stride, scales, ratios):
        super(Proposal, self).__init__()
        self._feat_stride = feat_stride
        self._anchor = generate_anchors(scales=np.array(scales), ratios=np.array(ratios))
        self._num_anchors = self._anchor.shape[0]

