from __future__ import absolute_import
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------
# --------------------------------------------------------
# Modify based on the version in https://github.com/jwyang/faster-rcnn.pytorch
# --------------------------------------------------------
import torch
import torch.nn as nn
import numpy as np

from utils.config import cfg
from rpn.generate_anchors import generate_anchors
from rpn.bbox_transform import bbox_transform_inv, clip_boxes


class Proposal(nn.Module):
    """
    Get region proposals from base model outputs
    """
    def __init__(self, feat_stride, scales, ratios):
        super(Proposal, self).__init__()
        self._feat_stride = feat_stride
        self._anchor = generate_anchors(scales=np.array(scales), ratios=np.array(ratios))
        self._num_anchors = self._anchor.shape[0]

    def forward(self, scores, bbox_delta, im_info, cfg_key):
        fg_scores = scores[:, self._num_anchors:, :, :]

        pre_nms_topN  = cfg[cfg_key].RPN_PRE_NMS_TOP_N
        post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
        nms_thresh    = cfg[cfg_key].RPN_NMS_THRESH

        batch_size = bbox_delta.size(0)
        assert (batch_size == 1) # Only support batch size = 1

        # Get the full anchor
        feat_height, feat_width = scores.size(2), scores.size(3)
        shift_x = np.arange(0, feat_width) * self._feat_stride
        shift_y = np.arange(0, feat_height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()
        A = self._num_anchors
        K = shifts.shape[0]
        anchors = self._anchor.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
        anchors = anchors.reshape((K * A, 4)).astype(np.float32, copy=False)

        # Convert the anchor into proposal
        bbox_delta = bbox_delta.permute(0, 2, 3, 1).contiguous()
        bbox_delta = bbox_delta.view(-1, 4)
        proposals = bbox_transform_inv(torch.from_numpy(anchors), bbox_delta)
        proposals = clip_boxes(proposals, im_info)

        # choose the proposals
        scores = scores.permute(0, 2, 3, 1).contiguous()
        scores = scores.view(batch_size, -1)

