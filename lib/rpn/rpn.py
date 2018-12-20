import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.config import cfg
from rpn.create_proposal import CreateProposal
from rpn.anchor_target import AnchorTarget
from utils.tools import _smooth_l1_loss

class RPN(nn.Module):
    """Region Proposal Network"""
    def __init__(self, dim):
        super(RPN, self).__init__()

        self.dim = dim
        self.anchor_scales = cfg.ANCHOR_SCALES
        self.anchor_ratios = cfg.ANCHOR_RATIOS
        self.feat_stride = cfg.FEAT_STRIDE

        self.RPN_conv = nn.Conv2d(self.dim, 512, 3, 1, 1, bias=True)

        # define bg/fg classification score layer
        self.rpn_score_out_dim = len(self.anchor_scales) * len(self.anchor_ratios) * 2
        self.RPN_cls_score = nn.Conv2d(512, self.rpn_score_out_dim, 1, 1, 0)

        # define anchor box offset prediction layer
        self.rpn_bbox_out_dim = len(self.anchor_scales) * len(self.anchor_ratios) * 4
        self.RPN_bbox_pred = nn.Conv2d(512, self.rpn_bbox_out_dim, 1, 1, 0)

        # define proposal layer
        self.RPN_proposal = CreateProposal(self.feat_stride, self.anchor_scales, self.anchor_ratios)

        # define anchor target layers
        self.Anchor_target = AnchorTarget()

        # define loss
        self.rpn_loss_box = 0
        self.rpn_loss_cls = 0

    @staticmethod
    def reshape(x, d):
        input_shape = x.size()
        x = x.view(input_shape[0],
                   int(d),
                   int(float(input_shape[1] * input_shape[2]) / float(d)),
                   input_shape[3])
        return x

    def forward(self, base_feat, im_info, gt_boxes):
        batch_size = base_feat.size(0)

        rpn_conv1 = F.relu(self.RPN_conv(base_feat), inplace=True)

        # Get class score
        rpn_cls_score = self.RPN_cls_score(rpn_conv1)
        rpn_cls_score_reshape = self.reshape(rpn_cls_score, 2)
        rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape, 1)
        rpn_cls_prob = self.reshape(rpn_cls_prob_reshape, self.rpn_score_out_dim)

        # Get rpn offsets
        rpn_bbox_pred = self.RPN_bbox_pred(rpn_conv1)

        # Get region of interest
        cfg_key = 'TRAIN' if self.training else 'TEST'
        rois, anchors = self.RPN_proposal(rpn_cls_prob.data, rpn_bbox_pred.data, im_info, cfg_key)

        # Compute rpn loss
        self.rpn_loss_cls, self.rpn_loss_box = 0, 0
        if self.training:
            assert gt_boxes is not None
            rpn_data = self.Anchor_target(rpn_cls_score.data, gt_boxes, im_info, anchors)

            # compute classification loss
            rpn_cls_score = rpn_cls_score_reshape.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)
            rpn_label = rpn_data[0].view(batch_size, -1)

            rpn_keep = rpn_label.view(-1).ne(-1).nonzero().view(-1)
            rpn_cls_score = torch.index_select(rpn_cls_score.view(-1, 2), 0, rpn_keep)
            rpn_label = torch.index_select(rpn_label.view(-1), 0, rpn_keep).long()
            self.rpn_loss_cls = F.cross_entropy(rpn_cls_score, rpn_label)

            # compute bbox regression loss
            rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = rpn_data[1:]
            self.rpn_loss_box = _smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets,
                                                rpn_bbox_inside_weights, rpn_bbox_outside_weights)

        return rois, self.rpn_loss_cls, self.rpn_loss_box












