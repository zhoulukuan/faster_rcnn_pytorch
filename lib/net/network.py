import torch
import torch.nn as nn
import torch.nn.functional as F

from rpn.rpn import RPN
from rpn.proposal_target import ProposalTarget
from utils.config import cfg

class FasterRCNN(nn.Module):
    """Faster RCNN base class"""

    def __init__(self, classes, class_agnostic=False):
        super(FasterRCNN, self).__init__()
        self.classes = classes
        self.num_classes = len(classes)

        self.RCNN_rpn = RPN(self.base_model_out_dim)
        self.RCNN_proposal_target = ProposalTarget(self.num_classes)

    def forward(self, image, im_info, gt_boxes):
        batch_size = image.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        ## feed images to rcnn_base to obtain base feature for rpn
        base_feat = self.RCNN_base(image)

        # feed base feat to get RPN
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes)

        # if in training mode, use predicted rois for refine box
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes)
            rois, rois_label, rois_bbox_target, rois_inside_ws, rois_outside_ws = roi_data

            rois_label = rois_label.view(-1).long()
            rois_bbox_target = rois_bbox_target.view(-1, rois_bbox_target.size(1))
            rois_inside_ws = rois_inside_ws.view(-1, rois_inside_ws.size(1))
            rois_outside_ws = rois_outside_ws.view(-1, rois_outside_ws.size(1))
        else:
            rois_label = None
            rois_bbox_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        # Roi Align

        return None


    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

            normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
            normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
            normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
            normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
            normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()