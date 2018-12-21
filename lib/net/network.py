import torch
import torch.nn as nn
import torch.nn.functional as F

from rpn.rpn import RPN
from rpn.proposal_target import ProposalTarget
from utils.config import cfg
from pooling.roi_pooling.modules.roi_pool import _RoIPooling
from pooling.roi_align.modules.roi_align import RoIAlignAvg
from utils.tools import _smooth_l1_loss

class FasterRCNN(nn.Module):
    """Faster RCNN base class"""

    def __init__(self, classes):
        super(FasterRCNN, self).__init__()
        self.classes = classes
        self.num_classes = len(classes)

        self.RCNN_rpn = RPN(self.base_model_out_dim)
        self.RCNN_proposal_target = ProposalTarget(self.num_classes)

        # Pooling method
        self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)

    def forward(self, image, im_info, gt_boxes):
        batch_size = image.size(0)

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

        # Roi Pooling
        if cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1, 5))

        # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)

        # compute bbox
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        if self.training:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score, 1)

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        if self.training:
            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)

            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_bbox_target, rois_inside_ws, rois_outside_ws)

        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label


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