import torch
import torch.nn as nn
import torch.nn.functional as F

from rpn.rpn import  RPN

class FasterRCNN(nn.Module):
    """Faster RCNN base class"""

    def __init__(self, classes, class_agnostic=False):
        super(FasterRCNN, self).__init__()
        self.classes = classes
        self.num_classes = len(classes)

        self.RCNN_rpn = RPN(self.base_model_out_dim)

    def forward(self, image, im_info, gt_boxes):
        batch_size = image.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        base_feat = self.RCNN_base(image)

        ## feed base feat to get RPN
        rois, rpn_cls, rpn_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes)


        ## feed images to rcnn_base to obtain base feature for rpn



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