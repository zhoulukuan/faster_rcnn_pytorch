import numpy as np

import torch
import torch.nn as nn

from utils.bbox import bbox_overlaps
from rpn.bbox_transform import bbox_transform
from utils.config import cfg
class AnchorTarget(nn.Module):
    """
        Assign anchors to ground-truth targets. Produces anchor classification
        labels and bounding-box regression targets.
    """
    def __init__(self, allowed_border=0):
        super(AnchorTarget, self).__init__()


        self._allowed_border = allowed_border


    def forward(self, rpn_cls_score, gt_boxes, im_info, anchors):
        gt_boxes = gt_boxes[0, :, :4]
        # Convert to tensor for convenience of follow-up operation
        all_anchors = torch.from_numpy(anchors)

        total_anchors = all_anchors.size(0)
        keep = ((all_anchors[:, 0] >= -self._allowed_border) &
                (all_anchors[:, 1] >= -self._allowed_border) &
                (all_anchors[:, 2] < int(im_info[0][1]) + self._allowed_border) &
                (all_anchors[:, 3] < int(im_info[0][0]) + self._allowed_border))

        id_insides = torch.nonzero(keep).view(-1)
        num_insides = id_insides.size(0)

        # keep only inside anchors
        anchors = all_anchors[keep, :]

        # allocate labels: 1 for positive, 0 for negative, -1 for not care
        labels = gt_boxes.new(num_insides).fill_(-1)
        bbox_inside_weights = gt_boxes.new(num_insides).zero_()
        bbox_outside_weights = gt_boxes.new(num_insides).zero_()

        overlaps = bbox_overlaps(anchors, gt_boxes)

        anchor_max_overlaps, anchor_achieve_max = torch.max(overlaps, 1)
        gt_max_ove, gt_achieve_max = torch.max(overlaps, 0)

        # allocate negative labels
        labels[anchor_max_overlaps <= cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

        # allocate positive labels
        labels[anchor_max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1
        labels[gt_achieve_max] = 1
        # keep = torch.sum(overlaps.eq(gt_max_overlaps.view(1,-1).expand_as(overlaps)), 1)
        # labels[keep > 0] = 1

        num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)
        fg_inds = torch.nonzero(labels == 1).view(-1)
        sum_fg = fg_inds.size(0)
        if sum_fg > num_fg:
            rand_num = torch.randperm(sum_fg).type_as(gt_boxes).long()
            disable_inds = fg_inds[rand_num[:sum_fg-num_fg]]
            labels[disable_inds] = -1

        num_bg = cfg.TRAIN.RPN_BATCHSIZE - min(num_fg, sum_fg)
        bg_inds = torch.nonzero(labels == 0).view(-1)
        sum_bg = bg_inds.size(0)
        if sum_bg > num_fg:
            rand_num = torch.randperm(sum_bg).type_as(gt_boxes).long()
            disable_inds = bg_inds[rand_num[:sum_bg-num_bg]]
            labels[disable_inds] = -1

        bbox_target = bbox_transform(anchors, gt_boxes[anchor_achieve_max])

        # bbox_inside_weights: gain for diff(pred_boxes - gt_boxes)
        # use single number 1 to represent unchanged, change for 4 different numbers if needed
        # bbox_outside_weight: coefficient for positive and negative examples
        bbox_inside_weights = gt_boxes.new(id_insides.size(0), 4).zero_()
        bbox_inside_weights[labels == 1, :] = 1
        bbox_outside_weights = gt_boxes.new(id_insides.size(0), 4).zero_()
        positive_weight = negative_weight = 1.0 / cfg.TRAIN.RPN_BATCHSIZE
        bbox_outside_weights[labels == 1, :] = positive_weight
        bbox_outside_weights[labels == 0, :] = negative_weight

        labels = _unmap(labels, total_anchors, id_insides, fill=-1)
        bbox_target = _unmap(bbox_target, total_anchors, id_insides, fill=0)
        bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, id_insides, fill=0)
        bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, id_insides, fill=0)

        height, width = rpn_cls_score.size(2), rpn_cls_score.size(3)
        A = len(cfg.ANCHOR_SCALES) * len(cfg.ANCHOR_RATIOS)
        batch_size = 1

        labels = labels.view(batch_size, height, width, A).permute(0, 3, 1, 2).contiguous()
        labels = labels.view(batch_size, 1, A*height, width)

        bbox_target = bbox_target.view(batch_size, height, width, A*4).permute(0, 3, 1, 2).contiguous()
        bbox_inside_weights = bbox_inside_weights.view(batch_size, height, width, A*4).permute(0, 3, 1, 2).contiguous()
        bbox_outside_weights = bbox_outside_weights.view(batch_size, height, width, A*4).permute(0, 3, 1, 2).contiguous()

        return [labels, bbox_target, bbox_inside_weights, bbox_outside_weights]


def _unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if data.dim() == 1:
        ret = torch.Tensor(count).fill_(fill).type_as(data)
        ret[inds] = data
    else:
        ret = torch.Tensor(count, data.size(1)).fill_(fill).type_as(data)
        ret[inds, :] = data
    return ret
