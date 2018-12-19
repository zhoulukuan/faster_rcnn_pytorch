import torch
import torch.nn as nn
import numpy as np

from utils.config import cfg
from utils.bbox import bbox_overlaps
from rpn.bbox_transform import bbox_transform, bbox_transform_inv

class ProposalTarget(nn.Module):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """
    def __init__(self, num_classes):
        super(ProposalTarget, self).__init__()
        self._num_classes = num_classes
        self.BBOX_NORMALIZE_MEANS = (0.0, 0.0, 0.0, 0.0)
        self.BBOX_NORMALIZE_STDS = (0.1, 0.1, 0.2, 0.2)
        self.BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)

    def forward(self, all_rois, gt_boxes):
        batch_size = 1

        num_images = 1
        rois_per_image = int(cfg.TRAIN.BATCH_SIZE / num_images)
        fg_rois_per_image = int(np.round(cfg.TRAIN.FG_FRACTION * rois_per_image))
        fg_rois_per_image = 1 if fg_rois_per_image == 0 else fg_rois_per_image

        overlaps = bbox_overlaps(all_rois, gt_boxes[:, :4])
        max_overlaps, rois_achieve_max = torch.max(overlaps, 1)

        labels = gt_boxes[:, 4].contiguous().view(-1)
        labels_new = labels.new(rois_per_image).zero_()
        rois_new = all_rois.new(rois_per_image, 4).zero_()
        gt_rois = all_rois.new(rois_per_image, 5).zero_()

        fg_inds = torch.nonzero(max_overlaps >= cfg.TRAIN.FG_THRESH).view(-1)
        fg_num_rois = fg_inds.numel()
        bg_inds = torch.nonzero((max_overlaps < cfg.TRAIN.BG_THRESH_HI) & (max_overlaps >= cfg.TRAIN.BG_THRESH_LO)).view(-1)
        bg_num_rois = bg_inds.numel()

        if fg_num_rois > 0 and bg_num_rois > 0:
            # sampling fg
            fg_num = min(fg_rois_per_image, fg_num_rois)
            rand_num = torch.randperm(fg_num_rois).type_as(gt_boxes).long()
            fg_inds = fg_inds[rand_num[:fg_num]]

            # sampling bg
            bg_num = rois_per_image - fg_num
            rand_num = np.floor(np.random.rand(bg_num) * bg_num_rois)
            rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()
            bg_inds = bg_inds[rand_num]

        elif fg_num_rois > 0 and bg_num_rois == 0:
            # sample fg
            rand_num = np.floor(np.random.rand(rois_per_image) * fg_num_rois)
            rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()
            fg_inds = fg_inds[rand_num]
            fg_num = rois_per_image
            bg_num = 0

        elif bg_num_rois > 0 and fg_num_rois == 0:
            # sample bg
            rand_num = np.floor(np.random.rand(rois_per_image) * bg_num_rois)
            rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()
            bg_inds = bg_inds[rand_num]
            fg_num = 0
            bg_num = rois_per_image

        else:
            raise ValueError("bg_num_rois = 0 and fg_num_rois = 0, this should not happen!")

        keep_inds = torch.cat([fg_inds, bg_inds], 0)
        labels_new.copy_(labels[keep_inds])

        # Clamp labels
        if fg_num < rois_per_image:
            labels_new[fg_num:] = 0

        rois_new = all_rois[keep_inds]
        gt_rois = gt_boxes[rois_per_image[keep_inds]]

        bbox_target_data = self._compute_target(rois_new, gt_rois[:, :4])
        bbox_targets, bbox_inside_weights = self._get_bbox_regression_labels(bbox_target_data, labels_new)

        return labels_new, rois_new, bbox_targets, bbox_inside_weights

    def _compute_target(self, ex_rois, gt_rois):
        """Compute bounding-box regression targets for an image."""
        assert ex_rois.size(1) == gt_rois.size(1)
        assert ex_rois.size(2) == 4
        assert gt_rois.size(2) == 4

        targets = bbox_transform(ex_rois, gt_rois)
        targets = ((targets - self.BBOX_NORMALIZE_MEANS.expand_as(targets))
                   / self.BBOX_NORMALIZE_STDS.expand_as(targets))

        return targets


    def _get_bbox_regression_labels(self, bbox_target_data, labels):
        rois_per_image = labels.size(0)
        clss = labels
        bbox_targets = bbox_target_data.new(rois_per_image, 4).zero_()
        bbox_inside_weights = bbox_target_data.new(bbox_targets.size()).zero_()

        if clss.sum() > 0:
            inds = torch.nonzero(clss > 0).view(-1)
            for i in range(inds.numel()):
                ind = inds[i]
                bbox_targets[ind, :] = bbox_target_data[ind, :]
                bbox_inside_weights[ind, :] = self.BBOX_INSIDE_WEIGHTS
        return bbox_targets, bbox_inside_weights



