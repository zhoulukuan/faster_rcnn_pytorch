import torch
import numpy as np

def bbox_overlaps(anchors, gt_boxes):
    """
    anchors: (N, 4) ndarray of float
    gt_boxes: (K, 4) ndarray of float

    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    Written by jwyang (see in https://github.com/jwyang/faster-rcnn.pytorch)
    """
    N = anchors.size(0)
    K = gt_boxes.size(0)

    gt_boxes_x = gt_boxes[:,2] - gt_boxes[:,0] + 1
    gt_boxes_y = gt_boxes[:,3] - gt_boxes[:,1] + 1
    gt_boxes_area = (gt_boxes_x * gt_boxes_y).view(1, K)

    anchors_boxes_x = anchors[:,2] - anchors[:,0] + 1
    anchors_boxes_y = anchors[:,3] - anchors[:,1] + 1
    anchors_area = (anchors_boxes_x * anchors_boxes_y).view(N, 1)

    boxes = anchors.view(N, 1, 4).expand(N, K, 4)
    query_boxes = gt_boxes.view(1, K, 4).expand(N, K, 4)

    iw = (torch.min(boxes[:,:,2], query_boxes[:,:,2]) -
        torch.max(boxes[:,:,0], query_boxes[:,:,0]) + 1)
    iw[iw < 0] = 0

    ih = (torch.min(boxes[:,:,3], query_boxes[:,:,3]) -
        torch.max(boxes[:,:,1], query_boxes[:,:,1]) + 1)
    ih[ih < 0] = 0

    ua = anchors_area + gt_boxes_area - (iw * ih)
    overlaps = iw * ih / ua

    gt_area_zero = (gt_boxes_x == 1) & (gt_boxes_y == 1)
    anchors_area_zero = (anchors_boxes_x == 1) & (anchors_boxes_y == 1)
    overlaps.masked_fill_(gt_area_zero.view(1, K).expand(N, K), 0)
    overlaps.masked_fill_(anchors_area_zero.view(N, 1).expand(N, K), -1)

    return overlaps