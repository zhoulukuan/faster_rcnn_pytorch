import torch
import torch.nn as nn

class AnchorTarget(nn.Module):
    """
        Assign anchors to ground-truth targets. Produces anchor classification
        labels and bounding-box regression targets.
    """
    def __init__(self, allow_border=0):
        super(AnchorTarget, self).__init__()


        self._allow_border = allow_border


    def forward(self, rpn_cls_score, gt_boxes, im_info, all_anchors):

        total_anchors = all_anchors.shape[0]
        keep = ((all_anchors[:, 0] >= -self._allowed_border) &
                (all_anchors[:, 1] >= -self._allowed_border) &
                (all_anchors[:, 2] < int(im_info[0][1]) + self._allowed_border) &
                (all_anchors[:, 3] < int(im_info[0][0]) + self._allowed_border))

        id_insides = torch.nonzero(keep).view(-1)

        a = 1