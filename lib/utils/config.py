from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import numpy as np
from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C

# Data directory
__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
__C.DATA_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'data'))

# Pixel mean values (BGR order) as a (1, 1, 3) array
# We use the same pixel mean for all networks even though it's not exactly what
# they were trained with
__C.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])

#
# Training options
#
__C.TRAIN = edict()

# Pretrained model
__C.TRAIN.PRETRAINED_MODEL = osp.abspath(osp.join(__C.DATA_DIR, 'resnet101_caffe.pth'))

# Some parametres about training
# Initial learning rate
__C.TRAIN.LEARNING_RATE = 0.001
# Momentum
__C.TRAIN.MOMENTUM = 0.9
# Weight decay, for regularization
__C.TRAIN.WEIGHT_DECAY = 0.0005
# If double learning rate of bias
__C.TRAIN.DOUBLE_BIAS = True
# Max iters
__C.TRAIN.MAX_EPOCHS = 10
# Decay learning rate after some epochs
__C.TRAIN.LR_DECAY_EPOCH = 4
# Images in a batch
__C.TRAIN.IMS_PER_BATCH = 1



# Settint about dataset
# If use flipped image
__C.TRAIN.USE_FLIPPED = True
# Train using these proposals
__C.TRAIN.PROPOSAL_METHOD = 'gt'
# Whether to use all ground truth bounding boxes for training,
__C.TRAIN.USE_ALL_GT = True
# Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)
__C.TRAIN.FG_THRESH = 0.5
# Overlap threshold for a ROI to be considered background (class = 0 if
# overlap in [LO, HI))
__C.TRAIN.BG_THRESH_HI = 0.5
__C.TRAIN.BG_THRESH_LO = 0
# Scale to use during training (can list multiple scales)
# The scale is the pixel size of an image's shortest side
__C.TRAIN.SCALES = (600,)
# Max pixel size of the longest side of a scaled input image
__C.TRAIN.MAX_SIZE = 1000


# Minibatch size (number of regions of interest [ROIs])
__C.TRAIN.BATCH_SIZE = 128
# Fraction of minibatch that is labeled foreground (i.e. class > 0)
__C.TRAIN.FG_FRACTION = 0.25


# If class agnostic during rpn stage
__C.TRAIN.CLASS_AGNOSTIC = False
__C.TRAIN.BBOX_NORMALIZE_MEANS = (0.0, 0.0, 0.0, 0.0)
__C.TRAIN.BBOX_NORMALIZE_STDS = (0.1, 0.1, 0.2, 0.2)

# Feature stride for RPN
__C.FEAT_STRIDE = 16


# Setting for anchors and rpn
# Anchor scales for RPN
__C.ANCHOR_SCALES = [8,16,32]
# Anchor ratios for RPN
__C.ANCHOR_RATIOS = [0.5,1,2]
# NMS threshold used on RPN proposals
__C.TRAIN.RPN_NMS_THRESH = 0.7
# Number of top scoring boxes to keep before apply NMS to RPN proposals
__C.TRAIN.RPN_PRE_NMS_TOP_N = 12000
# Number of top scoring boxes to keep after applying NMS to RPN proposals
__C.TRAIN.RPN_POST_NMS_TOP_N = 2000
# IOU >= thresh: positive example
__C.TRAIN.RPN_POSITIVE_OVERLAP = 0.7
# IOU < thresh: negative example
__C.TRAIN.RPN_NEGATIVE_OVERLAP = 0.3
# Max number of foreground examples
__C.TRAIN.RPN_FG_FRACTION = 0.5
# Total number of examples
__C.TRAIN.RPN_BATCHSIZE = 256


# Pooling method and size
__C.POOLING_MODE = 'pool'
__C.POOLING_SIZE = 7



# Number of fixed blocks during training, by default the first of all 4 blocks is fixed
# Range: 0 (none) to 3 (all)
__C.RESNET = edict()
__C.RESNET.FIXED_BLOCKS = 1


# Testing options
#
__C.TEST = edict()

# Scale to use during testing (can NOT list multiple scales)
# The scale is the pixel size of an image's shortest side
__C.TEST.SCALES = (600,)

# Max pixel size of the longest side of a scaled input image
__C.TEST.MAX_SIZE = 1000

# Overlap threshold used for non-maximum suppression (suppress boxes with
# IoU >= this threshold)
__C.TEST.NMS = 0.3

## NMS threshold used on RPN proposals
__C.TEST.RPN_NMS_THRESH = 0.7

## Number of top scoring boxes to keep before apply NMS to RPN proposals
__C.TEST.RPN_PRE_NMS_TOP_N = 6000

## Number of top scoring boxes to keep after applying NMS to RPN proposals
__C.TEST.RPN_POST_NMS_TOP_N = 300

# If use GPU for training
__C.CUDA = True



### Debug


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print(('Error under config key: {}'.format(k)))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)


def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert subkey in d
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), \
            'type {} does not match original type {}'.format(
                type(value), type(d[subkey]))
        d[subkey] = value