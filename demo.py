import argparse
import pprint
import time
import numpy as np
import cv2
import os
from scipy.misc import imread

import torch
from torch.utils.data import DataLoader

import __init__path
from net.resnet import resnet
from utils.config import cfg, cfg_from_file, cfg_from_list
from data_layer.roidb import combined_roidb
from data_layer.coco_loader import CocoDataset
from utils.bbox_transform import bbox_transform_inv, clip_boxes
from utils.nms_wrapper import nms
from utils.blob import im_list_to_blob


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a deeplab network')
    parser.add_argument('--image', dest='image',
                      help='testing image path',
                      default='COCO', type=str)
    parser.add_argument('--net', dest='net',
                    help='vgg16, res101',
                    default='res101', type=str)
    parser.add_argument('--models', dest='models',
                      help='directory to load models', default="omodels/fasterRCNN_1_234531.pth",
                      type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    return args

def get_image(im):
    im_orig = im.astype(np.float32, copy=True)
    im = im_orig - cfg.PIXEL_MEANS

    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)
    return blob, np.array(im_scale_factors)


if __name__ == "__main__":

    args = parse_args()
    print('Called with args: ')
    print(args)

    args.cfg_file = "cfgs/{}.yml".format(args.net)

    args.cfg_file = "cfgs/{}.yml".format(args.net)
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)

    if torch.cuda.is_available() and not cfg.CUDA:
        print("Warning: You have a CUDA device, so you should run on it")

    imdbval_name = "coco_2014_minival"
    imdb, roidb = combined_roidb(imdbval_name, False)
    if args.net == 'res101':
        fasterRCNN = resnet(imdb.classes, 101, pretrained=True)
    fasterRCNN.create_architecture()
    checkpoint = torch.load(args.models)
    fasterRCNN.load_state_dict(checkpoint['model'])
    print("Load model from %s" % (args.models))
    if cfg.CUDA:
        fasterRCNN.cuda()

    fasterRCNN.eval()
    max_per_image = 100
    thresh = 0.05
    # vis = True

    imglist = os.listdir(args.image)
    num_images = len(imglist)
    print('Loaded Photo: {} images.'.format(num_images))

    while (num_images >= 0):
        num_images = num_images - 1
        im_file = os.path.join(args.image, imglist[num_images])
        im_in = np.array(imread(im_file))

        # if gray image
        if len(im_in.shape) == 2:
            im_in = im_in[:, :, np.newaxis]
            im_in = np.concatenate((im_in, im_in, im_in), axis=2)

        # rgb->bgr
        im = im_in[:, :, ::-1]
        blobs, im_scales = get_image(im)
        im_blob = blobs
        im_info = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

        image = torch.from_numpy(im_blob)
        image = image.permute(0, 3, 1, 2)
        im_info = torch.from_numpy(im_info)
        gt_boxes = torch.FloatTensor([1,1,1,1,1])

        if cfg.CUDA:
            image = image.cuda()
            info = info.cuda()
            gt_boxes = gt_boxes.cuda()

        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label = fasterRCNN(image, info, gt_boxes)

        scores = cls_prob.data
        boxes = rois.data[:, 1:5]
        box_deltas = bbox_pred.data
        if cfg.TRAIN.CLASS_AGNOSTIC:
            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                         + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
            box_deltas = box_deltas.view(-1, 4)
        else:
            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                         + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
            box_deltas = box_deltas.view(-1, 4 * len(imdb.classes))

        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes = clip_boxes(pred_boxes, info)
        pred_boxes /= im_scales[0]

        for j in range(1, imdb.num_classes):
            inds = torch.nonzero(scores[:, j] > thresh).view(-1)
            if inds.numel() > 0:
                cls_scores = scores[:, j][inds]
                _, order = torch.sort(cls_scores, 0, True)
                if cfg.TRAIN.CLASS_AGNOSTIC:
                    cls_boxes = pred_boxes[inds, :]
                else:
                    cls_boxes = pred_boxes[inds][:, j*4:(j+1)*4]

                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                cls_dets = cls_dets[order]
                keep = nms(cls_dets, cfg.TEST.NMS)
                cls_dets = cls_dets[keep.view(-1).long()]



