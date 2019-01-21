import os.path as osp
import argparse
import pprint
import time
import numpy as np
import shutil
import os
import sys

import torch
from torch.utils.data import DataLoader

import __init__path
from net.resnet import resnet
from utils.config import cfg, cfg_from_file, cfg_from_list
from data_layer.roidb import combined_roidb
from data_layer.coco_loader import CocoDataset
from utils.bbox_transform import bbox_transform_inv, clip_boxes
from utils.nms_wrapper import nms


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a deeplab network')
    parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='COCO', type=str)
    parser.add_argument('--net', dest='net',
                    help='vgg16, res101',
                    default='res101', type=str)
    parser.add_argument('--models', dest='models',
                      help='directory to load models', default="omodels/fasterRCNN_9_234531.pth",
                      type=str)
    parser.add_argument('--log', dest='output_dir',
                        help='directory to save detections log', default="detections/",
                        type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    return args

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

    if args.dataset == 'COCO':
        args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
        args.imdbval_name = "coco_2014_minival"

    start = time.time()

    cfg.TRAIN.USE_FLIPPED = False
    imdb, roidb = combined_roidb(args.imdbval_name, False)
    imdb.competition_mode(on=True)

    if args.net == 'res101':
        fasterRCNN = resnet(imdb.classes, 101, pretrained=True)
    fasterRCNN.create_architecture()
    checkpoint = torch.load(args.models)
    fasterRCNN.load_state_dict(checkpoint['model'])
    print("Load model from %s" % (args.models))
    if cfg.CUDA:
        fasterRCNN.cuda()

    dataset = CocoDataset(roidb, imdb.num_classes, training=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    data_iter = dataloader.__iter__()

    fasterRCNN.eval()
    num_images = len(imdb.image_index)
    max_per_image = 100
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(imdb.num_classes)]
    empty_array = np.transpose(np.array([[], [], [], [], []]), (1, 0))

    for i in range(num_images):
        image, info, gt_boxes = data_iter.next()
        if cfg.CUDA:
            image = image.cuda()
            info = info.cuda()
            gt_boxes = gt_boxes.cuda()

        det_tic = time.time()
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
        pred_boxes /= info[0][2].item()
        det_toc = time.time()
        detect_time = det_toc - det_tic
        misc_tic = time.time()
        for j in range(1, imdb.num_classes):
            inds = torch.nonzero(scores[:, j] > 0).view(-1)
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
                all_boxes[j][i] = cls_dets.cpu().numpy()
            else:
                all_boxes[j][i] = empty_array

        # Limit each images detection numbers over all classes
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                      for j in range(1, imdb.num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in range(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]

        misc_toc = time.time()
        nms_time = misc_toc - misc_tic
        sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
                         .format(i + 1, num_images, detect_time, nms_time))
        sys.stdout.flush()

    print("Evaluating detections")
    imdb.evaluate_detections(all_boxes, args.output_dir)

    end = time.time()
    print("test time: %0.4fs" % (end - start))