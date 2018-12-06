import os.path as osp
import os
import argparse
import pprint
from pycocotools.coco import COCO

import torch
from torch.utils.data import DataLoader
from torch import nn

import __init__path
from utils.config import cfg, cfg_from_file, cfg_from_list
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.coco_loader import CocoDataset

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
    parser.add_argument('--disp_interval', dest='disp_interval',
                      help='number of iterations to display',
                      default=20, type=int)
    parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                      help='number of save model and evaluate it',
                      default=1000, type=int)
    parser.add_argument('--save_dir', dest='save_dir',
                      help='directory to save models', default="models",
                      type=str)
    parser.add_argument('--resume', dest='resume',
                      help='If resume training', default=False,
                      type=bool)
    parser.add_argument('--log_dir', dest='log_dir',
                      help='directory to save logs', default='logs',
                      type=str)
    parser.add_argument('--use_tfboard', dest='use_tfboard',
                      help='if use tensorboardX', default=True,
                      type=bool)
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
        imdb, roidb = combined_roidb(args.imdb_name)
        dataset = CocoDataset(roidb, imdb.num_classes, training=True)

    if args.net == 'res101':
        fasterRCNN = resnet(imdb.num_classes, 101, pretrained=True)

    """
    Only support batch size = 1
    """
    dataloader = DataLoader(dataset, batch_size=1)
    data_iter = dataloader.__iter__()

    image, info, gt_boxes = data_iter.next()

    a = 1

