import os.path as osp
import os
import argparse
import pprint
import shutil

import torch
from torch.utils.data import DataLoader
from torch import nn
import numpy as np

import __init__path
from net.resnet import resnet
from utils.config import cfg, cfg_from_file, cfg_from_list
from utils.tools import adjust_learning_rate
from data_layer.roidb import combined_roidb
from data_layer.coco_loader import CocoDataset

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
                      default=100, type=int)
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
        fasterRCNN = resnet(imdb.classes, 101, pretrained=True)

    ### Use tensorboardX
    if args.use_tfboard:
        from tensorboardX import SummaryWriter
        logger = SummaryWriter(args.log_dir)
        shutil.rmtree(args.log_dir)
        os.mkdir(args.log_dir)

    lr = cfg.TRAIN.LEARNING_RATE
    params = []

    ### Debug


    from roibatchLoader import roibatchLoader
    # def rank_roidb_ratio(roidb):
    #     # rank roidb based on the ratio between width and height.
    #     ratio_large = 2  # largest ratio to preserve.
    #     ratio_small = 0.5  # smallest ratio to preserve.
    #
    #     ratio_list = []
    #     for i in range(len(roidb)):
    #         width = roidb[i]['width']
    #         height = roidb[i]['height']
    #         ratio = width / float(height)
    #
    #         if ratio > ratio_large:
    #             roidb[i]['need_crop'] = 1
    #             ratio = ratio_large
    #         elif ratio < ratio_small:
    #             roidb[i]['need_crop'] = 1
    #             ratio = ratio_small
    #         else:
    #             roidb[i]['need_crop'] = 0
    #
    #         ratio_list.append(ratio)
    #     ratio_list = np.array(ratio_list)
    #     ratio_index = np.argsort(ratio_list)
    #     return ratio_list[ratio_index], ratio_index
    #
    # ratio_list, ratio_index = rank_roidb_ratio(roidb)
    #
    # dataset = roibatchLoader(roidb, ratio_list, ratio_index, 1, imdb.num_classes, training=True)


    ### 123

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    iters_per_epoch = int(len(roidb) / 1)

    fasterRCNN.create_architecture()
    fasterRCNN.train()

    if cfg.CUDA:
        fasterRCNN.cuda()

    for key, value in dict(fasterRCNN.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), 'weight_decay': 0}]
            else:
                params += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
    optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

    loss_temp = 0
    for epoch in range(1, cfg.TRAIN.MAX_EPOCHS + 1):

        if epoch % (cfg.TRAIN.LR_DECAY_EPOCH + 1) == 0:
            adjust_learning_rate(optimizer)
        data_iter = dataloader.__iter__()

        for step in range(iters_per_epoch):
            image, info, gt_boxes = data_iter.next()
            if cfg.CUDA:
                image = image.cuda()
                info = info.cuda()
                gt_boxes = gt_boxes.cuda()

            # image.data.resize_(image.size()).copy_(image)
            # info.data.resize_(info.size()).copy_(info)
            # gt_boxes.data.resize_(info.size()).copy_(gt_boxes)
            rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            rois_label, pp, np, cls_p, cls_n = fasterRCNN(image, info, gt_boxes)

            loss = rpn_loss_cls.mean() + rpn_loss_box.mean() + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
            # loss = rpn_loss_cls.mean() + rpn_loss_box.mean()
            loss_temp += loss.item()

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % args.disp_interval == 0:
                # loss_temp /= args.disp_interval
                loss_temp = loss.mean().item()
                loss_rpn_cls = rpn_loss_cls.mean().item()
                loss_rpn_box = rpn_loss_box.mean().item()
                loss_rcnn_cls = RCNN_loss_cls.mean().item()
                loss_rcnn_box = RCNN_loss_bbox.mean().item()
                fg_cnt = torch.sum(rois_label.data.ne(0))
                bg_cnt = rois_label.data.numel() - fg_cnt

                print("[epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
                      % (epoch, step, iters_per_epoch, loss_temp, lr))
                print("rcnn_cls: %.4f, rcnn_box %.4f, rpn_cls: %.4f, rpn_box: %.4f, pp: %.4f, np: %.4f, cls_p : %.4f, cls_n: %.4f" \
                      % (loss_rcnn_cls, loss_rcnn_box, loss_rpn_cls, loss_rpn_box, pp, np, cls_p, cls_n))

                # print("[epoch %2d][iter %4d/%4d] rpn_cls: %.4f, rpn_box: %.4f, positive: %.4f   negative: %.4f, fg/bg=(%d/%d)" \
                #       % (epoch, step, iters_per_epoch, loss_rpn_cls, loss_rcnn_box, pp, np, fg_cnt, bg_cnt))

                if args.use_tfboard:
                    info = {
                        'loss': loss,
                        'loss_rpn_cls': loss_rpn_cls,
                        'loss_rpn_box': loss_rpn_box,
                        'loss_rcnn_cls': loss_rcnn_cls,
                        'loss_rcnn_box': loss_rcnn_box
                    }
                    logger.add_scalars("loss", info, (epoch - 1) * iters_per_epoch + step)

                loss_temp = 0

        save_path = osp.join(args.save_dir, 'fasterRCNN_{}_{}.pth'.format(epoch, step))
        torch.save({'model': fasterRCNN.state_dict(),
                    'epoch': epoch,
                    'step' : step,
                    'optimizer': optimizer.state_dict()
                    }, save_path)
        print("save model: {}".format(save_path))

    if args.use_tfboard:
        logger.close()