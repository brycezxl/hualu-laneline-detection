# coding: utf8
# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

# GPU memory garbage collection optimization flags
os.environ['FLAGS_eager_delete_tensor_gb'] = "0.0"
os.environ['CUDA_VISIBLE_DEVICE'] = "2"
os.chdir("../../")

import sys
import argparse
import pprint
import cv2
import numpy as np
import paddle.fluid as fluid

from PIL import Image as PILImage
from utils.config import cfg
from reader import SegDataset
from models.model_builder import build_model
from models.model_builder import ModelPhase
from tools.gray2pseudo_color import get_color_map_list
from loss import multi_softmax_with_loss


def parse_args():
    parser = argparse.ArgumentParser(description='PaddeSeg visualization tools')
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        help='Config file for training (and optionally testing)',
        default="./configs/sf-hr18-city-1.yaml",
        type=str)
    parser.add_argument(
        '--use_gpu', dest='use_gpu', help='Use gpu or cpu', action='store_true')
    parser.add_argument(
        '--vis_dir',
        dest='vis_dir',
        help='visual save dir',
        type=str,
        default='visual')
    parser.add_argument(
        '--local_test',
        dest='local_test',
        help='if in local test mode, only visualize 5 images for testing',
        action='store_true')
    parser.add_argument(
        'opts',
        help='See config.py for all options',
        default=None,
        nargs=argparse.REMAINDER)
    # if len(sys.argv) == 1:
    #     parser.print_help()
    #     sys.exit(1)
    return parser.parse_args()


def makedirs(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def to_png_fn(fn):
    """
    Append png as filename postfix
    """
    directory, filename = os.path.split(fn)
    basename, ext = os.path.splitext(filename)

    return basename + ".png"


def join(png1, png2, flag='horizontal'):
    """
    :param png1: path
    :param png2: path
    :param flag: horizontal or vertical
    :return:
    """
    img1, img2 = png1, png2
    size1, size2 = img1.size, img2.size
    if flag == 'horizontal':
        joint = PILImage.new('RGB', (size1[0]+size2[0], size1[1]))
        loc1, loc2 = (0, 0), (size1[0], 0)
        joint.paste(img1, loc1)
        joint.paste(img2, loc2)
    else:
        joint = PILImage.new('RGB', (size1[0], size1[1]+size2[1]))
        loc1, loc2 = (0, 0), (0, size1[1])
        joint.paste(img1, loc1)
        joint.paste(img2, loc2)
    return joint


def cross_entropy(pred, label):
    from sklearn.metrics import log_loss
    pred = np.array(pred).squeeze(0)
    label = label.squeeze(0)
    loss = 0
    labels = [i for i in range(pred.shape[0])]
    for i in range(pred.shape[1]):
        p_ = np.resize(pred[:, :, i], (pred.shape[1], pred.shape[0]))
        l_ = label[:, i]
        loss += log_loss(l_, p_, labels=labels)
    return loss / pred.shape[1]


def visualize(cfg,
              vis_file_list=None,
              use_gpu=False,
              vis_dir="show",
              ckpt_dir=None,
              log_writer=None,
              local_test=False,
              **kwargs):
    for i in range(20):
        # Generator full colormap for maximum 256 classes
        color_map = get_color_map_list(256)

        res_map = np.ones((100, 100)) * i
        pred_mask = PILImage.fromarray(res_map.astype(np.uint8), mode='L')
        pred_mask.putpalette(color_map)
        # pred_mask.save(vis_fn)

        pred_mask_np = np.array(pred_mask.convert("RGB"))
        im_pred = PILImage.fromarray(pred_mask_np)
        im_pred.save("/home/zxl/" + str(i) + ".png")


if __name__ == '__main__':
    args = parse_args()
    if args.cfg_file is not None:
        cfg.update_from_file(args.cfg_file)
    if args.opts:
        cfg.update_from_list(args.opts)
    cfg.check_and_infer()
    print(pprint.pformat(cfg))
    visualize(cfg, **args.__dict__)
