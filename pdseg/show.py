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
os.chdir("../")

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


def visualize(cfg,
              vis_file_list=None,
              use_gpu=False,
              vis_dir="show",
              ckpt_dir=None,
              log_writer=None,
              local_test=False,
              **kwargs):
    if vis_file_list is None:
        vis_file_list = cfg.DATASET.VAL_FILE_LIST
    dataset = SegDataset(
        file_list=vis_file_list,
        mode=ModelPhase.VISUAL,
        data_dir=cfg.DATASET.DATA_DIR)

    startup_prog = fluid.Program()
    test_prog = fluid.Program()
    pred, logit = build_model(test_prog, startup_prog, phase=ModelPhase.VISUAL)
    # Clone forward graph
    test_prog = test_prog.clone(for_test=True)

    # Generator full colormap for maximum 256 classes
    color_map = get_color_map_list(256)

    # Get device environment
    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup_prog)

    ckpt_dir = cfg.TEST.TEST_MODEL if not ckpt_dir else ckpt_dir

    if ckpt_dir is not None:
        print('load test model:', ckpt_dir)
        try:
            fluid.load(test_prog, os.path.join(ckpt_dir, 'model'), exe)
        except:
            fluid.io.load_params(exe, ckpt_dir, main_program=test_prog)

    save_dir = "show"
    makedirs(save_dir)

    fetch_list = [pred.name]
    test_reader = dataset.batch(dataset.generator, batch_size=1, is_test=True)
    img_cnt = 0
    for imgs, grts, img_names, valid_shapes, org_shapes in test_reader:
        pred_shape = (imgs.shape[2], imgs.shape[3])
        pred, = exe.run(
            program=test_prog,
            feed={'image': imgs},
            fetch_list=fetch_list,
            return_numpy=True)

        num_imgs = pred.shape[0]
        # TODO: use multi-thread to write images
        for i in range(num_imgs):
            # Add more comments
            res_map = np.squeeze(pred[i, :, :, :]).astype(np.uint8)
            img_name = img_names[i]
            res_shape = (res_map.shape[0], res_map.shape[1])
            if res_shape[0] != pred_shape[0] or res_shape[1] != pred_shape[1]:
                res_map = cv2.resize(
                    res_map, pred_shape, interpolation=cv2.INTER_NEAREST)
            valid_shape = (valid_shapes[i, 0], valid_shapes[i, 1])
            res_map = res_map[0:valid_shape[0], 0:valid_shape[1]]
            org_shape = (org_shapes[i, 0], org_shapes[i, 1])
            res_map = cv2.resize(
                res_map, (org_shape[1], org_shape[0]),
                interpolation=cv2.INTER_NEAREST)

            png_fn = to_png_fn(img_name)
            # colorful segment result visualization
            vis_fn = os.path.join(save_dir, png_fn)
            dirname = os.path.dirname(vis_fn)
            makedirs(dirname)

            pred_mask = PILImage.fromarray(res_map.astype(np.uint8), mode='L')
            pred_mask.putpalette(color_map)
            # pred_mask.save(vis_fn)

            pred_mask_np = np.array(pred_mask.convert("RGB"))
            im_pred = PILImage.fromarray(pred_mask_np)

            # Original image
            # BGR->RGB
            img = cv2.imread(os.path.join(cfg.DATASET.DATA_DIR,
                                          img_name))[..., ::-1]
            im_ori = PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            # log_writer.add_image("Images/{}".format(img_name), img, epoch)
            # add ground truth (label) images
            grt = grts[i]
            grt = grt[0:valid_shape[0], 0:valid_shape[1]]
            grt_pil = PILImage.fromarray(grt.astype(np.uint8), mode='P')
            grt_pil.putpalette(color_map)
            grt_pil = grt_pil.resize((org_shape[1], org_shape[0]))
            grt = np.array(grt_pil.convert("RGB"))
            im_grt = PILImage.fromarray(grt)

            im_grt_cat = PILImage.blend(im_ori, im_grt, 0.5)
            im_pred_cat = PILImage.blend(im_ori, im_pred, 0.5)

            im_ori = join(im_ori, im_ori, flag="vertical")
            im_grt_cat = join(im_grt_cat, im_grt, flag="vertical")
            im_pred_cat = join(im_pred_cat, im_pred, flag="vertical")
            new_img = join(im_ori, im_grt_cat)
            new_img = join(new_img, im_pred_cat)
            new_img.save(vis_fn)

            img_cnt += 1
            print("#{} show image path: {}".format(img_cnt, vis_fn))


if __name__ == '__main__':
    args = parse_args()
    if args.cfg_file is not None:
        cfg.update_from_file(args.cfg_file)
    if args.opts:
        cfg.update_from_list(args.opts)
    cfg.check_and_infer()
    print(pprint.pformat(cfg))
    visualize(cfg, **args.__dict__)
