# Copyright UCL Business plc 2017. Patent Pending. All rights reserved.
#
# The MonoDepth Software is licensed under the terms of the UCLB ACP-A licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
#
# For any other use of the software not covered by the UCLB ACP-A Licence,
# please contact info@uclb.com

from __future__ import absolute_import, division, print_function

# only keep warnings and errors
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import numpy as np
import argparse
import re
import time
import tensorflow as tf
import tensorflow.contrib.slim as slim
import sys
import cv2

from monodepth_model import *
from monodepth_dataloader import *
from monodepth_my_set_loader import *
from average_gradients import *
from . import utils_

parser = argparse.ArgumentParser(description='Monodepth TensorFlow implementation.')
parser.add_argument('--mode', type=str, help='train or test', default='train')
parser.add_argument('--task', type=str, help='depth, semantic, semantic-depth', default='semantic',
                    choices=['depth', 'semantic', 'semantic-depth'])
parser.add_argument('--model_name', type=str, help='model name', default='semantic-monodepth')
parser.add_argument('--encoder', type=str, help='type of encoder, vgg or resnet50', default='vgg',
                    choices=['vgg', 'resnet50'])
parser.add_argument('--dataset', type=str, help='dataset to train on, kitti, or cityscapes', default='cityscapes',
                    choices=['kitti', 'cityscapes'])
parser.add_argument('--data_path', type=str, help='path to the data', required=True)
parser.add_argument('--filenames_file', type=str, help='path to the filenames text file', required=True)
parser.add_argument('--input_height', type=int, help='input height', default=256)
parser.add_argument('--input_width', type=int, help='input width', default=512)
parser.add_argument('--batch_size', type=int, help='batch size', default=2)
parser.add_argument('--num_epochs', type=int, help='number of epochs', default=50)
parser.add_argument('--learning_rate', type=float, help='initial learning rate', default=1e-4)
parser.add_argument('--lr_loss_weight', type=float, help='left-right consistency weight', default=1.0)
parser.add_argument('--alpha_image_loss', type=float, help='weight between SSIM and L1 in the image loss', default=0.85)
parser.add_argument('--disp_gradient_loss_weight', type=float, help='disparity smoothness weigth', default=0.1)
parser.add_argument('--do_stereo', help='if set, will train the stereo model', action='store_true')
parser.add_argument('--wrap_mode', type=str, help='bilinear sampler wrap mode, edge or border', default='border')
parser.add_argument('--use_deconv', help='if set, will use transposed convolutions', action='store_true')
parser.add_argument('--num_gpus', type=int, help='number of GPUs to use for training', default=1)
parser.add_argument('--num_threads', type=int, help='number of threads to use for data loading', default=8)
parser.add_argument('--output_directory', type=str,
                    help='output directory for test disparities, if empty outputs to checkpoint folder', default='')
parser.add_argument('--log_directory', type=str, help='directory to save checkpoints and summaries', default='./logs/')
parser.add_argument('--checkpoint_path', type=str, help='path to a specific checkpoint to load', default='')
parser.add_argument('--retrain', help='if used with checkpoint_path, will restart training from step zero',
                    action='store_true')
parser.add_argument('--full_summary',
                    help='if set, will keep more data for each summary. Warning: the file can become very large',
                    action='store_true')
parser.add_argument('--no_shuffle', help='Disabling shuffling at train time', action='store_true')
args = parser.parse_args()


def post_process_disparity(disp):
    _, h, w = disp.shape
    l_disp = disp[0, :, :]
    r_disp = np.fliplr(disp[1, :, :])
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def count_text_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return len(lines)

def recognize(params):
    """Test function."""

    dataloader = MonodepthDataloaderMy(args.data_path, args.filenames_file, params, args.dataset, args.mode)
    dataset = dataloader.left_image_batch
    semantic=valid=vars_to_restore= []  # dataloader.right_image_batch


    model = MonodepthModel(params, args.mode, args.task, dataset, dataset, semantic, valid)

    if args.checkpoint_path != '' and len(vars_to_restore) == 0:
        vars_to_restore = utils_.get_var_to_restore_list(args.checkpoint_path)
        print('Vars to restore ' + str(len(vars_to_restore)) + ' vs total vars ' + str(len(tf.trainable_variables())))
    else:
        print("No vars :(")

    # SESSION
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)

    # SAVER
    train_loader = tf.train.Saver()
    if args.checkpoint_path != '':
        train_loader = tf.train.Saver(var_list=vars_to_restore)

    # INIT
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

    # RESTORE
    if args.checkpoint_path == '':
        restore_path = tf.train.latest_checkpoint(args.log_directory + '/' + args.model_name)
    else:
        restore_path = args.checkpoint_path
    train_loader.restore(sess, restore_path)

    num_test_samples = count_text_lines(args.filenames_file)
    # num_test_samples=100

    print('now testing {} files'.format(num_test_samples))


    loop_size = int(num_test_samples / 100)
    for count in range(6,loop_size):
        disparities = np.zeros((100, params.height, params.width), dtype=np.float32)
        disparities_pp = np.zeros((100, params.height, params.width), dtype=np.float32)
        print("Память была выделена...")
        for step in range(100):
            #step_ = count * 100 + step
            disp = sess.run(model.disp_left_est[0])
            disparities[step] = disp[0].squeeze()
            disparities_pp[step] = post_process_disparity(disp.squeeze())
        progress = count / loop_size * 100
        print("Обработано {0} процентов файлов".format(progress))

        print('done.')

        print('writing results.')
        if args.output_directory == '':
            output_directory = os.path.dirname(args.checkpoint_path)
        else:
            output_directory = args.output_directory

        output_directory="C:/Temp/result_19"
        np.save('{0}/disparities_{1}.npy'.format(output_directory,count), disparities)
        np.save('{0}/disparities_pp_{1}.npy'.format(output_directory,count), disparities_pp)


    print('done.')


def main(_):
    params = monodepth_parameters(
        encoder=args.encoder,
        height=args.input_height,
        width=args.input_width,
        batch_size=args.batch_size,
        num_threads=args.num_threads,
        num_epochs=args.num_epochs,
        do_stereo=args.do_stereo,
        wrap_mode=args.wrap_mode,
        use_deconv=args.use_deconv,
        alpha_image_loss=args.alpha_image_loss,
        disp_gradient_loss_weight=args.disp_gradient_loss_weight,
        lr_loss_weight=args.lr_loss_weight, task=args.task,
        full_summary=args.full_summary)

    print(params)
    recognize(params)


if __name__ == '__main__':
    tf.app.run()
