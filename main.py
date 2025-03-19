from __future__ import print_function
import os
import argparse
import random
from glob import glob
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
tf = tf.compat.v1

from model import lowlight_enhance
from utils import load_hsi

parser = argparse.ArgumentParser(description='Hyperspectral Low-light Enhancement')

parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1, help='gpu flag, 1 for GPU and 0 for CPU')
parser.add_argument('--gpu_idx', dest='gpu_idx', default="0", help='GPU idx')
parser.add_argument('--gpu_mem', dest='gpu_mem', type=float, default=0.8, help="0 to 1, gpu memory usage")
parser.add_argument('--phase', dest='phase', default='train', help='train or test')

parser.add_argument('--epoch', dest='epoch', type=int, default=100, help='number of total epoches')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='number of samples in one batch')
parser.add_argument('--patch_size', dest='patch_size', type=int, default=48, help='patch size')
parser.add_argument('--start_lr', dest='start_lr', type=float, default=0.001, help='initial learning rate for adam')
parser.add_argument('--eval_every_epoch', dest='eval_every_epoch', type=int, default=20, help='evaluating and saving checkpoints every epoch')
parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default='./checkpoint', help='directory for checkpoints')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='directory for evaluating outputs')

parser.add_argument('--train_low_dir', dest='train_low_dir', default='./data/indoor/train/low', help='directory for train low inputs')
parser.add_argument('--train_high_dir', dest='train_high_dir', default='./data/indoor/train/high', help='directory for train high inputs')
parser.add_argument('--eval_low_dir', dest='eval_low_dir', default='./data/indoor/eval', help='directory for val inputs')

parser.add_argument('--save_dir', dest='save_dir', default='./test_results', help='directory for testing outputs')
parser.add_argument('--test_dir', dest='test_dir', default='./data/indoor/test', help='directory for testing inputs')
parser.add_argument('--decom', dest='decom', default=0, help='decom flag, 0 for enhanced results only and 1 for decomposition results')
parser.add_argument('--num_channels', dest='num_channels', type=int, default=64, help='number of hyperspectral channels')

args = parser.parse_args()

def lowlight_train(lowlight_enhance):
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)

    lr = args.start_lr * np.ones([args.epoch])
    lr[20:] = lr[0] / 10.0

    train_low_data = []
    train_high_data = []

    train_low_data_names = glob(args.train_low_dir +'/*.mat')
    train_low_data_names.sort()
    train_high_data_names = glob(args.train_high_dir +'/*.mat')
    train_high_data_names.sort()
    assert len(train_low_data_names) == len(train_high_data_names)
    print('[*] Number of training data: %d' % len(train_low_data_names))

    for idx in range(len(train_low_data_names)):
        low_im = load_hsi(train_low_data_names[idx])
        train_low_data.append(low_im)
        high_im = load_hsi(train_high_data_names[idx])
        train_high_data.append(high_im)

    eval_low_data = []
    eval_low_data_names = glob(args.eval_low_dir +'/*.mat')
    for idx in range(len(eval_low_data_names)):
        eval_low_im = load_hsi(eval_low_data_names[idx])
        eval_low_data.append(eval_low_im)

    lowlight_enhance.train(train_low_data, train_high_data, eval_low_data,
                           batch_size=args.batch_size, patch_size=args.patch_size,
                           epoch=args.epoch, lr=lr, sample_dir=args.sample_dir,
                           ckpt_dir=os.path.join(args.ckpt_dir, 'Decom'),
                           eval_every_epoch=args.eval_every_epoch, train_phase="Decom")

    lowlight_enhance.train(train_low_data, train_high_data, eval_low_data,
                           batch_size=args.batch_size, patch_size=args.patch_size,
                           epoch=args.epoch, lr=lr, sample_dir=args.sample_dir,
                           ckpt_dir=os.path.join(args.ckpt_dir, 'Relight'),
                           eval_every_epoch=args.eval_every_epoch, train_phase="Relight")


def lowlight_test(lowlight_enhance):
    if args.test_dir == None:
        print("[!] please provide --test_dir")
        exit(0)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    test_low_data_names = glob(os.path.join(args.test_dir) + '/*.mat')
    test_low_data = []
    for idx in range(len(test_low_data_names)):
        test_low_im = load_hsi(test_low_data_names[idx])
        test_low_data.append(test_low_im)

    lowlight_enhance.test(test_low_data, [], test_low_data_names, save_dir=args.save_dir, decom_flag=args.decom)


def main(_):

    seed_value = 42
    tf.set_random_seed(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    # Configure GPU options
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = False  # Set allow_growth to False
    config.intra_op_parallelism_threads = 1
    config.inter_op_parallelism_threads = 1

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    tf.config.set_visible_devices([], 'GPU')

    # Set the session with the config
    session = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(session)

    if args.use_gpu:
        print("[*] GPU\n")
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_idx
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_mem)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            model = lowlight_enhance(sess, num_channels=args.num_channels)
            if args.phase == 'train':
                lowlight_train(model)
            elif args.phase == 'test':
                lowlight_test(model)
            else:
                print('[!] Unknown phase')
                exit(0)
    else:
        print("[*] CPU\n")
        with tf.Session() as sess:
            model = lowlight_enhance(sess, num_channels=args.num_channels)
            if args.phase == 'train':
                lowlight_train(model)
            elif args.phase == 'test':
                lowlight_test(model)
            else:
                print('[!] Unknown phase')
                exit(0)

if __name__ == '__main__':
    tf.app.run()
