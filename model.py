from __future__ import print_function

import os
import time
import random

from PIL import Image
import tensorflow as tf
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

from utils import save_hsi, data_augmentation

tf.compat.v1.disable_eager_execution()
tf = tf.compat.v1

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

def concat(layers):
    return tf.concat(layers, axis=3)

def DecomNet(input_im, layer_num, filter_channel=64, num_channels=3, kernel_size=3):
    input_max = tf.reduce_max(input_im, axis=3, keepdims=True)
    input_im = concat([input_max, input_im])
    with tf.variable_scope('DecomNet', reuse=tf.AUTO_REUSE):
        conv = tf.layers.conv2d(input_im, filter_channel, kernel_size * 3, padding='same',
                                activation=None, name="shallow_feature_extraction")
        for idx in range(layer_num):
            conv = tf.layers.conv2d(conv, filter_channel, kernel_size, padding='same',
                                    activation=tf.nn.relu, name='activated_layer_%d' % idx)
        # Output num_channels reflectance + 1 illuminance channel
        conv = tf.layers.conv2d(conv, num_channels + 1, kernel_size, padding='same',
                                activation=None, name='recon_layer')

    R = tf.sigmoid(conv[:,:,:,0:num_channels])
    L = tf.sigmoid(conv[:,:,:,num_channels:num_channels+1])

    return R, L

def RelightNet(input_L, input_R, channel=64, kernel_size=3):
    input_im = concat([input_R, input_L])
    with tf.variable_scope('RelightNet'):
        conv0 = tf.layers.conv2d(input_im, channel, kernel_size, padding='same', activation=None)
        conv1 = tf.layers.conv2d(conv0, channel, kernel_size, strides=2, padding='same', activation=tf.nn.relu)
        conv2 = tf.layers.conv2d(conv1, channel, kernel_size, strides=2, padding='same', activation=tf.nn.relu)
        conv3 = tf.layers.conv2d(conv2, channel, kernel_size, strides=2, padding='same', activation=tf.nn.relu)
        
        up1 = tf.image.resize_nearest_neighbor(conv3, (tf.shape(conv2)[1], tf.shape(conv2)[2]))
        deconv1 = tf.layers.conv2d(up1, channel, kernel_size, padding='same', activation=tf.nn.relu) + conv2
        up2 = tf.image.resize_nearest_neighbor(deconv1, (tf.shape(conv1)[1], tf.shape(conv1)[2]))
        deconv2 = tf.layers.conv2d(up2, channel, kernel_size, padding='same', activation=tf.nn.relu) + conv1
        up3 = tf.image.resize_nearest_neighbor(deconv2, (tf.shape(conv0)[1], tf.shape(conv0)[2]))
        deconv3 = tf.layers.conv2d(up3, channel, kernel_size, padding='same', activation=tf.nn.relu) + conv0
        
        deconv1_resize = tf.image.resize_nearest_neighbor(deconv1, (tf.shape(deconv3)[1], tf.shape(deconv3)[2]))
        deconv2_resize = tf.image.resize_nearest_neighbor(deconv2, (tf.shape(deconv3)[1], tf.shape(deconv3)[2]))
        feature_gather = concat([deconv1_resize, deconv2_resize, deconv3])
        feature_fusion = tf.layers.conv2d(feature_gather, channel, 1, padding='same', activation=None)
        output = tf.layers.conv2d(feature_fusion, 1, 3, padding='same', activation=None)
    return output

class lowlight_enhance(object):
    def __init__(self, sess, num_channels=31):
        self.sess = sess
        self.num_channels = num_channels
        self.DecomNet_layer_num = 5

        # Build the model with hyperspectral channels
        self.input_low = tf.placeholder(tf.float32, [None, None, None, num_channels], name='input_low')
        self.input_high = tf.placeholder(tf.float32, [None, None, None, num_channels], name='input_high')

        [R_low, I_low] = DecomNet(self.input_low, layer_num=self.DecomNet_layer_num, num_channels=self.num_channels)
        [R_high, I_high] = DecomNet(self.input_high, layer_num=self.DecomNet_layer_num, num_channels=self.num_channels)
        
        I_delta = RelightNet(I_low, R_low)

        # Replicate the single-channel illuminance to match the hyperspectral channels
        I_low_rep = tf.concat([I_low] * self.num_channels, axis=3)
        I_high_rep = tf.concat([I_high] * self.num_channels, axis=3)
        I_delta_rep = tf.concat([I_delta] * self.num_channels, axis=3)

        self.output_R_low = R_low
        self.output_I_low = I_low_rep
        self.output_I_delta = I_delta_rep
        self.output_S = R_low * I_delta_rep

        # Loss functions
        self.recon_loss_low = tf.reduce_mean(tf.abs(R_low * I_low_rep - self.input_low))
        self.recon_loss_high = tf.reduce_mean(tf.abs(R_high * I_high_rep - self.input_high))
        self.recon_loss_mutal_low = tf.reduce_mean(tf.abs(R_high * I_low_rep - self.input_low))
        self.recon_loss_mutal_high = tf.reduce_mean(tf.abs(R_low * I_high_rep - self.input_high))
        self.equal_R_loss = tf.reduce_mean(tf.abs(R_low - R_high))
        self.relight_loss = tf.reduce_mean(tf.abs(R_low * I_delta_rep - self.input_high))

        self.Ismooth_loss_low = self.smooth(I_low, R_low)
        self.Ismooth_loss_high = self.smooth(I_high, R_high)
        self.Ismooth_loss_delta = self.smooth(I_delta, R_low)

        self.loss_Decom = (self.recon_loss_low + self.recon_loss_high +
                           0.001 * self.recon_loss_mutal_low + 0.001 * self.recon_loss_mutal_high +
                           0.1 * self.Ismooth_loss_low + 0.1 * self.Ismooth_loss_high +
                           0.01 * self.equal_R_loss)
        self.loss_Relight = self.relight_loss + 3 * self.Ismooth_loss_delta

        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        optimizer = tf.train.AdamOptimizer(self.lr, name='AdamOptimizer')

        self.var_Decom = [var for var in tf.trainable_variables() if 'DecomNet' in var.name]
        self.var_Relight = [var for var in tf.trainable_variables() if 'RelightNet' in var.name]

        self.train_op_Decom = optimizer.minimize(self.loss_Decom, var_list=self.var_Decom)
        self.train_op_Relight = optimizer.minimize(self.loss_Relight, var_list=self.var_Relight)

        self.sess.run(tf.global_variables_initializer())

        self.saver_Decom = tf.train.Saver(var_list=self.var_Decom)
        self.saver_Relight = tf.train.Saver(var_list=self.var_Relight)

        print("[*] Initialize model successfully...")

    def gradient(self, input_tensor, direction):
        self.smooth_kernel_x = tf.reshape(tf.constant([[0, 0], [-1, 1]], tf.float32), [2, 2, 1, 1])
        self.smooth_kernel_y = tf.transpose(self.smooth_kernel_x, [1, 0, 2, 3])

        if direction == "x":
            kernel = self.smooth_kernel_x
        elif direction == "y":
            kernel = self.smooth_kernel_y
        return tf.abs(tf.nn.conv2d(input_tensor, kernel, strides=[1, 1, 1, 1], padding='SAME'))

    def ave_gradient(self, input_tensor, direction):
        return tf.layers.average_pooling2d(self.gradient(input_tensor, direction), pool_size=3, strides=1, padding='SAME')

    def smooth(self, input_I, input_R):
        # For hyperspectral reflectance, average over channels to get a grayscale approximation.
        input_R_gray = tf.reduce_mean(input_R, axis=3, keepdims=True)
        return tf.reduce_mean(
            self.gradient(input_I, "x") * tf.exp(-10 * self.ave_gradient(input_R_gray, "x")) +
            self.gradient(input_I, "y") * tf.exp(-10 * self.ave_gradient(input_R_gray, "y"))
        )

    def evaluate(self, epoch_num, eval_low_data, sample_dir, train_phase):
        print("[*] Evaluating for phase %s / epoch %d..." % (train_phase, epoch_num))

        for idx in range(len(eval_low_data)):
            input_low_eval = np.expand_dims(eval_low_data[idx], axis=0)

            if train_phase == "Decom":
                result_1, result_2 = self.sess.run([self.output_R_low, self.output_I_low],
                                                   feed_dict={self.input_low: input_low_eval})
            if train_phase == "Relight":
                result_1, result_2 = self.sess.run([self.output_S, self.output_I_delta],
                                                   feed_dict={self.input_low: input_low_eval})

            save_hsi(os.path.join(sample_dir, 'eval_%s_%d_R_low.mat' % (train_phase, idx + 1)), result_1)
            save_hsi(os.path.join(sample_dir, 'eval_%s_%d_I.mat' % (train_phase, idx + 1)), result_2)

    def train(self, train_low_data, train_high_data, eval_low_data, batch_size, patch_size,
              epoch, lr, sample_dir, ckpt_dir, eval_every_epoch, train_phase):
        assert len(train_low_data) == len(train_high_data)
        numBatch = len(train_low_data) // int(batch_size)

        loss_history = {}
        if train_phase == "Decom":
            loss_history['loss_Decom'] = []
            loss_history['recon_loss_low'] = []
            loss_history['recon_loss_high'] = []
            loss_history['recon_loss_mutal_low'] = []
            loss_history['recon_loss_mutal_high'] = []
            loss_history['equal_R_loss'] = []
            loss_history['Ismooth_loss_low'] = []
            loss_history['Ismooth_loss_high'] = []
        elif train_phase == "Relight":
            loss_history['loss_Relight'] = []
            loss_history['relight_loss'] = []
            loss_history['Ismooth_loss_delta'] = []

        # Select the appropriate training operation and saver
        if train_phase == "Decom":
            train_op = self.train_op_Decom
            train_loss = self.loss_Decom
            saver = self.saver_Decom
        elif train_phase == "Relight":
            train_op = self.train_op_Relight
            train_loss = self.loss_Relight
            saver = self.saver_Relight

        load_model_status, global_step = self.load(saver, ckpt_dir)
        if load_model_status:
            iter_num = global_step
            start_epoch = global_step // numBatch
            start_step = global_step % numBatch
            print("[*] Model restore success!")
        else:
            iter_num = 0
            start_epoch = 0
            start_step = 0
            print("[*] Not find pretrained model!")

        print("[*] Start training for phase %s, with start epoch %d start iter %d : " % (train_phase, start_epoch, iter_num))

        start_time = time.time()
        image_id = 0

        for epoch in range(start_epoch, epoch):
            if train_phase == "Decom":
                epoch_loss_dict = {'loss_Decom': 0.0, 'recon_loss_low': 0.0, 'recon_loss_high': 0.0,
                                   'recon_loss_mutal_low': 0.0, 'recon_loss_mutal_high': 0.0,
                                   'equal_R_loss': 0.0, 'Ismooth_loss_low': 0.0, 'Ismooth_loss_high': 0.0}
            elif train_phase == "Relight":
                epoch_loss_dict = {'loss_Relight': 0.0, 'relight_loss': 0.0, 'Ismooth_loss_delta': 0.0}
            count_batches = 0

            for batch_id in range(start_step, numBatch):
                # Generate data for a batch
                batch_input_low = np.zeros((batch_size, patch_size, patch_size, self.num_channels), dtype="float32")
                batch_input_high = np.zeros((batch_size, patch_size, patch_size, self.num_channels), dtype="float32")
                for patch_id in range(batch_size):
                    h, w, _ = train_low_data[image_id].shape
                    x = random.randint(0, h - patch_size)
                    y = random.randint(0, w - patch_size)
            
                    rand_mode = random.randint(0, 7)
                    batch_input_low[patch_id] = data_augmentation(train_low_data[image_id][x : x+patch_size, y : y+patch_size, :], rand_mode)
                    batch_input_high[patch_id] = data_augmentation(train_high_data[image_id][x : x+patch_size, y : y+patch_size, :], rand_mode)
                    
                    image_id = (image_id + 1) % len(train_low_data)
                    if image_id == 0:
                        tmp = list(zip(train_low_data, train_high_data))
                        random.shuffle(tmp)
                        train_low_data, train_high_data  = zip(*tmp)

                # Train step
                if train_phase == "Decom":
                    fetches = [train_op, self.loss_Decom, self.recon_loss_low, self.recon_loss_high,
                                self.recon_loss_mutal_low, self.recon_loss_mutal_high,
                                self.equal_R_loss, self.Ismooth_loss_low, self.Ismooth_loss_high]
                elif train_phase == "Relight":
                    fetches = [train_op, self.loss_Relight, self.relight_loss, self.Ismooth_loss_delta]
                results = self.sess.run(fetches, feed_dict={
                    self.input_low: batch_input_low,
                    self.input_high: batch_input_high,
                    self.lr: lr[epoch]
                })
                if train_phase == "Decom":
                    _, loss_val, rlow, rhigh, recon_mut_low, recon_mut_high, equal_R, smooth_low, smooth_high = results
                    epoch_loss_dict['loss_Decom'] += loss_val
                    epoch_loss_dict['recon_loss_low'] += rlow
                    epoch_loss_dict['recon_loss_high'] += rhigh
                    epoch_loss_dict['recon_loss_mutal_low'] += recon_mut_low
                    epoch_loss_dict['recon_loss_mutal_high'] += recon_mut_high
                    epoch_loss_dict['equal_R_loss'] += equal_R
                    epoch_loss_dict['Ismooth_loss_low'] += smooth_low
                    epoch_loss_dict['Ismooth_loss_high'] += smooth_high
                elif train_phase == "Relight":
                    _, loss_val, relight_loss_val, smooth_delta_val = results
                    epoch_loss_dict['loss_Relight'] += loss_val
                    epoch_loss_dict['relight_loss'] += relight_loss_val
                    epoch_loss_dict['Ismooth_loss_delta'] += smooth_delta_val
                count_batches += 1

                print("%s Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f" %
                      (train_phase, epoch + 1, batch_id + 1, numBatch, time.time() - start_time, loss_val))
                iter_num += 1

            # Evaluate and save checkpoint every eval_every_epoch epochs
            if (epoch + 1) % eval_every_epoch == 0:
                #self.evaluate(epoch + 1, eval_low_data, sample_dir=sample_dir, train_phase=train_phase)
                self.save(saver, iter_num, ckpt_dir, "RetinexNet-%s" % train_phase)

                plt.figure()
                for key, values in loss_history.items():
                    plt.plot(range(1, len(values) + 1), values, label=key)
                plt.xlabel('Epoch')
                plt.ylabel('Loss Value')
                plt.title('Loss Curves up to Epoch %d' % (epoch + 1))
                plt.legend()
                plt.savefig(os.path.join(sample_dir, 'loss_plot.png'))
                plt.close()

            
            for key in epoch_loss_dict:
                epoch_loss_dict[key] /= count_batches
                loss_history[key].append(epoch_loss_dict[key])

        print("[*] Finish training for phase %s." % train_phase)

    def save(self, saver, iter_num, ckpt_dir, model_name):
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        print("[*] Saving model %s" % model_name)
        saver.save(self.sess, os.path.join(ckpt_dir, model_name), global_step=iter_num)

    def load(self, saver, ckpt_dir):
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            full_path = tf.train.latest_checkpoint(ckpt_dir)
            try:
                global_step = int(full_path.split('/')[-1].split('-')[-1])
            except ValueError:
                global_step = None
            saver.restore(self.sess, full_path)
            return True, global_step
        else:
            print("[*] Failed to load model from %s" % ckpt_dir)
            return False, 0

    def test(self, test_low_data, test_high_data, test_low_data_names, save_dir, decom_flag):
        tf.global_variables_initializer().run()

        print("[*] Reading checkpoint...")
        load_model_status_Decom, _ = self.load(self.saver_Decom, './checkpoint/Decom')
        load_model_status_Relight, _ = self.load(self.saver_Relight, './checkpoint/Relight')
        if load_model_status_Decom and load_model_status_Relight:
            print("[*] Load weights successfully...")
        
        print("[*] Testing...")
        for idx in range(len(test_low_data)):
            print(test_low_data_names[idx])
            [_, name] = os.path.split(test_low_data_names[idx])
            suffix = name[name.find('.') + 1:]
            name = name[:name.find('.')]

            input_low_test = np.expand_dims(test_low_data[idx], axis=0)
            [R_low, I_low, I_delta, S] = self.sess.run(
                [self.output_R_low, self.output_I_low, self.output_I_delta, self.output_S],
                feed_dict = {self.input_low: input_low_test}
            )

            if decom_flag == 1:
                save_hsi(os.path.join(save_dir, name + "_R_low.mat"), R_low)
                save_hsi(os.path.join(save_dir, name + "_I_low.mat"), I_low)
                save_hsi(os.path.join(save_dir, name + "_I_delta.mat"), I_delta)
            save_hsi(os.path.join(save_dir, name + ".mat"), S)
