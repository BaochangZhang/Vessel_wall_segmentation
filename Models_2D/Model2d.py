#!/usr/bin/python
# -*- coding: utf-8 -*-
import os

import cv2
import numpy as np
import tensorflow as tf
import scipy.io as sio

from My2D_Net.Unet import Unet
from My2D_Net.Unet_bias import Unet_bias
from My2D_Net.Unet_bias_BN import Unet_bias_norm
from My2D_Net.Unet_bias_BN_drop import Unet_bias_norm_drop
from My2D_Net.Unet_Res_bias_BN_drop import Unet_res_bias_norm_drop
from My2D_Net.Unet_plus import DASP_Unet


# from MyNet.Unet import Unet
# from MyNet.ASPPnet import ASPPnet
# from Nets.Unet import Unet
# from Nets.ASSPNet import ASPPNet

os.environ['CUDA_VISIBLE_DEVICES'] = '2'


def write_log(str):
    file_path = os.path.join('log.txt')
    file = open(file_path, 'a')
    file.write(str)
    file.write('\n')
    file.close()

class Model2d(object):
    """
    A unet2d implementation
    :param image_height: number of height in the input image
    :param image_width: number of width in the input image
    :param channels: number of channels in the input image
    :param n_class: number of output labels
    :param costname: name of the cost function.Default is "cross_entropy"
    """
    def __init__(self, paras):
        self.model_paras = paras

        self.X = tf.placeholder("float", shape=[None, self.model_paras.H, self.model_paras.W, self.model_paras.C], name="Input")
        self.Y_gt = tf.placeholder("float", shape=[None, self.model_paras.H, self.model_paras.W, self.model_paras.n_class], name="Output_GT")
        self.W = tf.placeholder("float", shape=[None, self.model_paras.H, self.model_paras.W, self.model_paras.n_class], name="WEIGHT")

        self.lr = tf.placeholder('float', name="Learning_rate")
        self.training = tf.placeholder(tf.bool, name="Phase")
        self.drop_rate = tf.placeholder('float', name="DropOut")
        self.dynamic_W = [1, 1]

        if self.model_paras.net_name == "Unet":
            self.Y_pred = Unet(self.X, n_class=self.model_paras.n_class, reg=paras.regularizer_rate)

        if self.model_paras.net_name == "Unet_bias":
            self.Y_pred = Unet_bias(self.X, n_class=self.model_paras.n_class, reg=paras.regularizer_rate)

        if self.model_paras.net_name == "Unet_bias_norm":
            self.Y_pred = Unet_bias_norm(self.X, n_class=self.model_paras.n_class, training=self.training,
                                         norm_name=self.model_paras.norm_name, reg=paras.regularizer_rate)

        if self.model_paras.net_name == "Unet_bias_norm_drop":
            self.Y_pred = Unet_bias_norm_drop(self.X, n_class=self.model_paras.n_class, training=self.training,
                                              norm_name=self.model_paras.norm_name, reg=paras.regularizer_rate,
                                              keep_prob=self.drop_rate)

        if self.model_paras.net_name == "Unet_res_bias_norm_drop":
            self.Y_pred = Unet_res_bias_norm_drop(self.X, n_class=self.model_paras.n_class, training=self.training,
                                                  norm_name=self.model_paras.norm_name, reg=paras.regularizer_rate,
                                                  keep_prob=self.drop_rate)

        if self.model_paras.net_name == "DASP_Unet":
            self.Y_pred = DASP_Unet(self.X, n_class=self.model_paras.n_class, training=self.training,
                                    norm_name=self.model_paras.norm_name, reg=paras.regularizer_rate,
                                    keep_prob=self.drop_rate)

        self.cost_dl = self.__get_cost(cost_name=self.model_paras.cost_name, loss_type=self.model_paras.loss_type)
        self.cost_dynamic = self.__get_cost(cost_name='dice_loss4', loss_type=self.model_paras.loss_type)
        # self.cost_ce = self.__get_cost(cost_name="WCE2", loss_type=self.model_paras.loss_type)
        self.cost = tf.add_n(tf.get_collection('L2_loss')) + 0.2 * self.cost_dl + 0.8 * self.cost_dynamic

        self.accuracy1 = self.__get_metrics(accname='dice', label_id=1)
        self.accuracy2 = self.__get_metrics(accname='dice', label_id=2)

    def __get_metrics(self, accname, label_id=1):  # 二分类
        GT = tf.argmax(self.Y_gt, axis=3)
        GT_label = tf.cast(tf.equal(GT, tf.ones_like(GT)*label_id), dtype=tf.float32)
        Pre = tf.argmax(self.Y_pred, axis=3)
        Pre_label = tf.cast(tf.equal(Pre, tf.ones_like(Pre)*label_id), dtype=tf.float32)
        TP = tf.count_nonzero(Pre_label * GT_label, axis=(1, 2), dtype=tf.float32)
        TN = tf.count_nonzero((Pre_label - 1) * (GT_label - 1), axis=(1, 2), dtype=tf.float32)
        FP = tf.count_nonzero(Pre_label * (GT_label - 1), axis=(1, 2), dtype=tf.float32)
        FN = tf.count_nonzero((Pre_label - 1) * GT_label, axis=(1, 2), dtype=tf.float32)

        if accname == "accuracy":
            metric = tf.reduce_mean((TP+TN)/(TP+TN+FP+FN))
            return metric
        if accname == "precision":
            metric = tf.reduce_mean(TP / (TP + FP))
            return metric
        if accname == "recall":
            metric = tf.reduce_mean(TP / (TP + FN))
            return metric
        if accname == "f1_score":
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            metric = tf.reduce_mean(2 * precision * recall / (precision + recall))
            return metric
        if accname == "dice":
            eps = 1e-5
            metric = tf.reduce_mean((2.0*TP+eps) / (2.0*TP+FP+FN+eps))
            return metric

    def __get_cost(self, cost_name, loss_type = None):
        H, W, C = self.Y_gt.get_shape().as_list()[1:]
        if cost_name == "dice_loss1":  # dice1 loss
            smooth = 1e-5
            inse = tf.reduce_sum(self.Y_gt * self.Y_pred * self.W, axis=(1, 2, 3))
            if loss_type == 'jaccard':
                l = tf.reduce_sum(self.Y_pred * self.Y_pred * self.W, axis=(1, 2, 3))
                r = tf.reduce_sum(self.Y_gt * self.Y_gt * self.W, axis=(1, 2, 3))
            elif loss_type == 'sorensen':
                l = tf.reduce_sum(self.Y_pred * self.W, axis=(1, 2, 3))
                r = tf.reduce_sum(self.Y_gt * self.W, axis=(1, 2, 3))
            else:
                raise Exception("Unknow loss_type")
            dice = (2. * inse + smooth) / (l + r + smooth)
            loss = 1 - tf.reduce_mean(dice)
            return loss

        if cost_name == "dice_loss2":  # dice2 loss
            smooth = 1e-5
            inse = tf.reduce_sum(self.Y_gt * self.Y_pred * self.W, axis=(1, 2))
            if loss_type == 'jaccard':
                l = tf.reduce_sum(self.Y_pred * self.Y_pred * self.W, axis=(1, 2))
                r = tf.reduce_sum(self.Y_gt * self.Y_gt * self.W, axis=(1, 2))
            elif loss_type == 'sorensen':
                l = tf.reduce_sum(self.Y_pred * self.W, axis=(1, 2))
                r = tf.reduce_sum(self.Y_gt * self.W, axis=(1, 2))
            else:
                raise Exception("Unknow loss_type")
            dice = tf.reduce_mean((2. * inse + smooth) / (l + r + smooth), axis=1)
            loss = 1 - tf.reduce_mean(dice)
            return loss

        if cost_name == "dice_loss3":  # dice3 loss
            smooth = 1e-5
            Output = self.Y_pred[:, :, :, 1:C]
            target = self.Y_gt[:, :, :, 1:C]
            pred_flat = tf.reshape(Output, [-1, H * W * (C - 1)])
            true_flat = tf.reshape(target, [-1, H * W * (C - 1)])
            intersection = 2 * tf.reduce_sum(pred_flat * true_flat, axis=1) + smooth
            denominator = tf.reduce_sum(pred_flat, axis=1) + tf.reduce_sum(true_flat, axis=1) + smooth
            loss = 1-tf.reduce_mean(intersection / denominator)
            return loss
        if cost_name == "dice_loss4":
            smooth = 1e-5
            Output = self.Y_pred[:, :, :, 1]
            target = self.Y_gt[:, :, :, 1]
            pred_flat = tf.reshape(Output, [-1, H * W])
            true_flat = tf.reshape(target, [-1, H * W])
            intersection = 2 * tf.reduce_sum(pred_flat * true_flat, axis=1) + smooth
            denominator = tf.reduce_sum(pred_flat, axis=1) + tf.reduce_sum(true_flat, axis=1) + smooth
            loss1 = 1-tf.reduce_mean(intersection / denominator)

            smooth = 1e-5
            Output = self.Y_pred[:, :, :, C-1]
            target = self.Y_gt[:, :, :, C-1]
            pred_flat = tf.reshape(Output, [-1, H * W])
            true_flat = tf.reshape(target, [-1, H * W])
            intersection = 2 * tf.reduce_sum(pred_flat * true_flat, axis=1) + smooth
            denominator = tf.reduce_sum(pred_flat, axis=1) + tf.reduce_sum(true_flat, axis=1) + smooth
            loss2 = 1-tf.reduce_mean(intersection / denominator)
            loss = loss1*self.dynamic_W[0] + loss2*self.dynamic_W[1]
            return loss

        if cost_name == "cross_entroy":  # Cross-Entroy
            loss = tf.reduce_mean(-tf.reduce_sum(self.Y_gt * tf.log(tf.clip_by_value(self.Y_pred, 1e-15, 1.0)),
                                                 axis=(1, 2, 3)))
            return loss

    def decayed_learning_rate(self, global_step):
        decayed_lr = self.model_paras.init_learning_rate * pow(self.model_paras.decay_rate,
                                                               global_step//self.model_paras.decay_step)
        return decayed_lr

    def _next_batch(self, train_data, index_in_epoch):
        start = index_in_epoch
        index_in_epoch += self.model_paras.batch_size

        num_examples = train_data.shape[0]
        # when all trainig data have been already used, it is reorder randomly
        if index_in_epoch > num_examples:
            # shuffle the data
            perm = np.arange(num_examples)
            np.random.shuffle(perm)
            train_data = train_data[perm]
            # start next epoch
            start = 0
            index_in_epoch = self.model_paras.batch_size
            assert self.model_paras.batch_size <= num_examples
        end = index_in_epoch

        batch_xs_path = train_data[start:end]

        batch_xs = np.empty((len(batch_xs_path), self.model_paras.H, self.model_paras.W, self.model_paras.C))
        batch_ys = np.empty((len(batch_xs_path), self.model_paras.H, self.model_paras.W, self.model_paras.n_class))
        batch_ws = np.empty((len(batch_xs_path), self.model_paras.H, self.model_paras.W, self.model_paras.n_class))

        for num in range(len(batch_xs_path)):
            filedata = sio.loadmat(batch_xs_path[num][0])
            Image = np.asarray(filedata['image'], dtype=np.float32)

            label = np.asarray(filedata['Label'], dtype=np.float32)

            label = np.reshape(label, (self.model_paras.H, self.model_paras.W, 1))
            label = np.tile(label, (1, 1, self.model_paras.n_class))
            weight = np.zeros(shape=(self.model_paras.H, self.model_paras.W, self.model_paras.n_class))
            for i in range(self.model_paras.n_class):
                label[:, :, i] = np.equal(label[:, :, i], i).astype(np.float32)
                weight[:, :, i] = label[:, :, i]/(np.sum(label[:, :, i], axis=(0, 1))/(self.model_paras.H * self.model_paras.W)+1e-5)

            # 标准化
            mean_I = np.mean(Image, axis=(0, 1))
            std_I = np.std(Image, axis=(0, 1))
            re_img = (Image - mean_I) / std_I

            batch_xs[num, :, :, :] = np.reshape(re_img, (self.model_paras.H, self.model_paras.W, self.model_paras.C))
            batch_ys[num, :, :, :] = np.reshape(label, (self.model_paras.H, self.model_paras.W, self.model_paras.n_class))
            batch_ws[num, :, :, :] = np.reshape(weight, (self.model_paras.H, self.model_paras.W, self.model_paras.n_class))

        # Extracting images and labels from given data
        batch_xs = batch_xs.astype(np.float)
        batch_ys = batch_ys.astype(np.float)

        return batch_xs, batch_ys, batch_ws, train_data, index_in_epoch

    def train(self, train_images, test_images, Continue_Train=False):
        if os.path.isfile('log.txt'):
            os.remove('log.txt')
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = tf.train.AdamOptimizer(self.lr).minimize(self.cost)
            # train_op = tf.train.GradientDescentOptimizer(self.lr).minimize(self.cost)

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

        tf.summary.scalar("loss", self.cost)
        tf.summary.scalar("accuracy1", self.accuracy1)
        tf.summary.scalar("accuracy2", self.accuracy2)
        merged_summary_op = tf.summary.merge_all()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        if not os.path.exists(self.model_paras.logs_path):
            os.makedirs(self.model_paras.logs_path)
        summary_writer = tf.summary.FileWriter(self.model_paras.logs_path, graph=tf.get_default_graph())

        init = tf.global_variables_initializer()
        sess.run(init)

        if Continue_Train:
            ckpt = tf.train.get_checkpoint_state(self.model_paras.latest_model)
            if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint
                saver.restore(sess, tf.train.latest_checkpoint(self.model_paras.latest_model))
                print("load model")
            else:
                print("Check")

        DISPLAY_STEP = 1
        index_in_epoch = 0
        train_steps = len(train_images)//self.model_paras.batch_size*self.model_paras.train_epochs
        # epoch_step = len(train_images)//self.model_paras.batch_size
        for i in range(train_steps):
            # get new batch
            batch_xs, batch_ys, batch_ws, train_images, index_in_epoch = self._next_batch(train_images,index_in_epoch)

            decayed_lr = self.decayed_learning_rate(i)

            # check progress on every 1st,2nd,...,10th,20th,...,100th... step
            if i % DISPLAY_STEP == 0 or (i + 1) == train_steps:
                train_loss, train_accuracy1,  train_accuracy2 = sess.run([self.cost, self.accuracy1, self.accuracy2],
                                                                          feed_dict={self.X: batch_xs, self.Y_gt: batch_ys,
                                                                                     self.W: batch_ws,
                                                                                  self.lr: decayed_lr, self.training: True,
                                                                                  self.drop_rate: self.model_paras.dropout_rate})

                pred = sess.run(self.Y_pred, feed_dict={self.X: batch_xs,
                                                        self.training: False,
                                                        self.drop_rate: 1})

                result = np.reshape(np.argmax(pred[0, :, :, :], axis=-1), (self.model_paras.H, self.model_paras.W))
                result = result * 120
                cv2.imwrite("prelabel.bmp", result)

                Ximg = np.reshape(batch_xs[0, :, :, :], (self.model_paras.H, self.model_paras.W))
                Ximg = (Ximg-np.min(Ximg))/(np.max(Ximg)-np.min(Ximg))*255.0
                Ximg = np.clip(Ximg, 0, 255).astype('uint8')
                cv2.imwrite("Image.bmp", Ximg)

                Yimg = np.reshape(np.argmax(batch_ys[0, :, :, :], axis=-1), (self.model_paras.H, self.model_paras.W))
                Yimg = Yimg * 120
                cv2.imwrite("gt.bmp", Yimg)

                print('Train_epochs %d Lr:%.10f ,loss:%.10f ,acc_wall:%.10f,acc_lumen:%.10f' % (i, decayed_lr, train_loss,
                                                                                          train_accuracy1, train_accuracy2))
                self.dynamic_W[0] = train_accuracy2 / np.minimum(train_accuracy1, train_accuracy2)
                self.dynamic_W[1] = train_accuracy1 / np.minimum(train_accuracy1, train_accuracy2)
                if self.dynamic_W[0] >= self.dynamic_W[1]:
                    self.dynamic_W[0] *= 2.0
                else:
                    self.dynamic_W[1] *= 2.0
                print(self.dynamic_W)
                if i % (DISPLAY_STEP * 10) == 0 and i:
                    if DISPLAY_STEP < 100:
                        DISPLAY_STEP *= 10

            # train on batch
            _, summary = sess.run([train_op, merged_summary_op], feed_dict={self.X: batch_xs,
                                                                            self.Y_gt: batch_ys,
                                                                            self.W: batch_ws,
                                                                            self.lr: decayed_lr,
                                                                            self.training: True,
                                                                            self.drop_rate: self.model_paras.dropout_rate})
            summary_writer.add_summary(summary, i)

            if i % 250 == 0 and i > 0:
                if not os.path.exists(self.model_paras.model_path):
                    os.makedirs(self.model_paras.model_path)
                checkpoint_path = os.path.join(self.model_paras.model_path, 'model.ckpt')
                save_path = saver.save(sess, checkpoint_path, global_step=i)
                print("Model saved in file:", save_path)
                write_log("Model saved in file:"+save_path)

            if i % 250 == 0 and i > 1:
                dice_wall, dice_lumen = self.validate(test_images, sess)
                print('validation,acc_wall:%.10f, acc_lumen:%.10f' % (dice_wall, dice_lumen))
                write_log('validation,acc_wall:%.10f, acc_lumen:%.10f' % (dice_wall, dice_lumen))
                write_log('\n')

        summary_writer.close()

        if not os.path.exists(self.model_paras.model_path):
            os.makedirs(self.model_paras.model_path)
        checkpoint_path = os.path.join(self.model_paras.model_path, 'model_final.ckpt')
        save_path = saver.save(sess, checkpoint_path)
        print("Model saved in file:", save_path)
        write_log("Model saved in file:" + save_path)

        # if not os.path.exists(self.model_paras.latest_model):
        #     os.makedirs(self.model_paras.latest_model)
        # checkpoint_path = os.path.join(self.model_paras.latest_model, 'model_final.ckpt')
        # save_path = saver.save(sess, checkpoint_path)
        # print("Model saved in file:", save_path)

        # final validation
        dice_wall, dice_lumen = self.validate(test_images, sess)
        print('validation,acc_wall:%.10f, acc_lumen:%.10f' % (dice_wall, dice_lumen))
        write_log('validation,acc_wall:%.10f, acc_lumen:%.10f' % (dice_wall, dice_lumen))
        sess.close()

    def validate(self, test_images, sess):
        dice_wall = 0.0
        dice_lumen = 0.0
        for num in range(len(test_images)):
            filedata = sio.loadmat(test_images[num][0])
            Image = np.asarray(filedata['image'], dtype=np.float32)
            label = np.asarray(filedata['Label'], dtype=np.float32)
            label = np.reshape(label, (self.model_paras.H, self.model_paras.W, 1))
            label = np.tile(label, (1, 1, self.model_paras.n_class))
            for i in range(self.model_paras.n_class):
                label[:, :, i] = np.equal(label[:, :, i], i).astype(np.float32)
            # 标准化
            mean_I = np.mean(Image, axis=(0, 1))
            std_I = np.std(Image, axis=(0, 1))
            re_img = (Image - mean_I) / std_I

            test_img = np.reshape(re_img, (1, self.model_paras.H, self.model_paras.W, self.model_paras.C))
            test_label = np.reshape(label, (1, self.model_paras.H, self.model_paras.W, self.model_paras.n_class))
            train_accuracy1, train_accuracy2 = sess.run([self.accuracy1, self.accuracy2],
                                                        feed_dict={self.X: test_img, self.Y_gt: test_label,
                                                                   self.training: False, self.drop_rate: 1})
            # print('acc_wall:%.10f, acc_lumen:%.10f' % (train_accuracy1, train_accuracy2))
            dice_wall += train_accuracy1
            dice_lumen += train_accuracy2
        dice_wall /= len(test_images)
        dice_lumen /= len(test_images)
        return dice_wall, dice_lumen

    def prediction(self, Image, label=None):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        sess = tf.InteractiveSession()
        sess.run(init)
        ckpt = tf.train.get_checkpoint_state(self.model_paras.latest_model)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, tf.train.latest_checkpoint(self.model_paras.latest_model))
            print('load', tf.train.latest_checkpoint(self.model_paras.latest_model))
        # 标准化
        mean_I = np.mean(Image, axis=(0, 1))
        std_I = np.std(Image, axis=(0, 1))
        re_img = (Image - mean_I) / std_I
        test_img = np.reshape(re_img, (1, self.model_paras.H, self.model_paras.W, self.model_paras.C))

        if label is not None:
            label = np.reshape(label, (self.model_paras.H, self.model_paras.W, 1))
            label = np.tile(label, (1, 1, self.model_paras.n_class))
            for i in range(self.model_paras.n_class):
                label[:, :, i] = np.equal(label[:, :, i], i).astype(np.float32)
            test_label = np.reshape(label, (1, self.model_paras.H, self.model_paras.W, self.model_paras.n_class))

            pred, train_accuracy1, train_accuracy2 = sess.run([self.Y_pred, self.accuracy1, self.accuracy2],
                                                              feed_dict={self.X: test_img, self.Y_gt: test_label,
                                                                         self.training: False, self.drop_rate: 1})
            print('acc_wall:%.10f, acc_lumen:%.10f' % (train_accuracy1, train_accuracy2))
        else:
            pred = sess.run(self.Y_pred, feed_dict={self.X: test_img, self.training: False, self.drop_rate: 1})

        result = np.reshape(np.argmax(pred[0, :, :, :], axis=-1), (self.model_paras.H, self.model_paras.W))

        return result, train_accuracy1, train_accuracy2

    def eval_testset(self, test_images):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        sess = tf.InteractiveSession()
        sess.run(init)
        ckpt = tf.train.get_checkpoint_state(self.model_paras.latest_model)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, tf.train.latest_checkpoint(self.model_paras.latest_model))
            print('load', tf.train.latest_checkpoint(self.model_paras.latest_model))
        dice_wall, dice_lumen = self.validate(test_images, sess)
        print('validation,acc_wall:%.10f, acc_lumen:%.10f' % (dice_wall, dice_lumen))

