#!/usr/bin/python
# -*- coding: utf-8 -*-
from My2D_Net.mylayer import *


def Unet_bias_norm(Input, training, reg, norm_name='instance', n_class=2):
    with tf.variable_scope('encoder'):
        # phase 1
        conv1 = conv_bn_relu_drop(Input, kernel=[3, 3], out_channels=64, strides=1, with_bias=True, reg=reg, norm_type=norm_name, istraining=training, activefunction=tf.nn.relu, namescope='Conv1_1')
        conv1 = conv_bn_relu_drop(conv1, kernel=[3, 3], out_channels=64, strides=1, with_bias=True, reg=reg, norm_type=norm_name, istraining=training, activefunction=tf.nn.relu, namescope='Conv1_2')
        pool1 = pooling2d(conv1, kernal_size=2, strides=2, poolfunction=tf.nn.max_pool)  # C:64, WH:64
        # phase 2
        conv2 = conv_bn_relu_drop(pool1, kernel=[3, 3], out_channels=128, strides=1, with_bias=True, reg=reg, norm_type=norm_name, istraining=training, activefunction=tf.nn.relu, namescope='Conv2_1')
        conv2 = conv_bn_relu_drop(conv2, kernel=[3, 3], out_channels=128, strides=1, with_bias=True, reg=reg, norm_type=norm_name, istraining=training, activefunction=tf.nn.relu, namescope='Conv2_2')
        pool2 = pooling2d(conv2, kernal_size=2, strides=2, poolfunction=tf.nn.max_pool)  # C:128, WH:32
        # phase 3
        conv3 = conv_bn_relu_drop(pool2, kernel=[3, 3], out_channels=256, strides=1, with_bias=True, reg=reg, norm_type=norm_name, istraining=training, activefunction=tf.nn.relu, namescope='Conv3_1')
        conv3 = conv_bn_relu_drop(conv3, kernel=[3, 3], out_channels=256, strides=1, with_bias=True, reg=reg, norm_type=norm_name, istraining=training, activefunction=tf.nn.relu, namescope='Conv3_2')
        pool3 = pooling2d(conv3, kernal_size=2, strides=2, poolfunction=tf.nn.max_pool)  # C:256, WH:16
        # phase 4
        conv4 = conv_bn_relu_drop(pool3, kernel=[3, 3], out_channels=512, strides=1, with_bias=True, reg=reg, norm_type=norm_name, istraining=training, activefunction=tf.nn.relu, namescope='Conv4_1')
        conv4 = conv_bn_relu_drop(conv4, kernel=[3, 3], out_channels=512, strides=1, with_bias=True, reg=reg, norm_type=norm_name, istraining=training, activefunction=tf.nn.relu, namescope='Conv4_2')
        # C:512, WH:16
    with tf.variable_scope('decoder'):
        # phase 1
        dconv1 = transpose_conv2d(conv4, kernel=[2, 2], out_channels=256, strides=2, with_bias=True, reg=reg, norm_type=norm_name, istraining=training, activefunction=tf.nn.relu, namescope='de_Conv1')
        dconv_concat1 = tf.concat(values=[conv3, dconv1], axis=3)
        conv5 = conv_bn_relu_drop(dconv_concat1, kernel=[3, 3], out_channels=256, strides=1, with_bias=True, reg=reg, norm_type=norm_name, istraining=training, activefunction=tf.nn.relu, namescope='Conv5_1')
        conv5 = conv_bn_relu_drop(conv5, kernel=[3, 3], out_channels=256, strides=1, with_bias=True, reg=reg, norm_type=norm_name, istraining=training, activefunction=tf.nn.relu, namescope='Conv5_2')
        # phase 2
        dconv2 = transpose_conv2d(conv5, kernel=[2, 2], out_channels=128, strides=2, with_bias=True, reg=reg, norm_type=norm_name, istraining=training, activefunction=tf.nn.relu, namescope='de_Conv2')
        dconv_concat2 = tf.concat(values=[conv2, dconv2], axis=3)
        conv6 = conv_bn_relu_drop(dconv_concat2, kernel=[3, 3], out_channels=128, strides=1, with_bias=True, reg=reg, norm_type=norm_name, istraining=training, activefunction=tf.nn.relu, namescope='Conv6_1')
        conv6 = conv_bn_relu_drop(conv6, kernel=[3, 3], out_channels=128, strides=1, with_bias=True, reg=reg, norm_type=norm_name, istraining=training, activefunction=tf.nn.relu, namescope='Conv6_2')
        # phase 3
        dconv3 = transpose_conv2d(conv6, kernel=[2, 2], out_channels=64, strides=2, with_bias=True, reg=reg, norm_type=norm_name, istraining=training, activefunction=tf.nn.relu, namescope='de_Conv3')
        dconv_concat3 = tf.concat(values=[conv1, dconv3], axis=3)
        conv7 = conv_bn_relu_drop(dconv_concat3, kernel=[3, 3], out_channels=64, strides=1, with_bias=True, reg=reg, norm_type=norm_name, istraining=training, activefunction=tf.nn.relu, namescope='Conv7_1')
        conv7 = conv_bn_relu_drop(conv7, kernel=[3, 3], out_channels=64, strides=1, with_bias=True, reg=reg, norm_type=norm_name, istraining=training, activefunction=tf.nn.relu, namescope='Conv7_2')

    with tf.variable_scope('classifier'):
        logits = conv_bn_relu_drop(conv7, kernel=[1, 1], out_channels=n_class, strides=1, with_bias=True, reg=reg, namescope='logits_conv')
        output_map = tf.nn.softmax(logits, name='output')
    return output_map


def stats_graph(graph):
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('FLPOS:{}; Trainable Params:{}'.format(flops.total_float_ops, params.total_parameters))


if __name__ == '__main__':
    with tf.Graph().as_default() as graph:
        Input = tf.placeholder("float", shape=[None, 128, 128, 1], name="Input")
        logits = Unet_bias_norm(Input, training=True, reg=0, n_class=2)
        stats_graph(graph)