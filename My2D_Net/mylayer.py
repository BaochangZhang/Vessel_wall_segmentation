#!/usr/bin/python
# -*- coding: utf-8 -*-
from My2D_Net.ops import *


def conv_bn_relu_drop(input, kernel, out_channels, strides=1, with_bias=False, reg=1e-4,
                      norm_type=None, istraining=False,
                      activefunction=None,
                      dropout_using=False, keep_prob=0.8,
                      namescope=None):
    """ A Convolution-BatchNorm-ReLU-dropout layer
    Args:
      input: 4D tensor
      kernel: integer,2D tensor of shape, the size of kernel
      out_channels: integer, number of filters (output depth)
      strides: integer
      with_bias： boolean
      reg: the regulation coefficient of L2 loss, default: 1e-4
      norm_type: 'instance' or 'batch' or None
      istraining: boolean
      activefunction: 'tf.nn.relu','tf.nn.tanh',etc. or None
      dropout_using: boolean
      keep_prob, float, the value between 0.0 to 1.0, default:0.8
      namescope: string, e.g. 'c7sk-32'
    Returns:
      4D tensor
    """
    with tf.variable_scope(namescope):
        conv = conv2d(input, kernel, out_channels, strides, with_bias, reg, namescope='Conv2d')
        output = norm(conv, istraining, norm_type)
        if activefunction is not None:
            output = activefunction(output)
        if dropout_using is True:
            output = tf.nn.dropout(output, keep_prob)
    return output


def transpose_conv2d(input, kernel, out_channels, strides=2, with_bias=False, reg=1e-4,
                     norm_type=None, istraining=False,
                     activefunction=None,
                     dropout_using=False, keep_prob=0.8,
                     namescope=None):
    """ A de-Convolution-BatchNorm-ReLU-dropout layer
    Args:
      input: 4D tensor
      kernel: integer,2D tensor of shape, the size of kernel
      out_channels: integer, number of filters (output depth)
      strides: integer, default：2
      with_bias： boolean
      reg: the regulation coefficient of L2 loss, default: 1e-4
      norm_type: 'instance' or 'batch' or None
      istraining: boolean
      activefunction: 'tf.nn.relu','tf.nn.tanh',etc. or None
      dropout_using: boolean
      keep_prob, float, the value between 0.0 to 1.0, default:0.8
      namescope: string, e.g. 'c7sk-32'
    Returns:
      4D tensor
    """
    with tf.variable_scope(namescope):
        dconv = deconv2d(input, kernel, out_channels, strides, with_bias, reg, namescope='De_Conv2d')
        output = norm(dconv, istraining, norm_type)
        if activefunction is not None:
            output = activefunction(output)
        if dropout_using is True:
            output = tf.nn.dropout(output, keep_prob)
    return output


def bilinear_resize(input, referred_data, namescope=None):
    """ b-spline algorithm for resizing input or downsampling/ upsampling
    Args:
      input: 4D tensor
      referred_data: 4D tensor
    Returns:
      4D tensor
    """
    with tf.name_scope(namescope):
        new_size = referred_data.get_shape().as_list()[1:3]
        output = tf.image.resize_bilinear(input, new_size, name='downsample')
    return output


def pooling2d(input, kernal_size=2, strides=2, poolfunction=tf.nn.max_pool):
    """ A Pooling layer for downsampling
    Args:
      input: 4D tensor
      kernal_size: integer: default: 2
      strides: integer, default: 2
      poolfunction: 'tf.nn.max_pool','tf.nn.avg_pool'
    Returns:
      4D tensor
    """
    return poolfunction(input, ksize=[1, kernal_size, kernal_size, 1],
                        strides=[1, strides, strides, 1], padding='SAME')


def DASPPv1(x, out_channels, aspp_rates, reg=1e-4, namescope=None):
    with tf.variable_scope(namescope):
        # 压缩特征
        conv_input = conv2d(x, kernel=[1, 1], out_channels=out_channels//2, with_bias=True, reg=reg,  namescope="conv_1x1")
        inputs_size = conv_input.get_shape().as_list()[1:3]
        # three 3x3 convolutions with rates
        net = conv_input
        for rate_value in aspp_rates:
            conv_atrous = conv2d_atrous(net, kernel=[3, 3], out_channels=out_channels//2, rate=rate_value,
                                        with_bias=True, reg=reg, namescope="aspp_rate"+str(rate_value))
            net = tf.concat(axis=3, values=[net, conv_atrous])

        # (b) the image-level features
        with tf.variable_scope("image_level_features"):
            image_level_features = tf.reduce_mean(conv_input, [1, 2], name='global_average_pooling', keepdims=True)
            # 1x1 convolution with 256 filters( and batch normalization)
            image_level_features = conv2d(image_level_features, kernel=[1, 1], out_channels=out_channels//2, reg=reg, namescope='conv_1x1')
            # bilinearly upsample features
            image_level_features = tf.image.resize_bilinear(image_level_features, inputs_size, name='upsample')
        net = tf.concat(axis=3, values=[net, image_level_features], name='concat')
        net = conv2d(net, kernel=[1, 1], out_channels=out_channels, with_bias=True, reg=reg,  namescope="conv_concat")
    return net


def DASPPv2(x, out_channels, aspp_rates, reg=1e-4, namescope=None):
    with tf.variable_scope(namescope):
        # 压缩特征
        conv_input = conv2d(x, kernel=[1, 1], out_channels=out_channels//2, with_bias=True, reg=reg,  namescope="conv_1x1")
        # three 3x3 convolutions with rates
        net = conv_input
        for rate_value in aspp_rates:
            conv_atrous = conv2d_atrous(net, kernel=[3, 3], out_channels=out_channels//2, rate=rate_value,
                                        with_bias=True, reg=reg, namescope="aspp_rate"+str(rate_value))
            net = tf.concat(axis=3, values=[net, conv_atrous])
        net = conv2d(net, kernel=[1, 1], out_channels=out_channels, with_bias=True, reg=reg,  namescope="conv_concat")
    return net
