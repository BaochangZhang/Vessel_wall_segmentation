#!/usr/bin/python
# -*- coding: utf-8 -*-
import tensorflow as tf


def _weights(name, shape, mean=0.0, stddev=0.02, init_method=0, reg=0.0):
    """ Helper to create an initialized Variable
    Args:
      name: name of the variable
      shape: list of ints
      mean: mean of a Gaussian
      stddev: standard deviation of a Gaussian
    Returns:
      A trainable variable
    """
    if init_method == 0:
        return _random_init(name, shape, mean, stddev, reg)
    if init_method == 1:  # MSRA
        return _hkm_init(name, shape, reg)
    if init_method == 2:  # Xavier
        return _avix_init(name, shape, reg)


def _biases(name, shape, constant=0.0):
    """ Helper to create an initialized Bias with constant
    """
    return tf.get_variable(name, shape, initializer=tf.constant_initializer(constant))


def _random_init(name, shape, mean, stddev, reg):
    var = tf.get_variable(
        name, shape,
        initializer=tf.random_normal_initializer(
            mean=mean, stddev=stddev, dtype=tf.float32))
    if reg > 0:
        loss = tf.multiply(tf.nn.l2_loss(var), reg)
        tf.add_to_collection("L2_loss", loss)
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, var)
    return var


def _hkm_init(name, shape, reg):
    n_inputs = 1
    for i in range(len(shape)-1):
        n_inputs = 1 * shape[i]
    stddev = tf.sqrt(2.0 / n_inputs)
    initial = tf.truncated_normal(shape, stddev=stddev)
    var = tf.get_variable(initializer=initial, name=name)

    if reg > 0:
        loss = tf.multiply(tf.nn.l2_loss(var), reg)
        tf.add_to_collection("L2_loss", loss)
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, var)
    return var


def _avix_init(name, shape, reg):
    n_inputs = 1
    for i in range(len(shape) - 1):
        n_inputs = 1 * shape[i]
    n_outputs = shape[-1]
    init_range = tf.sqrt(6.0 / (n_inputs + n_outputs))
    initial = tf.random_uniform(shape, -init_range, init_range)
    var = tf.get_variable(initializer=initial, name=name)

    if reg > 0:
        loss = tf.multiply(tf.nn.l2_loss(var), reg)
        tf.add_to_collection("L2_loss", loss)
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, var)
    return var


def _batch_norm(input, is_training):
  """ Batch Normalization
  """
  with tf.variable_scope("batch_norm"):
    return tf.layers.batch_normalization(input,
                                         momentum=0.9,
                                         training=is_training)


def _instance_norm(input):
  """ Instance Normalization
  """
  with tf.variable_scope("instance_norm"):
    depth = input.get_shape()[3]
    scale = _weights("scale", [depth], mean=1.0)
    offset = _biases("offset", [depth])
    mean, variance = tf.nn.moments(input, axes=[1,2], keep_dims=True)
    epsilon = 1e-5
    inv = tf.rsqrt(variance + epsilon)
    normalized = (input-mean)*inv
    return scale*normalized + offset


def norm(input, is_training, norm_type='batch'):
    """ Use Instance Normalization or Batch Normalization or None
      """
    if norm_type == 'instance':
        return _instance_norm(input)
    elif norm_type == 'batch':
        return _batch_norm(input, is_training)
    else:
        return input


def conv2d(input, kernel, out_channels, strides=1, with_bias=False, reg=1e-4, namescope=None):
    """ A Convolution layer
    Args:
      input: 4D tensor
      kernel: A 2-D tensor of shape, the size of kernel
      out_channels: 1-D tensor, the number of filters
      reg: the regulation coefficient of L2 loss, default: 1e-4
      namescope: string,
      with_bias: bool
    Returns:
      4D tensor
    """
    input_shape = input.get_shape().as_list()
    with tf.variable_scope(namescope):
        # 3 types of parameter initialization
        # random initialization
        W = _weights(name=namescope + '_W', shape=[kernel[0], kernel[1], input_shape[-1], out_channels],
                     mean=0.0, stddev=0.02, init_method=0, reg=reg)

        # MSRA initialization
        # W = _weights(name=namescope + '_W', shape=[kernel[0], kernel[1], input_shape[-1], out_channels],
        #              init_method=1, reg=reg)

        # Xavier initialization
        # W = _weights(name=namescope + '_W', shape=[kernel[0], kernel[1], input_shape[-1], out_channels],
        #              init_method=2, reg=reg)
        if with_bias:
            B = _biases(name=namescope + '_B', shape=[out_channels], constant=0.0)
            conv_2d = tf.nn.conv2d(input, W, strides=[1, strides, strides, 1], padding='SAME') + B
        else:
            conv_2d = tf.nn.conv2d(input, W, strides=[1, strides, strides, 1], padding='SAME')
    return conv_2d


def conv2d_atrous(input, kernel, out_channels, rate, with_bias=False, reg=1e-4, namescope=None):
    """ A dilated Convolution layer
    Args:
      input: 4D tensor
      kernel: A 2-D tensor of shape, the size of kernel
      rate: dilated rate
      out_channels: 1-D tensor, the number of filters
      reg: the regulation coefficient of L2 loss, default: 1e-4
      namescope: string,
      with_bias: bool
    Returns:
      4D tensor
    """
    with tf.variable_scope(namescope):
        input_shape = input.get_shape().as_list()
        # 3 types of parameter initialization
        # random initialization
        W = _weights(name=namescope + '_W', shape=[kernel[0], kernel[1], input_shape[-1], out_channels],
                     mean=0.0, stddev=0.02, init_method=0, reg=reg)

        # MSRA initialization
        # W = _weights(name=namescope + '_W', shape=[kernel[0], kernel[1], input_shape[-1], out_channels],
        #              init_method=1, reg=reg)

        # Xavier initialization
        # W = _weights(name=namescope + '_W', shape=[kernel[0], kernel[1], input_shape[-1], out_channels],
        #              init_method=2, reg=reg)

        if with_bias:
            B = _biases(name=namescope + '_B', shape=[out_channels], constant=0.0)
            conv_atrous = tf.nn.atrous_conv2d(input, W, rate, padding='SAME') + B
        else:
            conv_atrous = tf.nn.atrous_conv2d(input, W, rate, padding='SAME')
    return conv_atrous


def deconv2d(input, kernel, out_channels, strides=2, with_bias=False, reg=1e-4, namescope=None):
    """ A de Convolution layer
    Args:
      input: 4D tensor
      kernel: A 2-D tensor of shape, the size of kernel
      out_channels: 1-D tensor, the number of filters
      reg: the regulation coefficient of L2 loss, default: 1e-4
      namescope: string,
      with_bias: bool
    Returns:
      4D tensor
    """

    with tf.variable_scope(namescope):
        input_shape = input.get_shape().as_list()
        # 3 types of parameter initialization
        # random initialization
        W = _weights(name=namescope + '_W', shape=[kernel[0], kernel[1], out_channels, input_shape[-1]],
                     mean=0.0, stddev=0.02, init_method=0, reg=reg)

        # MSRA initialization
        # W = _weights(name=namescope + '_W', shape=[kernel[0], kernel[1], out_channels, input_shape[-1]],
        #              init_method=1, reg=reg)

        # Xavier initialization
        # W = _weights(name=namescope + '_W', shape=[kernel[0], kernel[1], out_channels, input_shape[-1]],
        #              init_method=2, reg=reg)

        output_shape = tf.stack([tf.shape(input)[0], input_shape[1] * strides, input_shape[2] * strides, out_channels])
        if with_bias:
            B = _biases(name=namescope + '_B', shape=[out_channels], constant=0.0)
            dconv = tf.nn.conv2d_transpose(input, W, output_shape=output_shape, strides=[1, strides, strides, 1], padding='SAME') + B
        else:
            dconv = tf.nn.conv2d_transpose(input, W, output_shape=output_shape, strides=[1, strides, strides, 1], padding='SAME')
    return dconv