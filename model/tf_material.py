"""
TensorFlow wrapper functions.

date: 9/29
author: arabian9ts
"""

import tensorflow as tf
import numpy as np

import structure as vgg
from functools import reduce
from activation import Activation

def pooling(input, name, strides=None):
    """
    Args: output of just before layer
    Return: max_pooling layer
    """
    strides = vgg.pool_strides if not strides else strides
    return tf.nn.max_pool(input, ksize=vgg.ksize, strides=strides, padding='SAME', name=name)

def convolution(input, name, ksize=None, strides=None, train_phase=tf.constant(True)):
    """
    Args: output of just before layer
    Return: convolution layer
    """
    print('Current input size in convolution layer is: '+str(input.get_shape().as_list()))
    with tf.variable_scope(name):
        size = vgg.structure[name]
        kernel = get_weight(size[0]) if not ksize else ksize
        bias = get_bias(size[1])
        strides = vgg.conv_strides if not strides else strides
        conv = tf.nn.conv2d(input, kernel, strides=strides, padding='SAME', name=name)
        out = tf.nn.relu(tf.add(conv, bias))
    return batch_normalization(out, train_phase)

def fully_connection(input, activation, name, train_phase=tf.constant(True)):
    """
    Args: output of just before layer
    Return: fully_connected layer
    """
    size = vgg.structure[name]
    with tf.variable_scope(name):
        shape = input.get_shape().as_list()
        dim = reduce(lambda x, y: x * y, shape[1:])
        x = tf.reshape(input, [-1, dim])

        weights = get_weight([dim, size[0][0]])
        biases = get_bias(size[1])

        fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
        fc = activation(fc)

        print('Input shape is: '+str(shape))
        print('Total nuron count is: '+str(dim))

        return batch_normalization(fc, train_phase)

def batch_normalization(input,  train_phase=tf.constant(True)):
    """
    Batch Normalization
    Result in:
        * Reduce DropOut
        * Sparse Dependencies on Initial-value(e.g. weight, bias)
        * Accelerate Convergence

    Args: output of convolution or fully-connection layer
    Returns: Normalized batch
    """
    decay=0.9
    eps=1e-5
    shape = input.get_shape().as_list()
    n_out = shape[-1]
    beta = tf.Variable(tf.zeros([n_out]))
    gamma = tf.Variable(tf.ones([n_out]))

    if len(shape) == 2:
        batch_mean, batch_var = tf.nn.moments(input, [0])
    else:
        batch_mean, batch_var = tf.nn.moments(input, [0, 1, 2])

    ema = tf.train.ExponentialMovingAverage(decay=decay)

    def mean_var_with_update():
        ema_apply_op = ema.apply([batch_mean, batch_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)
    mean, var = tf.cond(train_phase, mean_var_with_update,
      lambda: (ema.average(batch_mean), ema.average(batch_var)))

    return tf.nn.batch_normalization(input, mean, var, beta, gamma, eps)


def get_weight(shape):
    """
    generate weight tensor

    Args: weight size
    Return: initialized weight tensor
    """
    initial = tf.truncated_normal(shape, 0.0, 1.0) * 0.01
    return tf.Variable(initial)

def get_bias(shape):
    """
    generate bias tensor

    Args: bias size
    Return: initialized bias tensor
    """
    return tf.Variable(tf.truncated_normal(shape, 0.0, 1.0) * 0.01)