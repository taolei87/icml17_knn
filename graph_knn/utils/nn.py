import numpy as np
from scipy.sparse import csc_matrix, vstack
import tensorflow as tf
import math

def batch_normalization(x, scope, decay=0.999, eps=1e-6, training=True):
    ndim = len(x.get_shape().as_list())
    fdim = x.get_shape().as_list()[-1]
    with tf.variable_scope(scope):
        gamma = tf.get_variable("scale", [fdim], tf.float32, tf.constant_initializer(1.0))
        beta = tf.get_variable("offset", [fdim], tf.float32, tf.constant_initializer(0.0))
        mean = tf.get_variable("mean", [fdim], tf.float32, tf.constant_initializer(0.0), trainable=False)
        var = tf.get_variable("variance", [fdim], tf.float32, tf.constant_initializer(1.0), trainable=False)
        if training:
            x_mean, x_var = tf.nn.moments(x, range(ndim - 1))
            avg_mean = tf.assign(mean, mean * decay + x_mean * (1.0 - decay))
            avg_var = tf.assign(var, var * decay + x_var * (1.0 - decay))
            with tf.control_dependencies([avg_mean, avg_var]):
                return tf.nn.batch_normalization(x, x_mean, x_var, beta, gamma, eps)
        else:
            return tf.nn.batch_normalization(x, mean, var, beta, gamma, eps)

def batch_normalization_with_mask(x, mask, scope, decay=0.999, eps=1e-6, training=True):
    ndim = len(x.get_shape().as_list())
    fdim = x.get_shape().as_list()[-1]
    with tf.variable_scope(scope):
        gamma = tf.get_variable("scale", [fdim], tf.float32, tf.constant_initializer(1.0))
        beta = tf.get_variable("offset", [fdim], tf.float32, tf.constant_initializer(0.0))
        mean = tf.get_variable("mean", [fdim], tf.float32, tf.constant_initializer(0.0), trainable=False)
        var = tf.get_variable("variance", [fdim], tf.float32, tf.constant_initializer(1.0), trainable=False)
        if training:
            x_mean, x_var = tf.nn.weighted_moments(x, range(ndim - 1), mask)
            avg_mean = tf.assign(mean, mean * decay + x_mean * (1.0 - decay))
            avg_var = tf.assign(var, var * decay + x_var * (1.0 - decay))
            with tf.control_dependencies([avg_mean, avg_var]):
                return tf.nn.batch_normalization(x, x_mean, x_var, beta, gamma, eps)
        else:
            return tf.nn.batch_normalization(x, mean, var, beta, gamma, eps)

def lookup_table(input_, vocab_size, output_size, scope):
    with tf.variable_scope(scope):
        embedding = tf.get_variable("embedding", [vocab_size, output_size], tf.float32, tf.random_normal_initializer(stddev=1.0 / math.sqrt(output_size)))
    return tf.nn.embedding_lookup(embedding, input_)

def linear(input_, output_size, scope, init_bias=0.0):
    shape = input_.get_shape().as_list()
    stddev = min(1.0 / math.sqrt(shape[-1]), 0.1)
    with tf.variable_scope(scope):
        W = tf.get_variable("Matrix", [shape[-1], output_size], tf.float32, tf.random_normal_initializer(stddev=stddev))
    if init_bias is None:
        return tf.matmul(input_, W)
    with tf.variable_scope(scope):
        b = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(init_bias))
    return tf.matmul(input_, W) + b

def linearND(input_, output_size, scope, init_bias=0.0):
    shape = input_.get_shape().as_list()
    ndim = len(shape)
    stddev = min(1.0 / math.sqrt(shape[-1]), 0.1)
    with tf.variable_scope(scope):
        W = tf.get_variable("Matrix", [shape[-1], output_size], tf.float32, tf.random_normal_initializer(stddev=stddev))
    X_shape = tf.gather(tf.shape(input_), range(ndim-1))
    target_shape = tf.concat(0, [X_shape, [output_size]])
    exp_input = tf.reshape(input_, [-1, shape[-1]])
    if init_bias is None:
        res = tf.matmul(exp_input, W)
    else:
        with tf.variable_scope(scope):
            b = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(init_bias))
        res = tf.matmul(exp_input, W) + b
    res = tf.reshape(res, target_shape)
    res.set_shape(shape[:-1] + [output_size])
    return res

