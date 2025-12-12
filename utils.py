""" Utility functions. """
import numpy as np
import os
import random
# import tensorflow as tf

# from tensorflow.contrib.layers.python import layers as tf_layers
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow.compat.v1.layers as tf_layers

from tensorflow.python.platform import flags


FLAGS = flags.FLAGS

## Image helper
def get_images(paths, labels, nb_samples=None, shuffle=True):
    if nb_samples is not None:
        sampler = lambda x: random.sample(x, nb_samples)
    else:
        sampler = lambda x: x
    images = [(i, os.path.join(path, image)) \
        for i, path in zip(labels, paths) \
        for image in sampler(os.listdir(path))]
    if shuffle:
        random.shuffle(images)
    return images

## Network helpers
def conv_block(inp, cweight, bweight, reuse, scope, activation=tf.nn.relu, max_pool_pad='VALID', residual=False):
    """ Perform, conv, batch norm, nonlinearity, and max pool """
    stride, no_stride = [1,2,2,1], [1,1,1,1]

    if FLAGS.max_pool:
        conv_output = tf.nn.conv2d(inp, cweight, no_stride, 'SAME') + bweight
    else:
        conv_output = tf.nn.conv2d(inp, cweight, stride, 'SAME') + bweight
    normed = normalize(conv_output, activation, reuse, scope)
    if FLAGS.max_pool:
        normed = tf.nn.max_pool(normed, stride, stride, max_pool_pad)
    return normed

# def normalize(inp, activation, reuse, scope):
#     if FLAGS.norm == 'batch_norm':
#         return tf_layers.batch_norm(inp, activation_fn=activation, reuse=reuse, scope=scope)
#     elif FLAGS.norm == 'layer_norm':
#         return tf_layers.layer_norm(inp, activation_fn=activation, reuse=reuse, scope=scope)
#     elif FLAGS.norm == 'None':
#         if activation is not None:
#             return activation(inp)
#         else:
#             return inp

def normalize(inp, activation, reuse, scope):
    if FLAGS.norm == 'batch_norm':
        # Use tf.nn.batch_normalization
        with tf.variable_scope(scope, reuse=reuse):
            # Get input dimensions
            input_shape = inp.get_shape()
            # Last dimension
            params_shape = input_shape[-1:]  
            
            # reused if reuse=True
            beta = tf.get_variable('beta', params_shape, initializer=tf.zeros_initializer())
            gamma = tf.get_variable('gamma', params_shape, initializer=tf.ones_initializer())
            
            # For conv layers: normalize over batch, height, width
            if len(input_shape) == 4:  
                # Conv layer
                axes = [0, 1, 2]
            else:  
                # FC layer
                axes = [0]
            
            # Calculate batch statistics
            batch_mean, batch_variance = tf.nn.moments(inp, axes=axes)
            # Apply batch normalization
            epsilon = 1e-8
            normalized = tf.nn.batch_normalization(inp, batch_mean, batch_variance, beta, gamma, epsilon)
            
            if activation:
                normalized = activation(normalized)
            
            return normalized
    
    elif FLAGS.norm == 'layer_norm':
        with tf.variable_scope(scope, reuse=reuse):
            # Manual layer norm implementation
            mean, variance = tf.nn.moments(inp, axes=[-1], keepdims=True)
            normalized = (inp - mean) / tf.sqrt(variance + 1e-8)
            
            # Learnable parameters
            params_shape = inp.get_shape()[-1:]
            gamma = tf.get_variable('gamma', params_shape, initializer=tf.ones_initializer())
            beta = tf.get_variable('beta', params_shape, initializer=tf.zeros_initializer())
            
            normalized = normalized * gamma + beta
            
            if activation:
                normalized = activation(normalized)
            
            return normalized
    
    elif FLAGS.norm == 'None':
        if activation is not None:
            return activation(inp)
        else:
            return inp
        

## Loss functions
def mse(pred, label):
    pred = tf.reshape(pred, [-1])
    label = tf.reshape(label, [-1])
    return tf.reduce_mean(tf.square(pred-label))

def xent(pred, label):
    # Note - with tf version <=0.12, this loss has incorrect 2nd derivatives
    return tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=label) / FLAGS.update_batch_size
