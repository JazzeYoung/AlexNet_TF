#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 10:23:31 2017

@author: ai-yang
"""
# AlexNet with tensorflow
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import re
import sys
import tarfile

from six.moves import urllib
import numpy

import tensorflow as tf

import cifar10_input


# initialization

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '/home/ai/tensorflow/AlexExp/alexnet_TF/',
                           """Path to the CIFAR-10 data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")

# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


# Constants describing the training process.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.

# For AlexNet
INITIAL_LEARNING_RATE = 0.01
LEARNING_RATE_DECAY_FACTOR = 0.1
MOMENTUM = 0.9
WEIGHT_DACAY = 5e-4
MOVING_AVERAGE_DECAY = 5e-3

def print_activations(t):
      print(t.op.name, ' ', t.get_shape().as_list())
      
  
def _variable_with_weight_decay(var, learning_rate):
    """ Update an initialized Variable with weight decay.
      Returns:
    Variable Tensor
    """
    #dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    #weight_decay = tf.Variable(tf.constant(0.0, shape=var.shape,dtype=tf.float32), name='weight_decay')
    weight_decay = learning_rate*WEIGHT_DACAY*var
    #print_activations(weight_decay)
    #tf.add_to_collection('losses', weight_decay)
    var -= weight_decay
    return var
# modeling--inference
def model(images, train=False):
    # Model Stucture
    parameters = []
#    weights = tf.Variable()  use this form in trainning
#    biases = tf.Variable()
    # Layer 1:conv1
    print_activations(images)
    with tf.variable_scope('conv1') as scope:#[11, 11, 3, 96]strides[1,4,4,1]
        kernel = tf.Variable(tf.truncated_normal([5, 5, 3, 64], dtype=tf.float32, stddev=0.01), name = 'weights')
        conv1 = tf.nn.conv2d(images, kernel, strides=[1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=True, name='biases')
        conv1 = tf.nn.relu(tf.nn.bias_add(conv1, biases), name=scope.name)
        #print_activations(conv1)
        parameters += [kernel, biases]
        
    norm1 = tf.nn.lrn(conv1, depth_radius=3, bias=2, alpha=1e-4, beta=0.75, name='norm1')
    #print_activations(norm1)
    pool1 = tf.nn.max_pool(norm1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')
    #print_activations(pool1)
    # Layer 2:conv2
    with tf.variable_scope('conv2') as scope:#[5, 5, 48, 256]2GPU [5,5,96,256]GPU
        kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 64], dtype=tf.float32, stddev=0.01), name='weights')
        conv2 = tf.nn.conv2d(pool1, kernel, strides=[1,1,1,1], padding='SAME')
        biases = tf.Variable(tf.constant(1.0, shape=[64], dtype=tf.float32), trainable=True, name='biases')
        conv2 = tf.nn.relu(tf.nn.bias_add(conv2, biases), name=scope.name)
        #print_activations(conv2)
        parameters += [kernel, biases]
    norm2 = tf.nn.lrn(conv2,depth_radius=3, bias=2, alpha=1e-4, beta=0.75, name='norm2')
    pool2 = tf.nn.max_pool(norm2, ksize=[1,3, 3, 1], strides=[1,2,2,1], padding='VALID', name='pool2')
    #print_activations(pool2)
    # Layer 3:conv3
    with tf.variable_scope('conv3') as scope:#[3, 3, 128, 384]2GPU [3,3,256,384]
        kernel = tf.Variable(tf.truncated_normal([1,1,64,64], dtype=tf.float32, stddev=0.01), name='weights')
        conv3 = tf.nn.conv2d(pool2, kernel, strides=[1,1,1,1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=True, name='biases')
        conv3 = tf.nn.relu(tf.nn.bias_add(conv3, biases), name=scope.name)
        #print_activations(conv3)
        parameters += [kernel, biases]
    
    
    with tf.variable_scope('conv4') as scope:#[3, 3, 192, 384]2GPU [3,3,384,384]
        kernel = tf.Variable(tf.truncated_normal([1,1, 64, 64], dtype=tf.float32, stddev=0.01), name='weights')
        conv4 = tf.nn.conv2d(conv3, kernel, strides=[1,1,1,1], padding='SAME')
        biases = tf.Variable(tf.constant(1.0, shape=[64], dtype=tf.float32), trainable=True, name='biases')
        conv4 = tf.nn.relu(tf.nn.bias_add(conv4, biases), name=scope.name)
        #print_activations(conv4)
        parameters += [kernel, biases]
        
    with tf.variable_scope('conv5') as scope:#[3,3,192,256]2GPU [3,3,384,256]
        kernel = tf.Variable(tf.truncated_normal([1,1, 64, 64], dtype=tf.float32, stddev=0.01), name='weights')
        conv5 = tf.nn.conv2d(conv4, kernel, strides=[1,1,1,1], padding='SAME')
        biases = tf.Variable(tf.constant(1.0, shape=[64], dtype=tf.float32), trainable=True, name='biases')
        conv5 = tf.nn.relu(tf.nn.bias_add(conv5, biases),  name=scope.name)
        #print_activations(conv5)
        parameters += [kernel, biases]
    pool5 = tf.nn.max_pool(conv5, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID', name='pool5')
    #print_activations(pool5)
    with tf.variable_scope('fc1') as scope:#[43264,4096]
        reshape = tf.reshape(pool5, [FLAGS.batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = tf.Variable(tf.truncated_normal([dim, 384],#[dim, 4096]IMAGENET
                                          stddev=0.1), name='weights')
        #print_activations(weights)
        #weights = tf.Variable(tf.truncated_normal([9216,4096], dtype=tf.float32, stddev=0.01), name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32), trainable=True, name='biases')
        if train:
            drop1 = tf.nn.dropout(reshape, keep_prob=0.5, name='drop1')
            fc1 = tf.nn.relu(tf.matmul(drop1, weights)+biases, name=scope.name)
        else:
            fc1 = tf.nn.relu(tf.matmul(reshape, weights)+biases, name=scope.name)
        parameters += [weights, biases]
        #print_activations(fc1)
        
    with tf.variable_scope('fc2') as scope:
        weights = tf.Variable(tf.truncated_normal([384, 192], dtype=tf.float32, stddev=0.01), name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32), trainable=True, name='biases')
        if train:
            drop2 = tf.nn.dropout(fc1, keep_prob=0.5, name='drop2')
            fc2 = tf.nn.relu(tf.matmul(drop2, weights)+biases, name=scope.name)
        else:
            fc2 = tf.nn.relu(tf.matmul(fc1, weights)+biases, name=scope.name)
        #print_activations(fc2)
        parameters += [weights, biases]
    
    with tf.variable_scope('fc3') as scope:
        weights = tf.Variable(tf.truncated_normal([192, 100], dtype=tf.float32, stddev=0.01), name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[100], dtype=tf.float32), trainable=True, name='biases')
        fc3 = tf.nn.relu(tf.matmul(fc2, weights)+biases, name=scope.name)
        #print_activations(fc3)
        parameters += [weights, biases]
        
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.Variable(tf.truncated_normal([100, 10], dtype=tf.float32, stddev=0.01), name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[10],dtype=tf.float32), name='biases')
        softmax_linear = tf.add(tf.matmul(fc3, weights), biases, name=scope.name)
        #print_activations(softmax_linear)
        parameters += [weights, biases]
        
    return softmax_linear, parameters

# Loss function
def loss(output, parameters, label, train=False):
    # Cross entropy + L2 weight regularizer
    #Cross entropy loss
    label = tf.cast(label, tf.int64)
    #print_activations(label)
    #print_activations(output)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=label, name='xentropy')
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    # tf.nn.softmax_cross_entropy_with_logits
    # L2 regularization for the fully connected parameters.
    
    # Estimate the weight decay term 0.0005*lr*w_i, here, w_i is before the momentum SGD
    if train:
        for var in parameters:
            #print_activations(var)
            var = _variable_with_weight_decay(var=var, learning_rate=INITIAL_LEARNING_RATE)
        
    tf.add_to_collection('losses', loss)
    
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

# Training prototype
def train(total_loss, global_step):
    # learning parameters
    # Decay the learning rate exponentially based on the number of steps.
    # Variables that affect learning rate.
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
    tf.summary.scalar('learning_rate', lr)
    
    # Generate moving averages of all losses and associated summaries.
    # Compute gradients.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])
      # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
#        opt = tf.train.GradientDescentOptimizer(learning_rate=lr)
#        grads = opt.compute_gradients(total_loss)    
        opt = tf.train.MomentumOptimizer(learning_rate=lr, momentum=MOMENTUM,name='Momentum')
        grads = opt.compute_gradients(total_loss)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    # inner loop where global_step++
    
        
    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)
    # Track the moving averages of all trainable variables.
    
    variable_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # Get optimal training parameters
    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')
    

    return train_op

## Training process
#def run_alexnet_train_proc(x, y, val_x, val_y, sess):
#    batchsize = 10
#    n_epoch = 10
#    num_batches = 100
#    global_step = tf.Variable(0, name='global_step', trainable=False)
#    for i in xrange(num_batches):
#        total_loss = 0.0
#        softmax_linear, parameters = model(images=x(i), train=True)
#        total_loss = loss(output=softmax_linear, parameters=parameters,
#                          label=y(i), train=True)
#        train(total_loss=total_loss, global_step=global_step)
#        sess.run([parameters, total_loss])
#    return parameters

def distorted_inputs():
  """Construct distorted input for CIFAR training using the Reader ops.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
  images, labels = cifar10_input.distorted_inputs(data_dir=data_dir,
                                                  batch_size=FLAGS.batch_size)
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels

def inputs(eval_data):
  """Construct input for CIFAR evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
  images, labels = cifar10_input.inputs(eval_data=eval_data,
                                        data_dir=data_dir,
                                        batch_size=FLAGS.batch_size)
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels

#def main():
#    
#    # trainning data
#    train_x, train_y, val_x, val_y = initialize(dataset='MNIST')
#    
#    init = tf.global_variables_initializer()
#    
#    #get minibatch
#    session = tf.Session()
#    session.run(init)
#    run_alexnet_train_proc(x=train_x, y=train_y, val_x=val_x, val_y=val_y, session)
#    # testing erros
#    
#    logits, params = model(x)
#    loss = loss(logits, label_x)
#    
#    return

#if __name__ = '__main__' :
#    cifar10_input.inputs()
# 