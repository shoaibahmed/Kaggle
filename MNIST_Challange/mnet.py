# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains model definitions for versions of the Oxford VGG network.
These model definitions were introduced in the following technical report:
  Very Deep Convolutional Networks For Large-Scale Image Recognition
  Karen Simonyan and Andrew Zisserman
  arXiv technical report, 2015
  PDF: http://arxiv.org/pdf/1409.1556.pdf
  ILSVRC 2014 Slides: http://www.robots.ox.ac.uk/~karen/pdf/ILSVRC_2014.pdf
  CC-BY-4.0
More information can be obtained from the VGG website:
www.robots.ox.ac.uk/~vgg/research/very_deep/
Usage:
  with slim.arg_scope(vgg.vgg_arg_scope()):
    outputs, end_points = vgg.vgg_a(inputs)
  with slim.arg_scope(vgg.vgg_arg_scope()):
    outputs, end_points = vgg.vgg_16(inputs)
@@vgg_a
@@vgg_16
@@vgg_19
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import tensorflow.contrib.slim as slim


def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x * leak, x)

def prelu(x, name="prelu"):
  leak = tf.get_variable(name,
                         shape=x.get_shape()[-1],
                         initializer=tf.constant_initializer(0.1),
                         dtype=x.dtype)
  return tf.maximum(x * leak, x)

def resUnit(input_layer, numInputUnits, i):
    with tf.variable_scope("res_unit" + str(i)):
        part1 = slim.batch_norm(input_layer, activation_fn=None)
        part2 = tf.nn.relu(part1)
        part3 = slim.conv2d(part2, numInputUnits, [3, 3], activation_fn=None, padding='SAME')
        part4 = slim.batch_norm(part3, activation_fn=None)
        part5 = tf.nn.relu(part4)
        part6 = slim.conv2d(part5, numInputUnits, [3, 3], activation_fn=None, padding='SAME')
        output = input_layer + part6
        return output

def mnet_arg_scope(weight_decay=0.0005):
  """Defines the VGG arg scope.
  Args:
    weight_decay: The l2 regularization coefficient.
  Returns:
    An arg_scope.
  """
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      # activation_fn=tf.nn.relu,
                      activation_fn=prelu,
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      biases_initializer=tf.zeros_initializer()):
    with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
      return arg_sc


def mnet(inputs,
           dropout_keep_prob,
           num_classes=10,
           scope='mnet'):
  '''
  Custom CNN for MNIST called as MNET inspired from VGG architecture making use of consecutive 3x3 convolutions
  followed by max pooling
  '''
  with tf.variable_scope(scope, 'mnet', [inputs]) as sc:
    end_points_collection = sc.name + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      # 28x28
      net = slim.repeat(inputs, 2, slim.conv2d, 64, [5, 5], scope='conv1')
      net = resUnit(net, 64, 1)
      net = slim.max_pool2d(net, [2, 2], scope='pool1')

      # 14x14
      net = slim.batch_norm(net, activation_fn=None)
      net = slim.repeat(net, 3, slim.conv2d, 128, [5, 5], scope='conv2')
      net = resUnit(net, 128, 2)
      net = slim.max_pool2d(net, [2, 2], scope='pool2')

      # 7x7
      net = slim.batch_norm(net, activation_fn=None)
      net = slim.repeat(net, 2, slim.conv2d, 256, [3, 3], padding='VALID', scope='conv3')
      net = resUnit(net, 256, 3)

      # 3x3
      net = slim.conv2d(net, 1024, [3, 3], padding='VALID', scope='fc4')
      # net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='dropout4')
      # net = slim.conv2d(net, 4096, [1, 1], scope='fc5')
      net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='dropout5')
      
      net = slim.conv2d(net, num_classes, [1, 1],
                        activation_fn=None,
                        scope='fc6')
  # Convert end_points_collection into a end_point dict.
  end_points = slim.utils.convert_collection_to_dict(end_points_collection)
  logits = tf.squeeze(net, [1, 2], name='fc6/squeezed')
  end_points[sc.name + '/fc6'] = logits

  end_points['Predictions'] = tf.nn.softmax(logits, name='Predictions')
  return logits, end_points


def residual_mnet(inputs,
           dropout_keep_prob,
           num_classes=10,
           scope='residual_mnet'):
  '''
  Custom CNN for MNIST called as MNET inspired from VGG architecture making use of consecutive 3x3 convolutions
  followed by max pooling
  '''
  with tf.variable_scope(scope, 'residual_mnet', [inputs]) as sc:
    end_points_collection = sc.name + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      # 28x28
      net = slim.conv2d(inputs, 64, [3, 3], scope='conv1')
      net = resUnit(net, 64, 1)
      net = slim.max_pool2d(net, [2, 2], scope='pool1')

      # 14x14
      net = slim.conv2d(net, 128, [3, 3], scope='conv2')
      net = resUnit(net, 128, 2)
      net = slim.max_pool2d(net, [2, 2], scope='pool2')

      # 7x7
      net = slim.conv2d(net, 128, [3, 3], padding='VALID', scope='conv3_1')
      net = resUnit(net, 128, 3)
      net = slim.conv2d(net, 128, [3, 3], padding='VALID', scope='conv3_2')
      net = resUnit(net, 128, 4)

      # 3x3
      net = slim.conv2d(net, 4096, [3, 3], padding='VALID', scope='fc4')
      net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='dropout4')
      
      net = slim.conv2d(net, num_classes, [1, 1],
                        activation_fn=None,
                        scope='fc6')

  # Convert end_points_collection into a end_point dict.
  end_points = slim.utils.convert_collection_to_dict(end_points_collection)
  logits = tf.squeeze(net, [1, 2], name='fc6/squeezed')
  end_points[sc.name + '/fc6'] = logits

  end_points['Predictions'] = tf.nn.softmax(logits, name='Predictions')
  return logits, end_points


def lenet(inputs,
           dropout_keep_prob,
           num_classes=10,
           scope='mnet'):
  '''
  Custom CNN for MNIST called as MNET inspired from VGG architecture making use of consecutive 3x3 convolutions
  followed by max pooling
  '''
  with tf.variable_scope(scope, 'mnet', [inputs]) as sc:
    end_points_collection = sc.name + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      # 28x28
      net = slim.conv2d(inputs, 64, [7, 7], scope='conv1_1', activation_fn=prelu, padding='SAME') # 24x24
      net = resUnit(net, 64, 1)
      net = slim.conv2d(inputs, 64, [3, 3], scope='conv1_2', activation_fn=prelu, padding='SAME') # 24x24
      net = resUnit(net, 64, 2)
      net = slim.conv2d(inputs, 64, [5, 5], scope='conv1_3', activation_fn=prelu, padding='VALID') # 24x24
      net = resUnit(net, 64, 3)
      net = slim.max_pool2d(net, [2, 2], scope='pool1') # 12x12

      # 12x12
      net = slim.batch_norm(net, activation_fn=None)
      net = slim.conv2d(net, 128, [7, 7], scope='conv2_1', activation_fn=prelu, padding='SAME') # 8x8
      net = resUnit(net, 128, 4)
      net = slim.conv2d(net, 128, [3, 3], scope='conv2_2', activation_fn=prelu, padding='SAME') # 8x8
      net = resUnit(net, 128, 5)
      net = slim.conv2d(net, 128, [5, 5], scope='conv2_3', activation_fn=prelu, padding='VALID') # 8x8
      net = resUnit(net, 128, 6)
      net = slim.max_pool2d(net, [2, 2], scope='pool2') # 4x4

      # 4x4
      # net = slim.batch_norm(net, activation_fn=None)
      net = slim.flatten(net)
      net = slim.fully_connected(net, 500, activation_fn=prelu, scope='fc3')
      # net = slim.fully_connected(net, 500, activation_fn=lrelu, scope='fc3')
      net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='dropout3')
      
      logits = slim.fully_connected(net, num_classes, activation_fn=None, scope='fc4')
      # end_points[sc.name + '/fc4'] = logits

  # Convert end_points_collection into a end_point dict.
  end_points = slim.utils.convert_collection_to_dict(end_points_collection)
  # logits = tf.squeeze(net, [1, 2], name='fc6/squeezed')
  # end_points[sc.name + '/fc6'] = logits

  end_points['Predictions'] = tf.nn.softmax(logits, name='Predictions')
  return logits, end_points


def lenet_tf(inputs,
           dropout_keep_prob,
           num_classes=10,
           scope='mnet'):
  '''
  Custom CNN for MNIST called as MNET inspired from VGG architecture making use of consecutive 3x3 convolutions
  followed by max pooling
  '''
  # with tf.variable_scope(scope, 'mnet', [inputs]) as sc:
  # Convolutional Layer #1
  stddevConv1 = (2.0 / 1.0)**0.5
  conv1 = tf.layers.conv2d(
      inputs=inputs,
      filters=32,
      kernel_size=[5, 5],
      # kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
      kernel_initializer=tf.truncated_normal_initializer(stddev=stddevConv1),
      bias_initializer=tf.ones_initializer(),
      padding="valid",
      activation=None,
      name="Conv1")

  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2
  stddevConv2 = (2.0 / 32.0)**0.5
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      # kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
      kernel_initializer=tf.truncated_normal_initializer(stddev=stddevConv2),
      bias_initializer=tf.ones_initializer(),
      padding="valid",
      activation=None,
      name="Conv2")

  # Pooling Layer #2
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Dense Layer
  stddevFC1 = (2.0 / float(4 * 4 * 64))**0.5
  pool2_flat = tf.reshape(pool2, [-1, 4 * 4 * 64])
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu, 
    kernel_initializer=tf.truncated_normal_initializer(stddev=stddevFC1), name="fc1")
    # kernel_initializer=tf.contrib.layers.xavier_initializer(), name="fc1")
  dropout = tf.layers.dropout(inputs=dense, rate=dropout_keep_prob, training=True)

  # Logits Layer
  stddevLogits = (2.0 / 1024.0)**0.5
  logits = tf.layers.dense(inputs=dropout, units=num_classes, activation=None,
    kernel_initializer=tf.truncated_normal_initializer(stddev=stddevLogits), name="logits")
    # kernel_initializer=tf.contrib.layers.xavier_initializer(), name="logits")

  end_points = {}
  end_points['Predictions'] = tf.nn.softmax(logits, name='Predictions')
  return logits, end_points


mnet.default_image_size = 28