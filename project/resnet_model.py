import tensorflow as tf
import numpy as np

from flags import *


BN_EPSILON = 0.001

filter_dims = [[((3, 64), (3, 64)), 3], \
               [((3, 128), (3, 128)), 4], \
               [((3, 256), (3, 256)), 6], \
               [((3, 512), (3, 512)), 3]]


'''
####################################
Single Layer
####################################
'''

def validate_tensor_dim(tensor, tensors_dim):
    if FLAGS.Run_Mode != "dev":
        return
    actual_dim = tensor.get_shape().as_list()
    print(actual_dim)
    for i in range(4):
        if actual_dim[i] != tensors_dim[i]:
            raise ValueError('Dimension does not match E[%d] - A[%d]' % (tensors_dim[i], actual_dim[i]))
    

'''
####################################
Single Layer
####################################
'''

def single_bn_layer(input_layer, dimension):
    mean, variance = tf.nn.moments(input_layer, axes=[0, 1, 2])
    beta = tf.get_variable('beta', dimension, tf.float32,
                               initializer=tf.constant_initializer(0.0, tf.float32))
    gamma = tf.get_variable('gamma', dimension, tf.float32,
                                initializer=tf.constant_initializer(1.0, tf.float32))
    bn_layer_out = tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma, BN_EPSILON)
    return bn_layer_out

def single_conv_layer(input_layer, shape, stride):
    regularizer = tf.contrib.layers.l2_regularizer(scale=FLAGS.Weight_Decay)
    initializer=tf.contrib.layers.xavier_initializer()
    filter = tf.get_variable('conv', shape=shape, initializer=initializer, regularizer=regularizer)
    conv_layer_out = tf.nn.conv2d(input_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
    return conv_layer_out

def single_relu_layer(input_layer):
    relu_layer_out = tf.nn.relu(input_layer)
    return relu_layer_out

'''
####################################
Sandwich Layer
####################################
'''

def sandwich_conv_bn_relu_layer(input_layer, filter_shape, stride):
    in_channel = input_layer.get_shape().as_list()[-1]
    out_channel = filter_shape[-1]
    
    conv_out = single_conv_layer(input_layer, shape=filter_shape, stride=stride)
    bn_out = single_bn_layer(conv_out, out_channel)
    relu_output = single_relu_layer(bn_out)
    return relu_output


def sandwich_bn_relu_conv_layer(input_layer, filter_shape, stride):
    in_channel = input_layer.get_shape().as_list()[-1]
    out_channel = filter_shape[-1]
    
    bn_out = single_bn_layer(input_layer, in_channel)
    relu_out = single_relu_layer(bn_out)
    conv_out = single_conv_layer(relu_out, shape=filter_shape, stride=stride)
    return conv_out

'''
####################################
Block Layer
####################################
'''

def residual_block(input_layer, input_dim, output_dim, filter_dims, reuse):
    block_input_size = input_dim[0]
    block_input_channel = input_dim[1]
    block_output_size = output_dim[0]
    block_output_channel = output_dim[1]
    
    # Validate inputs
    # 1. Validate input dimensions.
    # 2. Validate block_output_size = block_input_size // 2 OR block_output_size = block_input_size
    # validate_tensor_dim(input_layer, (FLAGS.Train_Batch_Size, block_input_size, block_input_size, block_input_channel))
    
    # Conv Branch
    if block_input_size == block_output_size * 2:
        stride = 2
    else:
        stride = 1
    
    layer_out = input_layer
    for i in range(len(filter_dims)):
        input_channel = layer_out.get_shape().as_list()[-1]
        with tf.variable_scope('block_%d' % i, reuse=reuse):
            filter_dim = filter_dims[i]
            filter_size = filter_dim[0]
            out_channel = filter_dim[1]
            layer_out = sandwich_bn_relu_conv_layer(layer_out, [filter_size, filter_size, input_channel, out_channel], stride)
            stride = 1    # Only shrink size for at most once.
    
    # Identity Branch
    identity_out = input_layer
    # Shrink size if needed.
    if block_input_size == block_output_size * 2:
        identity_out = tf.nn.avg_pool(identity_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    # Pad channels if needed.
    channel_padding_size = block_output_channel - block_input_channel 
    if channel_padding_size > 0:
        pades = channel_padding_size // 2
        identity_out = tf.pad(identity_out, [[0, 0], [0, 0], [0, 0], [pades, pades]])
    
    # Merge Output
    block_output = layer_out + identity_out
    
    # Validate outputs
    # 1. Validate output dimensions
    # validate_tensor_dim(block_output, (FLAGS.Train_Batch_Size, block_output_size, block_output_size, block_output_channel))
    
    return block_output


def residual_section(input_layer, input_dim, output_dim, filter_dims, repeat_times, reuse):
    block_input_size = input_dim[0]
    block_input_channel = input_dim[1]
    block_output_size = output_dim[0]
    block_output_channel = output_dim[1]
    
    # Validate inputs
    # 1. Validate input dimensions.
    # 2. Validate block_output_size = block_input_size // 2 OR block_output_size = block_input_size
    # validate_tensor_dim(input_layer, (FLAGS.Train_Batch_Size, block_input_size, block_input_size, block_input_channel))
    
    sec_input_dim = input_dim
    sec_output_dim = output_dim
    
    block_out = input_layer
    for rp in range(repeat_times):
        with tf.variable_scope('rpt_%d' % rp, reuse=reuse):
            block_out = residual_block(block_out, sec_input_dim, sec_output_dim, filter_dims, reuse)
            sec_input_dim = sec_output_dim
            sec_output_dim = sec_output_dim
    
    # Validate outputs
    # 1. Validate output dimensions.
    # validate_tensor_dim(block_out, (FLAGS.Train_Batch_Size, block_output_size, block_output_size, block_output_channel))
    
    return block_out


def conv1_section(input_layer, input_dim, output_dim, filter_dim):
    block_input_size = input_dim[0]
    block_input_channel = input_dim[1]
    block_output_size = output_dim[0]
    block_output_channel = output_dim[1]
    
    # validate_tensor_dim(input_layer, (FLAGS.Train_Batch_Size, block_input_size, block_input_size, block_input_channel))
    
    sec_out = sandwich_conv_bn_relu_layer(input_layer, 
                  [filter_dim[0], filter_dim[0], block_input_channel, block_output_channel], 1)
    
    # validate_tensor_dim(sec_out, (FLAGS.Train_Batch_Size, block_output_size, block_output_size, block_output_channel))
    
    return sec_out


def fc_section(input_layer, num_labels):
    # Global Average Pooling
    in_channel = input_layer.get_shape().as_list()[-1]
    bn_layer = single_bn_layer(input_layer, in_channel)
    relu_layer = single_relu_layer(bn_layer)
    global_pool = tf.reduce_mean(relu_layer, [1, 2])
    
    # Get wx
    input_dim = global_pool.get_shape().as_list()[-1]
    initializer = tf.uniform_unit_scaling_initializer(factor=1.0)
    regularizer = tf.contrib.layers.l2_regularizer(scale=FLAGS.Weight_Decay)
    shape = [input_dim, num_labels]
    fc_w = tf.get_variable('fc_weights', shape=shape, initializer=initializer, regularizer=regularizer)

    # Get bx
    initializer = tf.zeros_initializer()
    regularizer = tf.contrib.layers.l2_regularizer(scale=FLAGS.Weight_Decay)
    shape = [num_labels]
    fc_b = tf.get_variable('fc_bias', shape=shape, initializer=initializer, regularizer=regularizer)

    # Fc layer
    fc_h = tf.matmul(global_pool, fc_w) + fc_b    
    
    return fc_h


def forward(input_tensor_batch, reuse=False):   
    layers = []
    sec_out = input_tensor_batch
    
    # Section 1: Conv1
    input_dim = (64, 3)
    output_dim = (64, 64)
    with tf.variable_scope('conv1', reuse=reuse):
        sec_out = conv1_section(sec_out, input_dim, output_dim, (1, 64))
    
    # Section 2: Conv2_x
    input_dim = (64, 64)
    output_dim = (64, 64)
    with tf.variable_scope('conv2', reuse=reuse):
        sec_out = residual_section(sec_out, input_dim, output_dim, filter_dims[0][0], filter_dims[0][1], reuse=reuse)
    
    # Section 3: Conv3_x
    input_dim = (64, 64)
    output_dim = (32, 128)
    with tf.variable_scope('conv3', reuse=reuse):
        sec_out = residual_section(sec_out, input_dim, output_dim, filter_dims[1][0], filter_dims[1][1], reuse=reuse)
    
    # Section 4: Conv4_x
    input_dim = (32, 128)
    output_dim = (16, 256)
    with tf.variable_scope('conv4', reuse=reuse):
        sec_out = residual_section(sec_out, input_dim, output_dim, filter_dims[2][0], filter_dims[2][1], reuse=reuse)
    
    # Section 5: Conv5_x
    input_dim = (16, 256)
    output_dim = (8, 512)
    with tf.variable_scope('conv5', reuse=reuse):
        sec_out = residual_section(sec_out, input_dim, output_dim, filter_dims[3][0], filter_dims[3][1], reuse=reuse)

    # Section 2: FC
    with tf.variable_scope('fc', reuse=reuse):
        sec_out = fc_section(sec_out, 200)

    return sec_out


def test_graph(train_dir='logs'):
    '''
    Run this function to look at the graph structure on tensorboard. A fast way!
    :param train_dir:
    '''
    input_tensor = tf.constant(np.ones([128, 32, 32, 3]), dtype=tf.float32)
    result = inference(input_tensor, 2, reuse=False)
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    summary_writer = tf.train.SummaryWriter(train_dir, sess.graph)