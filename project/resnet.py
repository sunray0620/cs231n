import tensorflow as tf
import numpy as np

from flags import *


class Resnet(object):
    def __init__(self, use_dropout, reuse_variables):
        self.reuse_variables = reuse_variables
        self.is_training = True
        
        self.use_dropout = use_dropout
        self.p_L = 0.5
        self.num_block_activated = tf.constant(0)
        
        # 34 Layer structure.
        self.filter_dim_list = [
                               [((3, 64), (3, 64)), 3], \
                               [((3, 128), (3, 128)), 4], \
                               [((3, 256), (3, 256)), 6], \
                               [((3, 512), (3, 512)), 3]]
        
        '''
        # 50 Layer structure.
        self.filter_dim_list = [
                               [((1, 64), (3, 64), (1, 256)), 3], \
                               [((1, 128), (3, 128), (1, 512)), 4], \
                               [((1, 256), (3, 256), (1, 1024)), 6], \
                               [((1, 512), (3, 512), (1, 2048)), 3]]
        '''
        
        '''
        # 101 Layer structure.
        self.filter_dim_list = [
                               [((1, 64), (3, 64), (1, 256)), 3], \
                               [((1, 128), (3, 128), (1, 512)), 4], \
                               [((1, 256), (3, 256), (1, 1024)), 23], \
                               [((1, 512), (3, 512), (1, 2048)), 3]]
        '''
        
        '''
        # 152 Layer structure.
        self.filter_dim_list = [
                               [((1, 64), (3, 64), (1, 256)), 3], \
                               [((1, 128), (3, 128), (1, 512)), 8], \
                               [((1, 256), (3, 256), (1, 1024)), 36], \
                               [((1, 512), (3, 512), (1, 2048)), 3]]
        '''
        
        self.total_res_blocks, self.total_res_layers = self.count_res_layers()
        print("There are totally {0} res blocks".format(self.total_res_blocks))
        print("There are totally {0} res layers".format(self.total_res_layers))
    
    
    def count_res_layers(self):
        total_res_blocks = 0
        total_res_layers = 0
        for filter_dim in self.filter_dim_list:
            total_res_blocks += filter_dim[1]
            total_res_layers += len(filter_dim[0]) * filter_dim[1]
        return total_res_blocks, total_res_layers
    
    '''
    ####################################
    Helpers
    ####################################
    '''

    def validate_tensor_dim(self, tensor, tensors_dim):
        if FLAGS.Run_Mode != "dev":
            return
        actual_dim = tensor.get_shape().as_list()
        print(actual_dim)
        for i in range(4):
            if actual_dim[i] != tensors_dim[i]:
                raise ValueError('Dimension does not match E[%d] - A[%d]' % (tensors_dim[i], actual_dim[i]))

    
    def create_variables(name, shape, initializer=None, regularizer=None):
        initializer = tf.contrib.layers.xavier_initializer()
        regularizer = tf.contrib.layers.l2_regularizer(scale=FLAGS.weight_decay)
        new_variables = tf.get_variable(name, shape=shape, 
                                        initializer=initializer,
                                        regularizer=regularizer)
        return new_variables
    
    '''
    ####################################
    Single Layer
    ####################################
    '''

    def single_bn_layer(self, x, reuse=None, scope=None):
        bn_layer_out = tf.contrib.layers.batch_norm(x,
                                                    decay=0.99,
                                                    center=True,
                                                    scale=True,
                                                    epsilon=0.001,
                                                    is_training=self.is_training,
                                                    reuse=reuse,
                                                    trainable=True,
                                                    fused=True,
                                                    data_format="NCHW",
                                                    scope=scope)
        return bn_layer_out

    
    def single_conv_layer(self, x, filter_size, out_channel, stride):        
        conv_layer_out = tf.layers.conv2d(x,
                                          filters=out_channel,
                                          kernel_size=(filter_size, filter_size),
                                          strides=(stride, stride),
                                          padding='SAME',
                                          data_format='channels_first',
                                          use_bias=True,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          bias_initializer=tf.zeros_initializer(),
                                          kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=FLAGS.Weight_Decay),
                                          bias_regularizer=None,
                                          reuse=None)
        return conv_layer_out

    
    def single_relu_layer(self, x):
        relu_layer_out = tf.nn.relu(x)
        return relu_layer_out
    
    
    def single_fully_connect(self, x, num_outputs):
        fc_out = tf.contrib.layers.fully_connected(x,
                                                   num_outputs=num_outputs,
                                                   activation_fn=None,
                                                   weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                   weights_regularizer=tf.contrib.layers.l2_regularizer(scale=FLAGS.Weight_Decay),
                                                   biases_initializer=tf.zeros_initializer(),
                                                   biases_regularizer=None,
                                                   reuse=None,
                                                   scope=None)
        return fc_out

    '''
    ####################################
    Sandwich Layer
    ####################################
    '''
    '''
    def sandwich_conv_bn_relu_layer(self, input_layer, filter_shape, stride):
        in_channel = input_layer.get_shape().as_list()[-1]
        out_channel = filter_shape[-1]

        conv_out = self.single_conv_layer(input_layer, shape=filter_shape, stride=stride)
        bn_out = self.single_bn_layer(conv_out, out_channel)
        relu_output = self.single_relu_layer(bn_out)
        return relu_output
    '''
    
    def sandwich_bn_relu_conv_layer(self, x, filter_size, out_channel, stride):
        bn_out = self.single_bn_layer(x)
        relu_out = self.single_relu_layer(bn_out)
        conv_out = self.single_conv_layer(relu_out, filter_size=filter_size, 
                                          out_channel=out_channel, stride=stride)
        return conv_out

    '''
    ####################################
    Block Layer
    ####################################
    '''

    def residual_block(self, input_layer, input_dim, output_dim, filter_dims, sec_num, rpt_num):
        block_input_size = input_dim[0]
        block_input_channel = input_dim[1]
        block_output_size = output_dim[0]
        block_output_channel = output_dim[1]

        # Validate inputs
        # 1. Validate input dimensions.
        # 2. Validate block_output_size = block_input_size // 2 OR block_output_size = block_input_size
        # validate_tensor_dim(input_layer, (FLAGS.Train_Batch_Size, block_input_size, block_input_size, block_input_channel))
        print("Block {0}, rpt {1}, input shape: {2}".format(sec_num, rpt_num, input_layer.get_shape().as_list()))
        
        # Conv Branch
        if block_input_size == block_output_size * 2:
            stride = 2
        else:
            stride = 1

        layer_out = input_layer
        for i in range(len(filter_dims)):
            with tf.variable_scope('block_%d' % i, reuse=self.reuse_variables):
                filter_dim = filter_dims[i]
                filter_size = filter_dim[0]
                out_channel = filter_dim[1]
                layer_out = self.sandwich_bn_relu_conv_layer(layer_out, filter_size, out_channel, stride)
                stride = 1    # Only shrink size for at most once.

        # Identity Branch
        identity_out = input_layer
        # Shrink size if needed.
        if block_input_size == block_output_size * 2:
            identity_out = tf.nn.avg_pool(identity_out, 
                                          ksize=[1, 1, 2, 2], 
                                          strides=[1, 1, 2, 2], 
                                          padding='VALID',
                                          data_format='NCHW')
        # Pad channels if needed.
        channel_padding_size = block_output_channel - block_input_channel 
        if channel_padding_size > 0:
            pades = channel_padding_size // 2
            identity_out = tf.pad(identity_out, [[0, 0], [pades, pades], [0, 0], [0, 0]])

        # Merge Output
        # block_output = layer_out + identity_out

        bl = sum([self.filter_dim_list[i][1] for i in range(sec_num)]) + rpt_num + 1
        p_block_survival = 1 - (bl / self.total_res_blocks) * (1 - self.p_L)
        survival_rate = tf.constant(p_block_survival)
        survival_roll = tf.random_uniform(shape=[], minval=0.0, maxval=1.0)
        block_drop = tf.logical_and(tf.greater(survival_roll, survival_rate), 
                                       tf.constant(self.use_dropout))
        
        block_output = tf.cond(block_drop, lambda: identity_out, lambda: layer_out + identity_out)
        self.num_block_activated = tf.cond(block_drop, lambda: self.num_block_activated, lambda: self.num_block_activated + 1)
        
        # block_output = layer_out + identity_out
        
        # Validate outputs
        # 1. Validate output dimensions
        # validate_tensor_dim(block_output, (FLAGS.Train_Batch_Size, block_output_size, block_output_size, block_output_channel))
        print("\t\toutput shape: {2}".format(sec_num, rpt_num, block_output.get_shape().as_list()))
        
        return block_output


    def residual_section(self, input_layer, input_dim, output_dim, sec_num):
        block_input_size = input_dim[0]
        block_input_channel = input_dim[1]
        block_output_size = output_dim[0]
        block_output_channel = output_dim[1]

        # Validate inputs
        # 1. Validate input dimensions.
        # 2. Validate block_output_size = block_input_size // 2 OR block_output_size = block_input_size
        # self.validate_tensor_dim(input_layer, (FLAGS.Train_Batch_Size, block_input_channel, block_input_size, block_input_size))

        filter_dims = self.filter_dim_list[sec_num][0]
        repeat_times = self.filter_dim_list[sec_num][1]
        
        sec_input_dim = input_dim
        sec_output_dim = output_dim

        block_out = input_layer
        for rp in range(repeat_times):
            with tf.variable_scope('rpt_%d' % rp, reuse=self.reuse_variables):
                block_out = self.residual_block(block_out, 
                                                sec_input_dim, sec_output_dim, 
                                                filter_dims, sec_num, rp)
                sec_input_dim = sec_output_dim
                sec_output_dim = sec_output_dim

        # Validate outputs
        # 1. Validate output dimensions.
        # self.validate_tensor_dim(block_out, (FLAGS.Train_Batch_Size, block_output_channel, block_output_size, block_output_size))

        return block_out


    def conv1_section(self, x, filter_size, out_channel, stride):
        # validate_tensor_dim(input_layer, (FLAGS.Train_Batch_Size, block_input_size, block_input_size, block_input_channel))

        sec_out = self.single_conv_layer(x,
                                         filter_size=filter_size,
                                         out_channel=out_channel,
                                         stride=stride)
        
        # TODO: Add a pooling layer?
        sec_out = tf.layers.max_pooling2d(sec_out, pool_size=2, strides=1, 
                                          padding='SAME', data_format='channels_first')
        
        # validate_tensor_dim(sec_out, (FLAGS.Train_Batch_Size, block_output_size, block_output_size, block_output_channel))

        return sec_out


    def fc_section(self, x, num_labels):
        in_channel = x.get_shape().as_list()[1]
        bn_layer = self.single_bn_layer(x)
        relu_layer = self.single_relu_layer(bn_layer)
        global_pool = tf.reduce_mean(relu_layer, [2, 3])
        fc_out = self.single_fully_connect(global_pool, num_labels)

        return fc_out


    def forward(self, input_tensor_batch):   
        layers = []
        sec_out = input_tensor_batch

        # Section 1: Conv1
        input_dim = (64, 3)
        output_dim = (64, 32)
        with tf.variable_scope('conv1', reuse=self.reuse_variables):
            sec_out = self.conv1_section(sec_out, 1, 32, 1)
        
        # Section 2: Conv2_x
        input_dim = (64, 32)
        output_dim = (64, 64)
        with tf.variable_scope('conv2', reuse=self.reuse_variables):
            sec_out = self.residual_section(sec_out, input_dim, output_dim, sec_num=0)
            
        # Section 3: Conv3_x
        input_dim = (64, 64)
        output_dim = (32, 128)
        with tf.variable_scope('conv3', reuse=self.reuse_variables):
            sec_out = self.residual_section(sec_out, input_dim, output_dim, sec_num=1)
            
        # Section 4: Conv4_x
        input_dim = (32, 128)
        output_dim = (16, 256)
        with tf.variable_scope('conv4', reuse=self.reuse_variables):
            sec_out = self.residual_section(sec_out, input_dim, output_dim, sec_num=2)

        # Section 5: Conv5_x
        input_dim = (16, 256)
        output_dim = (8, 512)
        with tf.variable_scope('conv5', reuse=self.reuse_variables):
            sec_out = self.residual_section(sec_out, input_dim, output_dim, sec_num=3)
        
        # Section 2: FC
        with tf.variable_scope('fc', reuse=self.reuse_variables):
            sec_out = self.fc_section(sec_out, 200)
        
        return sec_out


    def test_graph(self, train_dir='logs'):
        '''
        Run this function to look at the graph structure on tensorboard. A fast way!
        :param train_dir:
        '''
        input_tensor = tf.constant(np.ones([256, 32, 32, 3]), dtype=tf.float32)
        result = inference(input_tensor, 2, reuse=False)
        init = tf.initialize_all_variables()
        sess = tf.Session()
        sess.run(init)
        summary_writer = tf.train.SummaryWriter(train_dir, sess.graph)