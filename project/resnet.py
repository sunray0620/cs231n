from resnet_model import *
from flags import *
from datetime import datetime
import time
import utils

class Resnet(object):
    '''
    This Object is responsible for all the training and validation process
    '''
    def __init__(self):
        self.init_placeholders()

        
    def init_placeholders(self):
        phshape = [FLAGS.Train_Batch_Size, FLAGS.IMG_SIZE, FLAGS.IMG_SIZE, FLAGS.IMG_CHANNEL]
        self.train_image_placeholder = tf.placeholder(dtype=tf.float32, shape=phshape)
        self.train_label_placeholder = tf.placeholder(dtype=tf.int32, shape=[FLAGS.Train_Batch_Size])
        
        phshape = [FLAGS.Val_Batch_Size, FLAGS.IMG_SIZE, FLAGS.IMG_SIZE, FLAGS.IMG_CHANNEL]
        self.val_image_placeholder = tf.placeholder(dtype=tf.float32, shape=phshape)
        self.val_label_placeholder = tf.placeholder(dtype=tf.int32, shape=[FLAGS.Val_Batch_Size])
        
        self.lr_placeholder = tf.placeholder(dtype=tf.float32, shape=[])

    
    def update_variables(self, total_loss, global_step):
        # opt = tf.train.AdamOptimizer(learning_rate=self.lr_placeholder, momentum=0.9)
        opt = tf.train.AdamOptimizer(learning_rate=0.01)
        train_op = opt.minimize(total_loss, global_step=global_step)
        return train_op
    
    
    def build_resnet(self):
        global_step = tf.Variable(0, trainable=False)
        validation_step = tf.Variable(0, trainable=False)

        # Train losses
        train_logits = forward(self.train_image_placeholder, reuse=False)
        self.train_softmax_loss = utils.softmax_loss(train_logits, self.train_label_placeholder)
        self.train_reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.train_loss = tf.add_n([self.train_softmax_loss] + self.train_reg_losses)

        train_predictions = tf.nn.softmax(train_logits)
        self.train_top1_error = utils.top_k_error(train_predictions, self.train_label_placeholder, 1)

        # Validation losses
        val_logits = forward(self.val_image_placeholder, reuse=True)
        self.val_softmax_loss = utils.softmax_loss(val_logits, self.val_label_placeholder)
        
        val_predictions = tf.nn.softmax(val_logits)
        self.val_top1_error = utils.top_k_error(val_predictions, self.val_label_placeholder, 1)

        self.train_op = self.update_variables(self.train_loss, global_step)
        
        
    def train(self):
        # Read training and validation data.
        data = utils.load_tiny_imagenet('./data/tiny-imagenet-200')
        print(data['X_train'].shape)
        print(data['y_train'].shape)
        print(data['X_val'].shape)
        print(data['y_val'].shape)
        print(data['X_test'].shape)
        print(len(data['class_names']))
        print(data['mean_image'].shape)
    
        # Build the graph for train and validation
        self.build_resnet()
        
        # Initialize or load a session
        saver = tf.train.Saver(tf.global_variables())
        sess = utils.get_sess(saver);
        
        #
        print('Start training...')
        print('----------------------------')
        for step in range(FLAGS.Train_Steps):
            print('---- step: %d -----' % step)
            
            start_time = time.time()
            
            X_train, y_train = utils.sample_batch(data['X_train'], data['y_train'], FLAGS.Train_Batch_Size, aug=False)
            X_val, y_val = utils.sample_batch(data['X_val'], data['y_val'], FLAGS.Val_Batch_Size, aug=False)
            
            end_time = time.time()
            print("Read data took {0:.2f}".format(end_time-start_time))
            
            start_time = time.time()
            _, train_loss_value, train_error_value, val_loss_val, val_error_val = sess.run([self.train_op, 
                                                                  self.train_loss, 
                                                                  self.train_top1_error,
                                                                  self.val_softmax_loss,
                                                                  self.val_top1_error],
                                { self.train_image_placeholder: X_train,
                                  self.train_label_placeholder: y_train,
                                  self.val_image_placeholder: X_val,
                                  self.val_label_placeholder: y_val,
                                  self.lr_placeholder: FLAGS.Init_lr })
            
            end_time = time.time()
            print("Learning took {0:.2f}".format(end_time-start_time))
            
            print("Train loss {0:.4f}, Error {1:.4f}".format(train_loss_value, train_error_value))
            print("Val loss {0:.4f}, Error {1:.4f}".format(val_loss_val, val_error_val))
            
            
            if step % 1000 == 0:
                utils.save_sess(saver, sess, step)
                      
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        