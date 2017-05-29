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
        self.cur_step = 1

        
    def init_placeholders(self):
        train_phshape = [FLAGS.Train_Batch_Size, FLAGS.IMG_SIZE, FLAGS.IMG_SIZE, FLAGS.IMG_CHANNEL]
        self.train_image_placeholder = tf.placeholder(dtype=tf.float32, shape=train_phshape)
        self.train_label_placeholder = tf.placeholder(dtype=tf.int32, shape=[FLAGS.Train_Batch_Size])
              
        val_phshape = [FLAGS.Val_Batch_Size, FLAGS.IMG_SIZE, FLAGS.IMG_SIZE, FLAGS.IMG_CHANNEL]
        self.val_image_placeholder = tf.placeholder(dtype=tf.float32, shape=val_phshape)
        self.val_label_placeholder = tf.placeholder(dtype=tf.int32, shape=[FLAGS.Val_Batch_Size])
        
        self.lr_placeholder = tf.placeholder(dtype=tf.float32, shape=[])

    
    def update_variables(self, total_loss, global_step):
        opt = tf.train.MomentumOptimizer(self.lr_placeholder, 0.9)
        # opt = tf.train.AdamOptimizer(learning_rate=0.01)
        train_op = opt.minimize(total_loss, global_step=global_step)
        return train_op
        
    
    def build_train_resnet(self):
        global_step = tf.Variable(1, trainable=False)
                
        train_logits = forward(self.train_image_placeholder, reuse=False)
        self.train_softmax_loss = utils.softmax_loss(train_logits, self.train_label_placeholder)
        self.train_reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.train_loss = tf.add_n([self.train_softmax_loss] + self.train_reg_losses)

        train_predictions = tf.nn.softmax(train_logits)
        self.train_top1_error = utils.top_k_error(train_predictions, self.train_label_placeholder, 1)
        self.train_op = self.update_variables(self.train_loss, global_step)
        
        # Add some summaries
        train_summary_lr = tf.summary.scalar('learning_rate', self.lr_placeholder)
        train_summary_loss = tf.summary.scalar('train_loss', self.train_loss)
        train_summary_error = tf.summary.scalar('train_top1_error', self.train_top1_error)
        self.train_summary = tf.summary.merge([train_summary_lr, train_summary_loss, train_summary_error])
    
    def build_val_resnet(self):
        val_logits = forward(self.val_image_placeholder, reuse=True)
        self.val_softmax_loss = utils.softmax_loss(val_logits, self.val_label_placeholder)
        
        val_predictions = tf.nn.softmax(val_logits)
        self.val_top1_error = utils.top_k_error(val_predictions, self.val_label_placeholder, 1)
        
        # Add some summaries       
        val_summary_loss = tf.summary.scalar('val_loss', self.val_softmax_loss)
        val_summary_error = tf.summary.scalar('val_top1_error', self.val_top1_error)
        self.val_summary = tf.summary.merge([val_summary_loss, val_summary_error])
   

    def get_lr(self, step):
        if step > 60000:
            return 0.001
        elif step > 40000:
            return 0.01
        else:
            return 0.1
        
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
        self.build_train_resnet()
        self.build_val_resnet()
                
        # Initialize or load a session
        saver = tf.train.Saver(tf.global_variables())
        sess = utils.get_sess(saver);
        summary_writer = tf.summary.FileWriter(FLAGS.LOG_PATH, sess.graph)
        
        # Restore global step.
        start_step = 1
        if FLAGS.USE_CKPT is True:
            start_step = FLAGS.CUR_STEP + 1
        print("Restoring global step to {0}".format(start_step))
        
        # Start Training.
        print('Start training...')
        print('----------------------------')
        for step in range(start_step, FLAGS.Train_Steps + 10):
            print('---- step: %d -----' % step)
            
            start_time = time.time()            
            X_train, y_train = utils.sample_batch(data['X_train'], data['y_train'], FLAGS.Train_Batch_Size, aug=True)                
            end_time = time.time()
            if FLAGS.Verbose_Mode:
                print("Read data took {0:.2f} secs".format(end_time-start_time))
            
            start_time = time.time()
            cur_lr = self.get_lr(step)
            (_,
             train_loss_value, 
             train_error_value,
             train_summary_value) = sess.run([self.train_op, 
                                        self.train_loss, 
                                        self.train_top1_error,
                                        self.train_summary],
                                    feed_dict = { self.train_image_placeholder: X_train,
                                                  self.train_label_placeholder: y_train,
                                                  self.lr_placeholder: cur_lr })
            summary_writer.add_summary(train_summary_value, step)
            end_time = time.time()
            if FLAGS.Verbose_Mode:
                print("Learning took {0:.2f} secs".format(end_time-start_time))
            print("Train loss {0:.4f}, Error {1:.4f}".format(train_loss_value, train_error_value))
            
            if step % FLAGS.Val_Freq == 0:
                # Get Validation Performance.
                start_time = time.time()
                X_val, y_val = utils.sample_batch(data['X_val'], data['y_val'], FLAGS.Val_Batch_Size, aug=False)
                                
                (val_loss_value, 
                 val_error_value, 
                 val_summary_value) = sess.run([self.val_softmax_loss,
                                          self.val_top1_error,
                                          self.val_summary],
                                     feed_dict = { self.val_image_placeholder: X_val,
                                                   self.val_label_placeholder: y_val })
                summary_writer.add_summary(val_summary_value, step)
                print("Val loss {0:.4f}, Error {1:.4f}".format(val_loss_value, val_error_value))
                end_time = time.time()
                if FLAGS.Verbose_Mode:
                    print("Validation took {0:.2f} secs".format(end_time-start_time))
                
                
            if step % FLAGS.CKPT_FREQ == 0:
                # Save Checkpoint.
                utils.save_sess(saver, sess, step)
                
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        