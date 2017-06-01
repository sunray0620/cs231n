from resnet import *
from flags import *
from datetime import datetime
import time
import utils

class ResnetTrainer(object):
    '''
    This Object is responsible for all the training and validation process
    '''
    def __init__(self):
        self.init_train_placeholders()
        self.init_val_placeholders()
        self.init_test_placeholders()

    '''
    #######################
    Initialize placeholders
    #######################
    '''
    def init_train_placeholders(self):
        train_phshape = [FLAGS.Train_Batch_Size, FLAGS.IMG_SIZE, FLAGS.IMG_SIZE, FLAGS.IMG_CHANNEL]
        self.train_image_placeholder = tf.placeholder(dtype=tf.float32, shape=train_phshape)
        self.train_label_placeholder = tf.placeholder(dtype=tf.int32, shape=[FLAGS.Train_Batch_Size])        
        self.lr_placeholder = tf.placeholder(dtype=tf.float32, shape=[])

        
    def init_val_placeholders(self):
        val_phshape = [FLAGS.Val_Batch_Size, FLAGS.IMG_SIZE, FLAGS.IMG_SIZE, FLAGS.IMG_CHANNEL]
        self.val_image_placeholder = tf.placeholder(dtype=tf.float32, shape=val_phshape)
        self.val_label_placeholder = tf.placeholder(dtype=tf.int32, shape=[FLAGS.Val_Batch_Size])
    
    
    def init_test_placeholders(self):
        # Just one batch size
        val_phshape = [FLAGS.Test_Batch_Size, FLAGS.IMG_SIZE, FLAGS.IMG_SIZE, FLAGS.IMG_CHANNEL]
        self.test_image_placeholder = tf.placeholder(dtype=tf.float32, shape=val_phshape)
                
    
    '''
    #######################
    Build networks.
    #######################
    '''
    def build_train_resnet(self):
        global_step = tf.Variable(1, trainable=False)
        
        train_resnet = Resnet(use_dropout=True, reuse_variables=False)
        train_logits = train_resnet.forward(self.train_image_placeholder)
        self.train_softmax_loss = utils.softmax_loss(train_logits, self.train_label_placeholder)
        self.train_reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.train_loss = tf.add_n([self.train_softmax_loss] + self.train_reg_losses)

        train_predictions = tf.nn.softmax(train_logits)
        self.train_top1_error = utils.top_k_error(train_predictions, self.train_label_placeholder, 1)
        self.train_op = self.update_variables(self.train_loss, global_step)
        
        # Add some summaries
        train_summary_loss = tf.summary.scalar('train_loss', self.train_loss)
        train_summary_error = tf.summary.scalar('train_top1_error', self.train_top1_error)
        self.train_summary = tf.summary.merge([train_summary_loss, train_summary_error])
    
    
    def build_val_resnet(self):
        val_resnet = Resnet(use_dropout=False, reuse_variables=True)
        val_logits = val_resnet.forward(self.val_image_placeholder)
        self.val_softmax_loss = utils.softmax_loss(val_logits, self.val_label_placeholder)
        
        val_predictions = tf.nn.softmax(val_logits)
        self.val_top1_error = utils.top_k_error(val_predictions, self.val_label_placeholder, 1)
        
        # Add some summaries       
        val_summary_loss = tf.summary.scalar('val_loss', self.val_softmax_loss)
        val_summary_error = tf.summary.scalar('val_top1_error', self.val_top1_error)
        self.val_summary = tf.summary.merge([val_summary_loss, val_summary_error])

    
    def build_test_resnet(self):       
        test_resnet = Resnet(use_dropout=False, reuse_variables=True)
        test_logits = test_resnet.forward(self.test_image_placeholder)      
        self.test_predictions = tf.nn.softmax(test_logits)

    
    def update_variables(self, total_loss, global_step):
        opt = tf.train.MomentumOptimizer(self.lr_placeholder, 0.9)
        # opt = tf.train.AdamOptimizer(learning_rate=0.01)
        train_op = opt.minimize(total_loss, global_step=global_step)
        return train_op

    '''
    #######################
    Build networks.
    #######################
    '''
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
        self.build_test_resnet()
                
        # Initialize or load a session
        saver = tf.train.Saver(max_to_keep=10)
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
            
            # Conduct one training step
            self.one_train_op(sess, data['X_train'], data['y_train'], FLAGS.Learning_Rate, step, summary_writer)
            
            # Conduct one partial validation step
            if step % FLAGS.Part_Val_Freq == 0:
                self.one_partial_val_op(sess, data['X_val'], data['y_val'], step, summary_writer)
            
            if step % FLAGS.Full_Val_Freq == 0:
                self.one_full_val_op(sess, data['X_val'], data['y_val'], data['label_to_wnid'], step)
                
            # Save Checkpoint.
            if step % FLAGS.CKPT_FREQ == 0:
                utils.save_sess(saver, sess, step)
                
    
    '''
    ##############################################
    Helper functions that run operations.
    ##############################################
    '''
    def one_train_op(self, sess, X_train, y_train, lr, step, sw):
        start_time = time.time()            
        X_train_batch, y_train_batch = utils.sample_batch(X_train, y_train, FLAGS.Train_Batch_Size, aug=True)
        
        if FLAGS.Verbose_Mode:
            print("Use learning rate {0}".format(lr))
            
        (_,
         train_loss_value, 
         train_error_value,
         train_summary_value) = sess.run([self.train_op, 
                                         self.train_loss, 
                                         self.train_top1_error,
                                         self.train_summary],
                                feed_dict = { self.train_image_placeholder: X_train_batch,
                                              self.train_label_placeholder: y_train_batch,
                                              self.lr_placeholder: lr })
        sw.add_summary(train_summary_value, step)
        end_time = time.time()

        print("Train loss {0:.4f}, Error {1:.4f} ({2:.2f} secs)".format(train_loss_value, 
                                                                        train_error_value, 
                                                                        end_time-start_time))

        
    def one_partial_val_op(self, sess, X_val, y_val, step, sw):
        start_time = time.time()
        X_val_batch, y_val_batch = utils.sample_batch(X_val, y_val, FLAGS.Val_Batch_Size, aug=False)
                        
        (val_loss_value, 
         val_error_value, 
         val_summary_value) = sess.run([self.val_softmax_loss,
                                        self.val_top1_error,
                                        self.val_summary],
                             feed_dict = { self.val_image_placeholder: X_val_batch,
                                           self.val_label_placeholder: y_val_batch })
        sw.add_summary(val_summary_value, step)
        end_time = time.time()
        
        print("Val loss [{0:.4f}], Error [{1:.4f}] ({2:.2f} secs)".format(val_loss_value, 
                                                                          val_error_value, 
                                                                          end_time-start_time))
        
        
    def one_full_val_op(self, sess, X_test, y_test, label_to_wnid, step):
        print("Doing Full Validation....")
        start_time = time.time()
        
        batch_size = FLAGS.Test_Batch_Size
        batch_count = FLAGS.Test_Batch_Count
        num_input = batch_size * batch_count
        
        predictions = []
        for btch in range(batch_count):
            # print("Process batch {0}".format(btch))
            skip_count = btch * batch_size
            batch_input = X_test[skip_count:skip_count + batch_size]
            
            batch_test_scores = sess.run(self.test_predictions,
                                        feed_dict = { self.test_image_placeholder: batch_input })
            
            batch_test_prediction = np.argmax(batch_test_scores, axis=1)
            predictions.extend(batch_test_prediction)
        
        end_time = time.time()
        
        actual_output = [label_to_wnid[pred] for pred in predictions]
        expected_output = [label_to_wnid[y] for y in y_test]
        num_correct = sum([actual_output[i] == expected_output[i] for i in range(num_input)])
        
        print("Full Validation took {0:.2f} secs".format(end_time-start_time))
        print("Full Validation accuracy: {0:.4f}".format(num_correct/num_input))
        
        f=open("full_val.txt", "a+")
        print("{0}\t{1}".format(step, num_correct), file=f)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        