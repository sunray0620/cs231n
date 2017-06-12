from resnet import *
from flags import *
from datetime import datetime
import time
import utils
import da

class ResnetTester(object):
    '''
    This Object is responsible for all the training and validation process
    '''
    def __init__(self):
        self.init_placeholders()
        

    def init_placeholders(self):
        test_phshape = [FLAGS.Test_Batch_Size, FLAGS.IMG_CHANNEL, FLAGS.IMG_SIZE, FLAGS.IMG_SIZE]
        self.test_image_placeholder = tf.placeholder(dtype=tf.float32, shape=test_phshape)

        
    def build_test_resnet(self):       
        test_resnet = Resnet(use_dropout=False, reuse_variables=None)
        test_logits = test_resnet.forward(self.test_image_placeholder)      
        self.test_predictions = tf.nn.softmax(test_logits)
    

    def test(self):
        # Read training and validation data.
        data_base = utils.load_tiny_imagenet('./data/tiny-imagenet-200', dtype=np.uint8, subtract_mean=False)
        class_names = data_base['class_names']
        mean_image = data_base['mean_image']
        wnid_to_label = data_base['wnid_to_label']
        label_to_wnid = data_base['label_to_wnid']
        
        data = utils.load_val_test_tiny_imagenet('./data/tiny-imagenet-200', dtype=np.float32, mean_image=data_base['mean_image'])
        X_val = data['X_val']
        y_val = data['y_val']
        X_test = data['X_test']
        
        print(X_val.shape)
        print(len(y_val))
        print(X_test.shape)
        
        print("===")
        # print(data['wnid_to_label'])
        # print(data['label_to_wnid'])
        
        # Build the graph for test
        self.build_test_resnet()
                
        # Initialize or load a session
        saver = tf.train.Saver(tf.global_variables())
        sess = tf.Session()
        print('Read weights from checkpoint...')
        file_path = "{0}-{1}".format("ckpts/model_ckpt.dat", 69999)
        saver.restore(sess, file_path)
        
        # Start Testing.
        print('Start testing...')
        print('----------------------------')
        
        start_time = time.time()
        
        input_data = X_val
        
        num_input = input_data.shape[0]
        batch_size = 200
        batch_count = 50
        assert num_input == batch_size * batch_count
        assert input_data.dtype == np.float32
        
        predictions = []
        for btch in range(batch_count):
            print("Process batch {0}".format(btch))
            skip_count = btch * batch_size
            batch_input = input_data[skip_count:skip_count + batch_size]
            
            batch_test_scores = sess.run(self.test_predictions,
                                        feed_dict = { self.test_image_placeholder: batch_input })
            
            batch_test_prediction = np.argmax(batch_test_scores, axis=1)
            predictions.extend(batch_test_prediction)
        
        end_time = time.time()
        
        actual_output = [label_to_wnid[pred] for pred in predictions]
        words_actual_output = [class_names[pred] for pred in predictions]
        
        '''
        retf = open('sunlei_result.txt', 'w')
        test_image_name = 'test_{0}.JPEG'
        img_files = [test_image_name.format(i) for i in range(10000)]
        
        for i in range(10000):
            # line = "{0}\t{1}\t{2}".format(img_files[i], actual_output[i], words_output[i])
            line = "{0}\t{1}".format(img_files[i], actual_output[i])
            print(line, file=retf)
        '''
        
        expected_output_int = y_val
        expected_output = [label_to_wnid[eoi] for eoi in expected_output_int]
        words_expected_output = [class_names[eoi] for eoi in expected_output_int]
        
        num_correct = sum([actual_output[i] == expected_output[i] for i in range(num_input)])
        print(num_correct)    
        
        retf = open('sunlei_result_error.txt', 'w')
        test_image_name = 'test_{0}.JPEG'
        img_files = [test_image_name.format(i) for i in range(10000)]
        for i in range(10000):
            if actual_output[i] == expected_output[i]:
                continue
            print("{0}\t{1}\t{2}\t{3}\t{4}".format(img_files[i], actual_output[i], words_actual_output[i],
                                                  expected_output[i], words_expected_output[i]), file=retf)
        
        print("Prediction took {0:.2f} secs".format(end_time-start_time))
        print("{0:.4f}".format(num_correct/num_input))


    def image_aug_test(self):
        data = utils.load_val_test_tiny_imagenet('./data/tiny-imagenet-200', dtype=np.uint8, mean_image=None)
        X_val = data['X_val']
        y_val = data['y_val']
        X_test = data['X_test']
        
        print(X_val.shape)
        print(len(y_val))
        print(X_test.shape)
        da.da_demo(X_test[423])

        
###### Main Function ######                
resnet_tester = ResnetTester()
resnet_tester.test()
        
        
        
        
        
        
        
        
        
        
        
        