import tensorflow as tf
import utils
import resnet


def main():
    data = utils.load_tiny_imagenet('./data/tiny-imagenet-200')
    print(data['X_train'].shape)
    print(data['y_train'].shape)
    print(data['X_val'].shape)
    print(data['y_val'].shape)
    print(data['X_test'].shape)
    # print(data['y_test'].shape) Since y_test is None
    print(len(data['class_names']))
    print(data['mean_image'].shape)
    
    batch_size = 256
    
    with tf.Session() as sess:      
        image_placeholder = tf.placeholder(dtype=tf.float32, shape=[batch_size, 64, 64, 3])
        resnet_out = resnet.inference(image_placeholder)
        
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        
        values =  {
            image_placeholder: data['X_train'][0:batch_size]
        }
        resnet_out_val = sess.run(resnet_out, feed_dict=values)
        print (resnet_out_val.shape)
        print (resnet_out_val[0])
        
        total_parameters = 0
        for variable in tf.trainable_variables():
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            print(shape)
            print(len(shape))
            variable_parametes = 1
            for dim in shape:
                print(dim)
                variable_parametes *= dim.value
            print(variable_parametes)
            total_parameters += variable_parametes
        print(total_parameters)

if __name__ == "__main__":
    main()
