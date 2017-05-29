from scipy.misc import imread
from flags import *
import tensorflow as tf
import numpy as np
import os
import cv2


def softmax_loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    return cross_entropy_mean


def top_k_error(predictions, labels, k):
    batch_size = predictions.get_shape().as_list()[0]
    in_top1 = tf.to_float(tf.nn.in_top_k(predictions, labels, k=1))
    num_correct = tf.reduce_sum(in_top1)
    num_error = batch_size - num_correct
    return 1.0 * num_error / batch_size


def randomly_horizontal_flip(image, axis):
    flip_odd = np.random.randint(low=0, high=2)
    if flip_odd == 0:
        image = cv2.flip(image, axis)
    return image


def whitening_image(image):
    mean = np.mean(image)
    num_pixel = FLAGS.IMG_SIZE * FLAGS.IMG_SIZE * FLAGS.IMG_CHANNEL
    std = np.max([np.std(image), 1.0 / np.sqrt(num_pixel)])
    image = (image - mean) / std
    return image


def sample_batch(X, y, batch_size, aug=False):
    num_train, _, _, _ = X.shape
    sampleIndice = np.random.choice(num_train, batch_size)
    X_batch = X[sampleIndice,:]
    y_batch = y[sampleIndice]
    
    if aug:
        for i in range(batch_size):
            # Randomly flip
            randomly_horizontal_flip(X_batch[i], axis=1)
            # Whitening
            whitening_image(X_batch[i])

    return X_batch, y_batch


def save_sess(saver, sess, step):
    print('Saving checkpoints')
    saver.save(sess, FLAGS.CKPT_PATH, global_step=step)


def get_sess(saver):
    sess = tf.Session()
    if FLAGS.USE_CKPT is True:
        print('Restored from checkpoint...')
        file_path = "{0}-{1}".format(FLAGS.CKPT_PATH, FLAGS.CUR_STEP)
        saver.restore(sess, file_path)
    else:
        print('Initialize new Session...')
        init = tf.global_variables_initializer()
        sess.run(init)
    
    return sess


def load_tiny_imagenet(path, dtype=np.float32, subtract_mean=True):
    """
    Load TinyImageNet. Each of TinyImageNet-100-A, TinyImageNet-100-B, and
    TinyImageNet-200 have the same directory structure, so this can be used
    to load any of them.
    Inputs:
    - path: String giving path to the directory to load.
    - dtype: numpy datatype used to load the data.
    - subtract_mean: Whether to subtract the mean training image.
    Returns: A dictionary with the following entries:
    - class_names: A list where class_names[i] is a list of strings giving the
      WordNet names for class i in the loaded dataset.
    - X_train: (N_tr, 3, 64, 64) array of training images
    - y_train: (N_tr,) array of training labels
    - X_val: (N_val, 3, 64, 64) array of validation images
    - y_val: (N_val,) array of validation labels
    - X_test: (N_test, 3, 64, 64) array of testing images.
    - y_test: (N_test,) array of test labels; if test labels are not available
      (such as in student code) then y_test will be None.
    - mean_image: (3, 64, 64) array giving mean training image
    """
    # First load wnids
    with open(os.path.join(path, 'wnids.txt'), 'r') as f:
        wnids = [x.strip() for x in f]

    # Map wnids to integer labels
    wnid_to_label = {wnid: i for i, wnid in enumerate(wnids)}

    # Use words.txt to get names for each class
    with open(os.path.join(path, 'words.txt'), 'r') as f:
        wnid_to_words = dict(line.split('\t') for line in f)
        for wnid, words in wnid_to_words.items():
            wnid_to_words[wnid] = [w.strip() for w in words.split(',')]
    class_names = [wnid_to_words[wnid] for wnid in wnids]

    # Next load training data.
    X_train = []
    y_train = []
    for i, wnid in enumerate(wnids):
        if (i + 1) % 20 == 0:
            print('loading training data for synset %d / %d'
                  % (i + 1, len(wnids)))
        # To figure out the filenames we need to open the boxes file
        boxes_file = os.path.join(path, 'train', wnid, '%s_boxes.txt' % wnid)
        with open(boxes_file, 'r') as f:
            filenames = [x.split('\t')[0] for x in f]
        num_images = len(filenames)

        X_train_block = np.zeros((num_images, 3, 64, 64), dtype=dtype)
        y_train_block = wnid_to_label[wnid] * np.ones(num_images, dtype=np.int64)
        for j, img_file in enumerate(filenames):
            img_file = os.path.join(path, 'train', wnid, 'images', img_file)
            img = imread(img_file)
            if img.ndim == 2:
                # grayscale file
                img.shape = (64, 64, 1)
            X_train_block[j] = img.transpose((2, 0, 1))
        X_train.append(X_train_block)
        y_train.append(y_train_block)

    # We need to concatenate all training data
    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)

    # Next load validation data
    with open(os.path.join(path, 'val', 'val_annotations.txt'), 'r') as f:
        img_files = []
        val_wnids = []
        for line in f:
            img_file, wnid = line.split('\t')[:2]
            img_files.append(img_file)
            val_wnids.append(wnid)
        num_val = len(img_files)
        y_val = np.array([wnid_to_label[wnid] for wnid in val_wnids])
        X_val = np.zeros((num_val, 3, 64, 64), dtype=dtype)
        for i, img_file in enumerate(img_files):
            img_file = os.path.join(path, 'val', 'images', img_file)
            img = imread(img_file)
            if img.ndim == 2:
                img.shape = (64, 64, 1)
            X_val[i] = img.transpose((2, 0, 1))

    # Next load test images
    # Students won't have test labels, so we need to iterate over files in the
    # images directory.
    img_files = os.listdir(os.path.join(path, 'test', 'images'))
    X_test = np.zeros((len(img_files), 3, 64, 64), dtype=dtype)
    for i, img_file in enumerate(img_files):
        img_file = os.path.join(path, 'test', 'images', img_file)
        img = imread(img_file)
        if img.ndim == 2:
            img.shape = (64, 64, 1)
        X_test[i] = img.transpose((2, 0, 1))

    y_test = None
    y_test_file = os.path.join(path, 'test', 'test_annotations.txt')
    if os.path.isfile(y_test_file):
        with open(y_test_file, 'r') as f:
            img_file_to_wnid = {}
            for line in f:
                line = line.split('\t')
                img_file_to_wnid[line[0]] = line[1]
        y_test = [wnid_to_label[img_file_to_wnid[img_file]]
                  for img_file in img_files]
        y_test = np.array(y_test)

    mean_image = X_train.mean(axis=0)
    if subtract_mean:
        X_train -= mean_image[None]
        X_val -= mean_image[None]
        X_test -= mean_image[None]
    
    X_train = X_train.transpose((0, 2, 3, 1))
    X_val = X_val.transpose((0, 2, 3, 1))
    X_test = X_test.transpose((0, 2, 3, 1))
    mean_image = mean_image.transpose((1, 2, 0))
    
    return {
      'X_train': X_train,
      'y_train': y_train,
      'X_val': X_val,
      'y_val': y_val,
      'X_test': X_test,
      'y_test': y_test,
      'class_names': class_names,
      'mean_image': mean_image,
    }
