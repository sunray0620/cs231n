from datetime import datetime
import time
import utils
import sys
from scipy.misc import imread
from scipy.misc import imsave

import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np

Prob_05 = lambda aug: iaa.Sometimes(0.05, aug)
Prob_10 = lambda aug: iaa.Sometimes(0.1, aug)
Prob_15 = lambda aug: iaa.Sometimes(0.15, aug)
Prob_20 = lambda aug: iaa.Sometimes(0.20, aug)
Prob_25 = lambda aug: iaa.Sometimes(0.25, aug)
Prob_30 = lambda aug: iaa.Sometimes(0.30, aug)
Prob_35 = lambda aug: iaa.Sometimes(0.35, aug)
Prob_40 = lambda aug: iaa.Sometimes(0.40, aug)
Prob_45 = lambda aug: iaa.Sometimes(0.45, aug)
Prob_50 = lambda aug: iaa.Sometimes(0.50, aug)
Prob_55 = lambda aug: iaa.Sometimes(0.55, aug)
Prob_60 = lambda aug: iaa.Sometimes(0.60, aug)
Prob_65 = lambda aug: iaa.Sometimes(0.65, aug)
Prob_70 = lambda aug: iaa.Sometimes(0.70, aug)
Prob_75 = lambda aug: iaa.Sometimes(0.75, aug)
Prob_80 = lambda aug: iaa.Sometimes(0.80, aug)
Prob_85 = lambda aug: iaa.Sometimes(0.85, aug)
Prob_90 = lambda aug: iaa.Sometimes(0.90, aug)
Prob_95 = lambda aug: iaa.Sometimes(0.95, aug)
Prob_100 = lambda aug: iaa.Sometimes(1.0, aug)

def random_aug_images(images):
    start_time = time.time()
    
    print("Images shape {0}".format(images.shape))
    print("Transpose to NHWC")
    images = images.transpose((0, 2, 3, 1))
    
    print("Start Transforming Images")
    '''
    seq = iaa.Sequential([
            iaa.Fliplr(0.5), # horizontally flip 50% of all images
            iaa.Flipud(0.1), # vertically flip 10% of all images
            # rarely(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
            Prob_50(iaa.Crop(percent=(0, 0.1))), # crop images by 0-10% of their height/width
            Prob_10(iaa.GaussianBlur((0, 1.0))), # blur images with a sigma between 0 and 3.0
            # rarely(iaa.AverageBlur(k=(2, 7))), # blur image using local means with kernel sizes between 2 and 7
            # rarely(iaa.MedianBlur(k=(3, 11))), # blur image using local medians with kernel sizes between 2 and 7
            Prob_10(iaa.Sharpen(alpha=(0, 1.0), lightness=(0.5, 1.5))), # sharpen images
            # Prob_20(iaa.Emboss(alpha=(0, 1.0), strength=(0, 0.75))), # emboss images
            # search either for all edges or for directed edges
            # rarely(iaa.Sometimes(0.5,
            #     iaa.EdgeDetect(alpha=(0, 0.7)),
            #     iaa.DirectedEdgeDetect(alpha=(0, 0.7), direction=(0.0, 1.0)),
            # )),
            Prob_20(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5)), # add gaussian noise to images
            Prob_20(iaa.Dropout((0.0, 0.1), per_channel=0.5)), # randomly remove up to 10% of the pixels
            Prob_10(iaa.CoarseDropout((0.0, 0.05), size_percent=(0.02, 0.15), per_channel=False)),
            Prob_10(iaa.Invert(0.25, per_channel=False)), # invert color channels
            # often(iaa.Add((-10, 10), per_channel=0.5)), # change brightness of images (by -10 to 10 of original value)
            Prob_80(iaa.Multiply((0.5, 1.5), per_channel=0.5)), # change brightness of images (50-150% of original value)
            Prob_30(iaa.ContrastNormalization((0.5, 1.5), per_channel=0.5)), # improve or worsen the contrast
            Prob_10(iaa.Grayscale(alpha=(0.0, 1.0))),
            Prob_70(iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                rotate=(-45, 45), # rotate by -45 to +45 degrees
                shear=(-25, 25), # shear by -16 to +16 degrees
                # order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                # cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )),
            # rarely(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)) # move pixels locally around (with random strengths)
        ],
        random_order=True
    )
    '''

    seq = iaa.Sequential(
        [
            # apply the following augmenters to most images
            iaa.Fliplr(0.5), # horizontally flip 50% of all images
            iaa.Flipud(0.05), # vertically flip 5% of all images
            Prob_50(iaa.Crop(percent=(0, 0.1))), # crop images by 0-10% of their height/width
            Prob_70(iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                rotate=(-45, 45), # rotate by -45 to +45 degrees
                shear=(-16, 16), # shear by -16 to +16 degrees
                order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )),
            iaa.SomeOf((0, 1),
                [
                    iaa.Multiply((0.5, 1.5), per_channel=0.5), # change brightness of images (50-150% of original value)
                    iaa.ContrastNormalization((0.5, 1.5), per_channel=0.5), # improve or worsen the contrast
                    Prob_10(iaa.Grayscale(alpha=(0.0, 1.0))),
                ],
                random_order=True
            ),
            iaa.SomeOf((0, 1),
                [
                    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                    iaa.Emboss(alpha=(0, 1.0), strength=(0, 1.0)), # emboss images
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                    iaa.OneOf([
                        iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                        iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                    ]),
                ],
                random_order=True
            )
        ],
        random_order=False
    )
    images_aug = seq.augment_images(images)
    save_images_for_debug(images, images_aug)
    
    print("Transpose to NCHW")
    images_aug = images_aug.transpose((0, 3, 1, 2))
    
    end_time = time.time()
    print("Image aug takes {0} secs".format(end_time - start_time))
    return images_aug


def save_images_for_debug(images, images_aug):
    for i in range(200):
        original_image = images[i]
        path = "./images/image_{0}.JPEG".format(i)    
        imsave(path, original_image)

        aug_image = images_aug[i]
        aug_path = "./images/image_{0}_aug.JPEG".format(i)
        imsave(aug_path, aug_image)

        
def da_demo(image):   
    print("Image shape {0}".format(image.shape))
    image = image.transpose((1, 2, 0))
    print("Image shape {0}".format(image.shape))
    
    imsave("./images/image.JPEG", image)

    #################################
    iaop = iaa.Fliplr(1.0)
    image_aug = iaop.augment_image(image)
    imsave("./images/image_aug_1.JPEG", image_aug)
    
    #################################
    iaop = iaa.Flipud(1.0)
    image_aug = iaop.augment_image(image)
    imsave("./images/image_aug_2.JPEG", image_aug)
    
    #################################
    iaop = iaa.Crop(percent=(0.15, 0.15))
    image_aug = iaop.augment_image(image)
    imsave("./images/image_aug_3.JPEG", image_aug)
        
    #################################
    iaop = iaa.Affine(
                scale={"x": (1.2, 1.2), "y": (1.2, 1.2)}, # scale images to 80-120% of their size, individually per axis
                mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )
    image_aug = iaop.augment_image(image)
    imsave("./images/image_aug_4.JPEG", image_aug)
    
    #################################
    iaop = iaa.Affine(
                translate_percent={"x": (0.2, 0.2), "y": (0.2, 0.2)},
                mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )
    image_aug = iaop.augment_image(image)
    imsave("./images/image_aug_5.JPEG", image_aug)
    
    #################################
    iaop = iaa.Affine(
                rotate=(45, 45),
                mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )
    image_aug = iaop.augment_image(image)
    imsave("./images/image_aug_6.JPEG", image_aug)
    
    #################################
    iaop = iaa.Affine(
                shear=(16, 16),
                mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )
    image_aug = iaop.augment_image(image)
    imsave("./images/image_aug_7.JPEG", image_aug)
    
    #################################
    iaop = iaa.Multiply((1.5, 1.5), per_channel=0)
    image_aug = iaop.augment_image(image)
    imsave("./images/image_aug_8.JPEG", image_aug)
        
    #################################
    iaop = iaa.ContrastNormalization((1.5, 1.5), per_channel=0)
    image_aug = iaop.augment_image(image)
    imsave("./images/image_aug_9.JPEG", image_aug)
    
    #################################
    iaop = iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.1*255), per_channel=0.5)
    image_aug = iaop.augment_image(image)
    imsave("./images/image_aug_10.JPEG", image_aug)
    
    #################################
    iaop = iaa.Dropout((0, 0.1), per_channel=0.5) # randomly remove up to 10% of the pixels
    image_aug = iaop.augment_image(image)
    imsave("./images/image_aug_11.JPEG", image_aug)