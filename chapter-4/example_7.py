#! usr/bin/env python
# coding:utf-8
#=====================================================
# Copyright (C) 2020 * Ltd. All rights reserved.
#
# Author      : Chen_Sheng19
# Editor      : VIM
# Create time : 2020-06-14
# File name   : 
# Description : generate Dataset from image files 
#
#=====================================================
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.utils import shuffle
from PIL import Image

def load_sample(sample_dir,shuffle_flag = True):
    print("loading dataset...")
    lfilenames = []
    labelnames = []

    for dirpath,dirname,filenames in os.walk(sample_dir):
        for filename in filenames:
            filepath = os.sep.join([dirpath,filename])
            lfilenames.append(filepath)
            labelname = dirpath.split("\\")[-1]
            labelnames.append(labelname)

    lab = list(sorted(set(labelnames)))
    labdict = dict(zip(lab,list(range(len(lab)))))
    labels = [labdict[i] for i in labelnames]

    if shuffle_flag:
        return shuffle(np.asarray(lfilenames),np.asarray(labels)),np.asarray(lab)
    else:
        return (np.asarray(lfilenames),np.asarray(labels)),np.asarray(lab)


def distorted_image(image,size,ch = 1,shuffle_flag = False,crop_flag = False,brightness_flag = False,contrast_flag = False):

    distorted_image = tf.image.random_flip_left_right(image)
    
    if crop_flag:
        s = tf.random_uniform((1,2),int(size[0]*0.8),size[0],tf.int32)
        distorted_image = tf.random_crop(distorted_image,[s[0][0],s[0][0],ch])

    distorted_image = tf.image.random_flip_up_down(distorted_image)

    if brightness_flag:
        distorted_image = tf.image.random_brightness(distorted_image,max_delta = 10)

    if contrast_flag:
        distorted_image = tf.image.random_contrast(distorted_image,lower = 0.2,upper = 1.8)

    if shuffle_flag:
        distorted_image = tf.random_shuffle(distorted_image)

    return distorted_image

def _norm_image(image,size,ch = 1,flatten_flag = False):
    norm_image = image / 255.
    if flatten_flag:
        norm_image.reshape(norm_image,[size[0]*size[1]*ch])
    return norm_image

from skimage import transform
def _random_rotated30(image,label):

    def _rotated(image):
        shift_y,shift_x = np.array(image.shape.as_list()[:2],np.float32)/2.
        tf_rotate = transform.SimilarityTransform(rotation = np.deg2rad(30))
        tf_shift = transform.SimilarityTransform(translation = [-shift_x,-shift_y])
        tf_shift_inv,image.size = transform.SimilarityTransform(translation = [shift_x,shift_y]),image.shape
        image_rotated = transform.warp(image,(tf_shift + (tf_rotate + tf_shift_inv)).inverse)
        return image_rotated

    def _roratedwarp():
        image_rotated = tf.py_function(_rotated,[image],[tf.float64])
        return tf.cast(image_rotated,tf.float32)[0]

    a = tf.random_uniform([1],0,2,tf.int32)

    image_decoded = tf.cond(tf.equal(tf.constant(0),a[0]),lambda:image,_roratedwarp)

    return image_decoded,label

def dataset(directory,size,batch_size,random_rotated = False):    
    (filenames,labels),_ = load_sample(directory,False)
    def _parseone(filename,label):
        image_str = tf.read_file(filename)
        image_decoded = tf.image.decode_image(image_str)
        image_decoded.set_shape([None,None,None])
        image_decoded = distorted_image(image_decoded,size)
        image_decoded = tf.image.resize(image_decoded,size)
        image_decoded = _norm_image(image_decoded,size)
        image_decoded = tf.cast(image_decoded,tf.float32)
        label = tf.cast(tf.reshape(label,[]),tf.int32)
        return image_decoded,label

    dataset = tf.data.Dataset.from_tensor_slices((filenames,labels))

    dataset = dataset.map(_parseone)
    
    if random_rotated:
        dataset = dataset.map(_random_rotated30)

    dataset = dataset.batch(batch_size)

    return dataset

def showres(subplot,title,thisimg):
    p = plt.subplot(subplot)
    p.axis("off")
    p.imshow(thisimg)
    p.set_title(title)

def showimg(index,label,img,ntop):
    plt.figure(figsize=(20,10))
    plt.axis("off")
    ntop = min(ntop,9)
    print(index)
    for i in range(ntop):
        showres(100+10*ntop+i+1,label[i],img[i])
    plt.show()

def getOne(dataset):
    iterator = dataset.make_one_shot_iterator()
    elem = iterator.get_next()
    return elem

sample_dir = "man_woman"
size = [96,96]
batch_size = 10
tdataset = dataset(sample_dir,size,batch_size)
tdataset2 = dataset(sample_dir,size,batch_size,True)

print(tdataset.output_types)
print(tdataset.output_shapes)

elem1 = getOne(tdataset)
elem2 = getOne(tdataset2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    try:
        for step in np.arange(1):
            value = sess.run(elem1)
            value2 = sess.run(elem2)

            showimg(step,value1[1],np.asarray(value1[0]*255,np.uint8),10)
            showimg(step,value2[1],np.asarray(value2[0]*255,np.uint8),10)
    except tf.errors.OutOfRangeError:
        print("Done!!")


