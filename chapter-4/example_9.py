#! usr/bin/env python
# coding:utf-8
#=====================================================
# Copyright (C) 2020 * Ltd. All rights reserved.
#
# Author      : Chen_Sheng19
# Editor      : VIM
# Create time : 2020-06-15
# File name   : 
# Description : 
#
#=====================================================
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.utils import shuffle
from PIL import Image

tf.enable_eager_execution()
print("Tensorflow Version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

def dataset(directory,size,batchsize):

    def _parseOne(ser_example):
        features = {
                "img_raw":tf.FixedLenFeature(shape=[],dtype=tf.string),
                "label":tf.FixedLenFeature(shape=[],dtype=tf.int64)}
        parsed_example = tf.parse_single_example(ser_example,features)

        image = tf.decode_raw(parsed_example["img_raw"],out_type=tf.uint8)
        image = tf.reshape(image,size)
        image = tf.cast(image,tf.float32)*(1./255)-0.5

        label = parsed_example["label"]
        label = tf.cast(label,tf.int32)
        label = tf.one_hot(label,depth=2,on_value=1)

        return image,label

    dataset = tf.data.TFRecordDataset(directory)
    dataset = dataset.map(_parseOne)
    dataset = dataset.batch(batchsize)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset

def showRes(subplot,title,thisimg):
    p = plt.subplot(subplot)
    p.axis("off")
    p.imshow(thisimg)
    p.set_title(title)

def showimg(index,label,img,ntop):
    plt.figure(figsize=(20,10))
    plt.axis("off")
    ntop = min(9,ntop)
    print(index)
    for i in range(ntop):
        showRes(100+10*ntop+1+i,label[i],img[i])
    plt.show()


sample_dir = ["mydata.tfrecords"]
size = [256,256,3]
batchsize = 10
tdataset = dataset(sample_dir,size,batchsize)

print(tdataset.output_types)
print(tdataset.output_shapes)


for step,value in enumerate(tdataset):
    showing(step,value[1].numpy(),np.asarray((value[0]+0.5)*255,np.uint8),10)
