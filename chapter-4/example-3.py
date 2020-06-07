#! usr/bin/env python
# coding:utf-8
#=====================================================
# Copyright (C) 2020 * Ltd. All rights reserved.
#
# Author      : Chen_Sheng19
# Editor      : VIM
# Create time : 2020-06-07
# File name   : 
# Description : producting dataset from images 
#
#=====================================================
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.utils import shuffle

def load_sample(sample_dir):
    """
    Recursive read files,return images path,digit labels and str labels 
    """
    lfilenames,labelsnames = [],[]
    for (dirpath,dirnames,filenames) in os.walk(sample_dir):
        for filename in filenames:
            image_path = os.sep.join([dirpath,filename])
            lfilenames.append(image_path)
            labelsnames.append(dirpath.split('\\')[-1])

    lab = list(sorted(set(labelsnames)))
    labdict = dict(zip(lab,list(range(len(lab)))))
    labels = [labdict[i] for i in labelsnames]

    return shuffle(np.asarray(lfilenames),np.asarray(labels)),np.asarray(lab)

data_dir = "minist_digits_images"
(image,label),labelnames = load_sample(data_dir)
print(len(image),image[:2],len(label),label[:2])
print(labelnames[label[:2]],labelnames)

def get_batches(image,label,resieze_w,resize_h,channels,batch_size):
    """
    1. generate queue
    2. read image and resize image
    3. generate batch image 
    """
    queue = tf.train.slice_input_producer([image,label])
    
    label = queue[1]
    image_c = tf.read_file(queue[0])
    image = tf.image.decode_bmp(image_c,channels)
    image = tf.image.resize_image_with_crop_or_pad(image,resieze_w,resize_h)
    image = tf.image.per_image_standardization(image)

    image_batch,label_batch = tf.train.batch([image,label],batch_size = batch_size,num_threads = 64)
    image_batch = tf.cast(image_batch,tf.float32)

    label_batch = tf.reshape(label_batch,[batch_size])
    return image_batch,label_batch

batch_size = 16
image_batch,label_batch = get_batches(image,label,28,28,1,batch_size)

def show_result(subplot,title,this_img):
    p = plt.subplot(subplot)
    p.axis('off')
    p.imshow(np.reshape(this_img,(28,28)))
    p.set_title(title)

def show_img(index,label,img,ntop):
    plt.figure(figsize=(20,20))
    plt.axis('off')
    ntop = min(ntop,9)
    print(index)
    for i in range(ntop):
        show_result(100+10*ntop+1+i,label[i],img[i])
    plt.show()

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess = sess,coord = coord)

    try:
        for step in np.arange(10):
            if coord.should_stop(): 
                break
            images,label = sess.run([image_batch,label_batch])

            show_img(step,label,images,batch_size)

    except tf.errors.OutOfRangeError:
        print("Done!")
    finally:
        coord.request_stop()

    coord.join(threads)
