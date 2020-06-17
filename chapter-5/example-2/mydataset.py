#! usr/bin/env python
# coding:utf-8
#=====================================================
# Copyright (C) 2020 * Ltd. All rights reserved.
#
# Author      : Chen_Sheng19
# Editor      : VIM
# Create time : 2020-06-16
# File name   : 
# Description : preprocessing images and return batch images 
#
#=====================================================
import tensorflow as tf
import sys
nets_path = r"slim"
if nets_path not in sys.path:
    sys.path.insert(0,nets_path)
else:
    print("Already add slim")
from nets.nasnet import nasnet
slim = tf.contrib.slim
image_size = nasnet.build_nasnet_mobile.default_image_size
from preprocessing import preprocessing_factory
import os


def list_images(directory):
    labels = os.listdir(directory)
    labels.sort()
    files_and_labels = []
    for label in labels:
        for f in os.listdir(os.path.join(directory,label)):
            if f.lower().endswith(('.jpg','.png')):
                files_and_labels.append((os.path.join(directory,label,f),label))

    filenames,labels = zip(*files_and_labels)
    filenames = list(filenames)
    labels = list(labels)
    unique_list = list(set(labels))

    label_to_int = {}
    for i,label in enumerate(sorted(unique_list)):
        label_to_int[label] = i + 1
    labels = [label_to_int[i] for i in labels]
    print(labels[:6],labels[-6:])
    return filenames,labels

num_workers = 2

image_preprocessing_fn = preprocessing_factory.get_preprocessing('nasnet_mobile',is_training=True)
image_val_preprocessing_fn = preprocessing_factory.get_preprocessing('nasnet_mobile',is_training=False)

def _parse_fn(filename,label):
    image_string = tf.read_file(filename)
    iamge = tf.image.decode_jpeg(image_string,channels=3)
    return image,label

def train_preprocessing(image,label):
    image = image_preprocessing_fn(image,image_size,image_size)
    return image,label

def val_preprocessing(image,label):
    image = image_eval_preprocessing_fn(image,image_size,image_size)
    return image,label

def create_batch_dataset(filenames,labels,batch_size,training=True):
    dataset = tf.data.Dataset.from_tensor_slices((filenames,labels))
    dataset = dataset.map(_parse_fn,num_parallel_calls=num_workers)

    if training:
        dataset = dataset.shuffle(buffer_size=len(filenames))
        dataset = dataset.map(train_preprocessing,num_parallel_calls=num_workers)
    else:
        dataset = dataset.map(val_preprocessing,num_parallel_calls=num_workers)

    return dataset.batch(batch_size)

def creat_dataset_fromdir(directory,batch_size,training=True):
    filenames,labels = list_images(directory)
    num_classes = len(set(labels))
    dataset = create_batch_dataset(filenames,labels,batch_size,training)
    return dataset,num_classes

