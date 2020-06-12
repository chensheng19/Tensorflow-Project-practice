#! usr/bin/env python
# coding:utf-8
#=====================================================
# Copyright (C) 2020 * Ltd. All rights reserved.
#
# Author      : Chen_Sheng19
# Editor      : VIM
# Create time : 2020-06-09
# File name   : 
# Description : product TFRecord data from image file 
#
#=====================================================
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import os
from sklearn.utils import shuffle
from PIL import Image

def load_smaple(sample_dir,shuffle_flag = True):

    print("loading dataset...")
    lfilenames = []
    labelnames = []

    for dirpath,dirnames,filenames in os.walk(sample_dir):
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

dir_path = "man_woman"
(filenames,labels),_ = load_smaple(dir_path,False)

def make_TFRec(filenames,labels):
    #1.创建writer
    writer = tf.python_io.TFRecordWriter("mydata.tfrecords")
    for i in tqdm(range(len(labels))):
        image = Image.open(filenames[i])
        img = image.resize((256,256))
        img_raw = img.tobytes()
        #2.读取到的内容转化为tfrecords格式
        example = tf.train.Example( #example
                  features = tf.train.Features(#features
                      feature = {"label": tf.train.Feature(int64_list = tf.train.Int64List(value = [labels[i]])),
                                 "img_raw": tf.train.Feature(bytes_list = tf.train.BytesList(value = [img_raw]))}))#feature字典
        writer.write(example.SerializeToString())#序列化压缩
    writer.close()

make_TFRec(filenames,labels)

def read_and_decode(filenames,flag="train",batch_size=3):
    #1.读取文件生成队列
    if flag == "train":
        filename_queue = tf.train.string_input_producer(filenames)
    else:
        filename_queue = tf.train.string_input_producer(filenames,num_epochs=1,shuffle=False)
    #2.从队列读取example
    reader = tf.TFRecordReader()
    _,serialized_example = reader.read(filename_queue)
    #3.将example解析为features
    features = tf.parse_single_example(serialized_example,
                                       features = {"label":tf.FixedLenFeature([],tf.int64),
                                                   "img_raw":tf.FixedLenFeature([],tf.string)})
    #4.将features解析为图片数据
    image = tf.decode_raw(features['img_raw'],tf.uint8)
    image = tf.reshape(image,[256,256,3])
    label = tf.cast(features['label'],tf.int32)
    
    if flag == "train":
        image = tf.cast(image,tf.float32) * (1./255) - 0.5
        img_batch,label_batch = tf.train.batch([image,label],batch_size=batch_size,capacity=20)
        return img_batch,label_batch

    return image,label
TFRecordfilnames = ["mydata.tfrecords"]
image,label = read_and_decode(TFRecordfilnames,flag='test')


save_image_path = "show\\"
if tf.gfile.Exists(save_image_path):
    tf.gfile.DeleteRecursively(save_image_path)
tf.gfile.MakeDirs(save_image_path)

with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    myset = set([])
    try:
        i = 0
        while True:
            example,example_label = sess.run([image,label])
            example_label = str(example_label)
            if example_label not in myset:
                myset.add(example_label)
                tf.gfile.MakeDirs(save_image_path+example_label)
            img = Image.fromarray(example,'RGB')
            img.save(save_image_path+example_label+"\\"+str(i)+'_Label_'+'.jpg')
            print(i)
            i += 1
    except tf.errors.OutOfRangeError:
        print('Done Test -- epoch limit reached')
    finally:
        coord.request_stop()
        coord.join(threads)
        print("stop()")
