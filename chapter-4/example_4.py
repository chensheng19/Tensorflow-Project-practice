#! usr/bin/env python
# coding:utf-8
#=====================================================
# Copyright (C) 2020 * Ltd. All rights reserved.
#
# Author      : Chen_Sheng19
# Editor      : VIM
# Create time : 2020-06-09
# File name   : 
# Description : product Memory object data from excel file 
#
#=====================================================
import tensorflow as tf

def read_data(file_queue):
    
    reader = tf.TextLineReader(skip_header_lines = 1)
    key,value = reader.read(file_queue)
    
    defaults = [[0],[0.],[0.],[0.],[0.],[0]]
    csv_column = tf.decode_csv(value,defaults)

    feature_column = [i for i in csv_column[1:-1]]
    label_column = csv_column[-1]
    return tf.stack(feature_column),label_column

def create_pipeline(filename,batch_size,num_epochs = None):
    
    file_queue = tf.train.string_input_producer([filename],num_epochs = num_epochs)

    feature,label = read_data(file_queue)

    min_after_dequeue = 1000
    capacity = min_after_dequeue + batch_size

    feature_batch,label_batch = tf.train.shuffle_batch([feature,label],
                                batch_size = batch_size,capacity = capacity,
                                min_after_dequeue = min_after_dequeue)

    return feature_batch,label_batch

x_train_batch,y_train_batch = create_pipeline('iris_training.csv',32,100)
x_test,y_test = create_pipeline('iris_test.csv',32)


with tf.Session() as sess:
    init = tf.global_variables_initializer()
    local_init = tf.local_variables_initializer()
    sess.run(init)
    sess.run(local_init)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)

    try:
        while True:
            if coord.should_stop():
                break
            example,label = sess.run([x_train_batch,y_train_batch])

            print("train data:",example)
            print("train label:",label)

    except tf.errors.OutOfRangeError:
        print("Done reading")
        example,label = sess.run([x_test,y_test])
        print("test data:",example)
        print("test label:",label)
    except KeyboardInterrupt:
        print("Program termination!")
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()

