#! usr/bin/env python
# coding:utf-8
#=====================================================
# Copyright (C) 2020 * Ltd. All rights reserved.
#
# Author      : Chen_Sheng19
# Editor      : VIM
# Create time : 2020-06-17
# File name   : 
# Description : loading and fine-tune model 
#
#=====================================================
import sys
nets_path = r"slim"
if nets_path not in sys.path:
    sys.path.insert(0,nets_path)
else:
    print("already add slim")

import tensorflow as tf
from nets.nasnet import nasnet
slim = tf.contrib.slim
import os

mydataset = __import__("mydataset")
create_dataset_fromdir = mydataset.create_dataset_fromdir

class MyNasNetModel(object):

    def __init__(self,model_path=""):
        self.model_path = model_path

    def MyNasNet(self,images,training):
        arg_scope = nasnet.nasnet_mobile_arg_scope()
        with slim.arg_scope(arg_scope):
            logits,end_points = nasnet.build_nasnet_mobile(
                    images,num_classes = self.num_classes+1,is_training=training)
        global_step = tf.train.get_or_create_global_step()
        return logits,end_points,global_step

    def FineTuneNasNet(self,training):
        model_path = self.model_path

        exclude = ['final_layer','aux_7']
        variables_to_restore = slim.get_variables_to_restore(exclude=exclude)
        if training:
            init_fn = slim.assign_from_checkpoint_fn(model_path,variables_to_restore,ignore_missing_vars=True)
        else:
            init_fn = None

        tuning_variables = []
        for v in exclude:
            tuning_variables += slim.get_variables(v)

        print("final_layer:",slim.get_variables('final_layer'))
        print("aux_7:",slim.get_variables('aux_7'))
        print("tuning_variables:",tuning_variables)

        return init_fn,tuning_variables

    def build_acc_base(self,labels):
        self.prediction = tf.cast(tf.argmax(self.logits,1),tf.int32)
        self.correct_prediction = tf.equal(self.prediction,labels)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction,tf.float32))
        self.accuracy_top_5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(predictions=self.logits,targets=labels,k=5),tf.float32))

    def load_cpk(self,global_step,sess,begin=0,saver=None,save_path=None):
        if begin == 0:
            save_path = r'./train_nasnet'
            if not os.path.exists(save_path):
                print("there is not model path:",save_path)
            saver = tf.train.Saver(max_to_keep=1)
            return saver,save_path
        else:
            kpt = tf.train.latest_checkpoint(save_path)
            print("load model:",kpt)
            startepo = 0
            if kpt != None:
                saver.restore(sess,kpt)
                ind = kpt.find("-")
                startepo = int(kpt[ind+1:])
                print("global_step=",global_step.eval(),startepo)
            return startepo

    def build_model_train(self,iamges,labels,learning_rate1,learning_rate2,training):
        self.logits,self.end_points,self.global_step = self.MyNasNet(images,training=training)
        self.step_init = self.global_step.initializer
        self.init_fn,self.tuning_variables = self.FineTuneNasNet(training=training)

        tf.losses.sparse_softmax_cross_entropy(labels=labels,logits=self.logits)
        loss = tf.losses.get_total_loss()

        learning_rate1 = tf.train.exponential_decay(learning_rate=learning_rate1,
                global_step=self.global_step,
                decay_steps=100,decay_rate=0.5)
        learning_rate2 = tf.train.exponential_decay(learning_rate=learning_rate2,
                global_step=self.global_step,
                decay_steps=100,decay_rate=0.2)

        last_optimizer = tf.train.AdamOPtimizer(learning_rate1)
        full_optimizer = tf.train.AdamOPtimizer(learning_rate2)

        updates_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(updates_ops):
            self.last_train_op = last_optimizer.minimize(loss,self.global_step,var_list=self.tuning_variables)
            self.full_train_op = full_optimizer.minimize(loss,self.global_step)

        self.build_acc_base(labels)

        tf.summary.scalar('accuracy',self.accuracy)
        tf.summary.scalar('accuracy_top_5',self.accuracy_top_5)

        self.merged = tf.summary.merge_all()

        self.train_writer = tf.summary.FileWriter('./log_dir/train')
        self.eval_writer = tf.summary.FileWriter('./log_dir/eval')

        self.saver,self.save_path = self.load_cpk(self.global_step,None)

    def build_model(self,mode='train',testdata_dir='./data/val',traindata_dir='./data/train',batch_size=32,learning_rate1=0.001,learning_rate2=0.001):

        if mode == 'train':
            tf.reset_default_graph()

            dataset,self.num_classes = create_dataset_fromdir(traindata_dir,batch_size)
            testdataset,_ = create_dataset_fromdir(testdata_dir,batch_size,training=False)

            iterator = tf.data.Iterator.from_structure(dataset.output_types,dataset.output_shapes)
            images,labels = iterator.get_next()
            iterator.make_initializer

            self.train_init_op = iterator.make_initializer(dataset)
            self.test_init_op = iterator.make_initializer(testdataset)

            self.build_model_train(images,labels,learning_rate1,learning_rate2,True)
            self.global_init = tf.global_variables_initializer()
            tf.get_default_graph().finalize()

        elif mode == 'test':
            tf.reset_default_graph()

            testdataset,self.num_classes = create_dataset_fromdir(testdata_dir,batch_size,training=False)

            iterator = tf.data.Iterator.from_structure(testdataset.output_types,testdataset.output_shapes)
            self.images,labels = iterator.get_next()

            self.test_init_op = iterator.make_initializer(testdataset)
            self.logits,self.end_points,self.global_step = self.MyNasNet(self.images,training=False)
            self.saver,self.save_path = self.load_cpk(self.global_step,None)

            self.build_acc_base(labels)
            tf.get_default_graph().finalize()

        elif mode == 'eval':
            tf.reset_default_graph()
            testdataset,self.num_classes = create_dataset_fromdir(testdata_dir,batch_size,training=False)

            iterator = tf.data.from_structure(testdataset.output_types,testdataset.output_shapes)
            self.images,labels = iterator.get_next()

            self.logits,self.end_points,self.global_step = slef.MyNasNet(slef.images,training=False)
            self.saver,self.save_path = self.load_cpk(self.global_step,None)
            tf.get_default_graph().finalize()

