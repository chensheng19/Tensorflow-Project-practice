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

mydataset = __import__("mydataset") #动态导入mydataset
create_dataset_fromdir = mydataset.create_dataset_fromdir

#构建MyNasNetModel类
#1.定义基本模型
#2.定义微调操作
#3.定义训练相关方法：评估模型的相关结点、载入及生成模型的检查点文件、损失函数及优化器等操作结点
#4.构建模型：用于训练、测试、使用
class MyNasNetModel(object):

    def __init__(self,model_path=""):
        self.model_path = model_path #初始化模型路径（仅在训练模式下有意义）

    def MyNasNet(self,images,training): #定义基本模型
        arg_scope = nasnet.nasnet_mobile_arg_scope() #获取模型命名空间
        with slim.arg_scope(arg_scope): #构建模型
            logits,end_points = nasnet.build_nasnet_mobile(
                    images,num_classes = self.num_classes+1,is_training=training)
        global_step = tf.train.get_or_create_global_step() #定义记录步长的张量
        return logits,end_points,global_step

    def FineTuneNasNet(self,training): #定义微调操作
        model_path = self.model_path

        exclude = ['final_layer','aux_7'] #定义需要微调的超参数
        variables_to_restore = slim.get_variables_to_restore(exclude=exclude) #获取除微调参数以外的参数
        if training:
            init_fn = slim.assign_from_checkpoint_fn(model_path,variables_to_restore,ignore_missing_vars=True) #恢复超参数函数
        else:
            init_fn = None

        tuning_variables = []
        for v in exclude:
            tuning_variables += slim.get_variables(v)

        print("final_layer:",slim.get_variables('final_layer'))
        print("aux_7:",slim.get_variables('aux_7'))
        print("tuning_variables:",tuning_variables)

        return init_fn,tuning_variables
#定义训练相关方法
    def build_acc_base(self,labels):#定义模型评估相关结点
        self.prediction = tf.cast(tf.argmax(self.logits,1),tf.int32) #返回logits张量中最大值的索引作为预测值
        self.correct_prediction = tf.equal(self.prediction,labels) #判断预测结果是否正确
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction,tf.float32)) #计算准确率
        self.accuracy_top_5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(predictions=self.logits,targets=labels,k=5),tf.float32)) #计算top-5 准确率

    def load_cpk(self,global_step,sess,begin=0,saver=None,save_path=None): #存储和导出模型函数
        if begin == 0: #模型存储
            save_path = r'./train_nasnet' #存储路径
            if not os.path.exists(save_path):
                print("there is not model path:",save_path)
            saver = tf.train.Saver(max_to_keep=1) #存储检查点文件，max_to_keep为保存最近检查点文件的数量
            return saver,save_path
        else: #模型导出
            kpt = tf.train.latest_checkpoint(save_path) #加载最近保存的检查点文件
            print("load model:",kpt)
            startepo = 0 #计步
            if kpt != None:
                saver.restore(sess,kpt) #还原模型
                ind = kpt.find("-") #找到 - 第一次出现的索引值
                startepo = int(kpt[ind+1:]) #返回开始步数
                print("global_step=",global_step.eval(),startepo)
            return startepo

    def build_model_train(self,iamges,labels,learning_rate1,learning_rate2,training): #构建训练模型损失函数、优化器等
        self.logits,self.end_points,self.global_step = self.MyNasNet(images,training=training) #获取模型logits、end_ponits、global_step张量
        self.step_init = self.global_step.initializer #初始化global_step
        self.init_fn,self.tuning_variables = self.FineTuneNasNet(training=training) #获取超参数恢复函数、微调参数

        tf.losses.sparse_softmax_cross_entropy(labels=labels,logits=self.logits) #计算交叉熵loss
        loss = tf.losses.get_total_loss() #获取总体loss
        
        #定义微调时衰减学习率
        learning_rate1 = tf.train.exponential_decay(learning_rate=learning_rate1, #decayed_learning_rate=learining_rate*decay_rate^(global_step/decay_steps)  
                global_step=self.global_step,
                decay_steps=100,decay_rate=0.5) 
        #定义联调时衰减学习率
        learning_rate2 = tf.train.exponential_decay(learning_rate=learning_rate2,
                global_step=self.global_step,
                decay_steps=100,decay_rate=0.2)

        last_optimizer = tf.train.AdamOPtimizer(learning_rate1) #定义模型优化器
        full_optimizer = tf.train.AdamOPtimizer(learning_rate2)

        updates_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)#tf.GraphKeys.UPDATE_OPS会保存在训练前需要完成的一些操作
        with tf.control_dependencies(updates_ops):#tf.control_dependencies，该函数保证其辖域中的操作必须要在该函数所传递的参数中的操作完成后再进行，保证batch归一化时移动均值和方差的更新
            self.last_train_op = last_optimizer.minimize(loss,self.global_step,var_list=self.tuning_variables)
            self.full_train_op = full_optimizer.minimize(loss,self.global_step)

        self.build_acc_base(labels) # 定义模型评估指标

        tf.summary.scalar('accuracy',self.accuracy) #日志记录
        tf.summary.scalar('accuracy_top_5',self.accuracy_top_5)

        self.merged = tf.summary.merge_all()#将收集的所有默认图表并合并

        self.train_writer = tf.summary.FileWriter('./log_dir/train') #日志写入
        self.eval_writer = tf.summary.FileWriter('./log_dir/eval')

        self.saver,self.save_path = self.load_cpk(self.global_step,None) #定义要保存到检查点文件中的变量
    #模型构建，并用参数mode来指定模型的使用场景，如训练、推理
    def build_model(self,mode='train',testdata_dir='./data/val',traindata_dir='./data/train',batch_size=32,learning_rate1=0.001,learning_rate2=0.001):

        if mode == 'train':
            tf.reset_default_graph() #用于清除默认图形堆栈并重置全局默认图形
            
            #创建训练、测试的dataset数据集
            dataset,self.num_classes = create_dataset_fromdir(traindata_dir,batch_size)
            testdataset,_ = create_dataset_fromdir(testdata_dir,batch_size,training=False)
            
            #创建一个可初始化的迭代器
            iterator = tf.data.Iterator.from_structure(dataset.output_types,dataset.output_shapes)
            images,labels = iterator.get_next() #读取数据

            self.train_init_op = iterator.make_initializer(dataset)
            self.test_init_op = iterator.make_initializer(testdataset)

            self.build_model_train(images,labels,learning_rate1,learning_rate2,True)
            self.global_init = tf.global_variables_initializer()
            tf.get_default_graph().finalize()#将后续图设为只读模式

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

