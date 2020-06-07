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
