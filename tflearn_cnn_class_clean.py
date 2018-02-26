#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 11:24:01 2018

@author: askery
"""

from __future__ import division, print_function, absolute_import

from datetime import datetime
start=datetime.now()

import tflearn
import numpy as np
import tensorflow as tf
tf.reset_default_graph()

from sklearn.model_selection import train_test_split,cross_val_score,StratifiedKFold,cross_val_predict
#from sklearn.metrics import accuracy_score,confusion_matrix, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression



# importing #################
#####
file1 = '/home/data/research/IIP/ML/colab/daniel/rawdata/d2_NPT.txt' # negative - N rows of 16 complex number comma separated
file2 = '/home/data/research/IIP/ML/colab/daniel/rawdata/d2_PPT.txt' # positive - M rows of 16 complex number comma separated

data = np.genfromtxt(file1,  delimiter=',',dtype=str)                # load positive
mapping = np.vectorize(lambda t:complex(t.replace('i','j')))         # map to convert to complex
data2 = mapping(data)                                                # create array out of the map
        
pdata = np.genfromtxt(file2,  delimiter=',',dtype=str)               # load negative 
pdata2 = mapping(pdata)                                              # create array out of the map
        

# X,y data
X = np.concatenate((data2.real, data2.imag), axis=1)                 # making a + bj -> a, b for posiive data
Xout = np.concatenate((pdata2.real, pdata2.imag), axis=1)            # making a + bj -> a, b for negative data
Xdata = np.concatenate((X, Xout), axis=0)                            # pasting Xout below X => length now (N + M, 32)

nfeat = len(Xdata[0])                                                               # number of features

yneg = list ( np.ones(len(X)) )                                      # labelling 1 to negative data - length N  
ypos = list ( np.zeros(len(Xout)) )                                  # labelling 0 to negative data - length M
ydata = yneg + ypos                                                  # pasting ypos 'below' yneg    - length N + M
                                                                       
aux = np.zeros((len(ydata),2))                                       # this makes a Hot Enconder for the target values: 
for i in range(len(ydata)):                                          # 0 -> [1 0]
    aux[i][int(ydata[i])] = 1.                                       # 1 -> [0 1] (this is to adapt binary classification to 4D tensors)
ydata = aux

# shuffle the data
shuffle_index = np.random.permutation(len(Xdata))
Xdata, ydata = Xdata[shuffle_index], ydata[shuffle_index]

# split Xdata and ydata in train and test sets
X, testX, Y, testY = train_test_split(Xdata, ydata, test_size=0.20, random_state=42)

# PCA
pca = PCA(n_components=nfeat )
pca.fit(X)
X =  pca.transform(X)
testX =  pca.transform(testX)

# log of PCA components
print("PCs", len(pca.explained_variance_ratio_))
print(pca.explained_variance_ratio_)

# to use tensorflow convolutional NN, we must convert to 4D arrays
# note that 4*8 = 32. Any combination leading to 32 works. 8*4, 1*32 etc
X = X.reshape([-1, 4, 8, 1])
testX = testX.reshape([-1, 4, 8, 1])

# Building convolutional network
network = input_data(shape=[None, 4, 8, 1], name='input')
network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = fully_connected(network, 126, activation='tanh')
network = dropout(network, 0.8)
network = fully_connected(network, 258, activation='tanh')
network = dropout(network, 0.8)
network = fully_connected(network, 2, activation='softmax')
network = regression(network, optimizer='adam', learning_rate=0.01,
                     loss='categorical_crossentropy', name='target')

# Training
model = tflearn.DNN(network, tensorboard_verbose=0)
model.fit({'input': X}, {'target': Y}, n_epoch=20,
           validation_set=({'input': testX}, {'target': testY}),
           snapshot_step=100, show_metric=True, run_id='convnet_mnist')

print ('job duration in s: ', datetime.now() - start)
