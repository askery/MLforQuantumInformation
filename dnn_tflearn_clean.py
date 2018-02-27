# -*- coding: utf-8 -*-

""" 
Adptation of MNIST tflearn to binary classification
"""
from __future__ import division, print_function, absolute_import

import tflearn
import tensorflow as tf
tf.reset_default_graph()
import numpy as np

from sklearn.model_selection import train_test_split,cross_val_score,StratifiedKFold,cross_val_predict
#from sklearn.metrics import accuracy_score,confusion_matrix, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
#from sklearn.decomposition import KernelPCA

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


# Data loading and preprocessing
#import tflearn.datasets.mnist as mnist
#X, Y, testX, testY = mnist.load_data(one_hot=True)

# split Xdata and ydata in train and test sets
X, testX, Y, testY = train_test_split(Xdata, ydata, test_size=0.20, random_state=42)

# PCA
#pca = PCA(n_components=nfeat )
pca = PCA(n_components=nfeat )
pca.fit(X)
X =  pca.transform(X)
testX =  pca.transform(testX)

# Building deep neural network
input_layer = tflearn.input_data(shape=[None, 32])
dense1 = tflearn.fully_connected(input_layer, 64, activation='tanh',
                                 regularizer='L2', weight_decay=0.001)
dropout1 = tflearn.dropout(dense1, 0.8)
dense2 = tflearn.fully_connected(dropout1, 64, activation='tanh',
                                 regularizer='L2', weight_decay=0.001)
dropout2 = tflearn.dropout(dense2, 0.8)
softmax = tflearn.fully_connected(dropout2, 2, activation='softmax')

# Regression using SGD with learning rate decay and Top-3 accuracy
sgd = tflearn.SGD(learning_rate=0.1, lr_decay=0.96, decay_step=1000)
#top_k = tflearn.metrics.Top_k(3)
net = tflearn.regression(softmax, optimizer=sgd,
                         loss='categorical_crossentropy')

# Training
model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(X, Y, n_epoch=20, validation_set=(testX, testY),
          show_metric=True, run_id="dense_model")
