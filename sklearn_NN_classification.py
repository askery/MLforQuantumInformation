#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 10:40:58 2018

@author: askery
"""

from datetime import datetime
start=datetime.now()

import numpy as np

from sklearn.model_selection import train_test_split,cross_val_score,StratifiedKFold,cross_val_predict
from sklearn.metrics import accuracy_score,confusion_matrix, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# importing #################
#####
file1 = '/home/data/research/IIP/ML/colab/daniel/rawdata/d2_NPT.txt' # negative - N rows of 16 (features) complex number comma separated
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

nfeat = len(Xdata[0])                                                # number of features

yneg = list ( np.ones(len(X)) )                                      # labelling 1 to negative data - length N  
ypos = list ( np.zeros(len(Xout)) )                                  # labelling 0 to negative data - length M
ydata = np.array (yneg + ypos)                                       # pasting ypos 'below' yneg    - length N + M
                                                                       
# shuffle the data
shuffle_index = np.random.permutation(len(Xdata))
Xdata, ydata = Xdata[shuffle_index], ydata[shuffle_index]

# split Xdata and ydata in train and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(Xdata, ydata, test_size=0.20, random_state=42)

# Scaler
scaler = StandardScaler()
scaler.fit(Xtrain)
scaler.fit(Xtest)

# PCA
#pca = PCA(n_components=15 )
pca = PCA(n_components=nfeat )
pca.fit(Xtrain)
Xtrain =  pca.transform(Xtrain)
Xtest =  pca.transform(Xtest)

print("PCs", len(pca.explained_variance_ratio_))
print(pca.explained_variance_ratio_)

# kernel PCA
#lin_pca = KernelPCA(n_components = 23, kernel="linear")
#rbf_pca = KernelPCA(n_components = 23, kernel="rbf", gamma=0.0433)
#sig_pca = KernelPCA(n_components = 23, kernel="sigmoid", gamma=0.001, coef0=1)

#lin_pca.fit(Xtrain)
#Xtrain =  lin_pca.transform(Xtrain)
#Xtest =  lin_pca.transform(Xtest)

# MLP classifier
# play with layers structure here
# ---
nlayers = 5                                                       # number hidden of layers
neurons = [400]                                                     # neurons per layer
#neurons = neurons + neurons[:len(neurons)-1][::-1]
#neurons = list (range(2,301))
layers = tuple ( neurons*nlayers )                                  # FINAL structure of the NN 
# ---
# useful notes about MLP hyperparameters 
# solver: {‘lbfgs’, ‘sgd’, ‘adam’} <> Generally: adam (default) is the best, sgd is faster and lbfgs is too slow 
# activation: {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}, default ‘relu’
clf = MLPClassifier(solver='adam', activation='relu', alpha=1e-5, hidden_layer_sizes = layers, max_iter = 1000, verbose = False, random_state=42)

# fit model with trin data
model = clf.fit(Xtrain, ytrain)

# uncomment if wanna see all the parameters of the model
print(model)

# prediciton for test set
preds = clf.predict(Xtest)

# print the result of the accurary score between preds and real target values
print ('accuracy score: ', accuracy_score(ytest, preds))

# Other metrics
y_train_pred = cross_val_predict(clf, Xtrain, ytrain, cv=3)
print ('confusion matrix on train: ')
print ( confusion_matrix(ytrain, y_train_pred) )

y_test_pred = cross_val_predict(clf, Xtest, ytest, cv=3)
print ('confusion matrix on test: ')
print ( confusion_matrix(ytest, y_test_pred) )


print('precision score :', precision_score(ytrain, y_train_pred) )

print('recall score : ', recall_score(ytrain, y_train_pred) )

print('F1 score : ', f1_score(ytrain, y_train_pred) )


print ('job duration in s: ', datetime.now() - start)
