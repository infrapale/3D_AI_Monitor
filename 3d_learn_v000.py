# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 09:49:28 2019

@author: tom_h
"""
# Ensure Python 2/3 compatibility
from __future__ import print_function

# Import the libraries we need
import numpy as np # Numerics

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#https://matplotlib.org/tutorials/introductory/images.html#sphx-glr-tutorials-introductory-images-py

from utils import load_data, plot_confusion_matrix

import os
from skimage.io import imread

######### Main code starts here ###########
    
if __name__ == "__main__":
    print("Koneoppiminen Mikrofan 2018")    
    # 1. load all images into a list X; labels go to a list y
    
    X = []
    y = []
    
    path = "pic_3d"
    print ("Loading...")
    X, y = load_data(path)
    
    img = X[2]
    lum_img = img[:,:,:]
    plt.imshow(lum_img)
    plt.imshow(lum_img,cmap="hot")
    imgplot =plt.imshow(lum_img)
    imgplot.set_cmap('nipy_spectral')
    plt.colorbar()
    
    #plt.hist(lum_img.ravel(), bins=256, range=(0.0,1.0), fc='k', ec='k')
    


    #imgplot =plt.imshow(X[2])
    
    
    # The data is an array of 303x64x64x3 images
    # The input to sklearn methods has to be vectors, so 
    # we average the color channels and vectorize the 64x64 
    # bitmaps.
    
    # Color to grey:
    X = np.mean(X, axis = -1)
    
    # reshape 64x64 -> 4096. Resize allows one dim to be 
    # computed automatically with "-1"
    X = np.reshape(X, (X.shape[0], -1))
    
    class_names = np.unique(y)
    
    # Now time for some machine learning.
    
    # 1. Split to training & testing (80% / 20%)
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y, 
                                                        test_size=0.2,
                                                        random_state=0)
    
    # 2. Train a LogReg classifier:
    #model = RandomForestClassifier(n_estimators = 100)
    #model =LogisticRegression(penalty = 'l1', C=10)
    #GridSearchCV
    #dlib.net   oject detection   FHOG
    model = SVC(C=0.01,kernel = 'linear')
    model.fit(X_train, y_train)
    
    # 3. Estimate accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("Classification Accuracy: {:.2f} %".format(100*accuracy))

    # Plot non-normalized confusion matrix
    plt.figure()
    
    # Compute and plot confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix, without normalization')
    
    plt.show()
    

    