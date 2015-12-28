# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 19:52:36 2015

@author: lijian
"""
import numpy as np
from numpy import *
import pandas as pd
from pandas import DataFrame

df = pd.read_csv('train_x.csv')
trainingSet = np.array(df)

labels = pd.read_csv('train_y.csv')
labelsArr = np.array(labels)

trainMat = [];trainLabels = []
testMat = [];testLabels = []

for i in range(0,14999,1):
    
    if i%10 == 1:
        testMat.append(trainingSet[i,:])
        testLabels.append(labelsArr[i,:])
    else:
        trainMat.append(trainingSet[i,:])
        trainLabels.append(labelsArr[i,:])
	
 #验证
shape(testMat)   #(1500L,1139L)
shape(trainMat)  #(13499L,1139L)
	

	
	
	
	
	
	
	
