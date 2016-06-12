# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 13:08:16 2016

@author: Luyixiao
"""
import csv
from sklearn import tree 
import numpy as np
def readData(path,ratio):
    reader = csv.reader(file(path, 'rb'))
    data = []
    flag = []
    dataTrain = []
    flagTrain = []   
    dataTest = []
    flagTest = [] 
    for line in reader:
        data.append([float(line[0]),float(line[1]),float(line[2]),float(line[3]),float(line[4]),float(line[5])])
        flag.append(float(line[6]))
    for i in range(0,int(len(data)*ratio)):
        dataTrain.append(data[i])
        flagTrain.append(flag[i])
    for i in range(int(len(data)*ratio),len(data)):
        dataTest.append(data[i])
        flagTest.append(flag[i])
    return dataTrain,flagTrain,dataTest,flagTest
path = 'I:/MLtrain/hData.csv'
X_train,Y_train,X_test,Y_test = readData(path,0.8)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, Y_train)
pre=clf.predict(X_test)
print np.sum(np.abs(pre - np.array(Y_test)))/len(pre)