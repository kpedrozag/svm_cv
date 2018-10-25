#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 09:55:19 2018

@author: ceisutb17
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import sklearn.metrics as mtr
import numpy as np

X, y = load_breast_cancer(return_X_y=True)

scaler = StandardScaler()
X= scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,
                                                    random_state=0, 
                                                    stratify=y)

c = [10**i for i in range(-10,11)]


svm_list = [SVC(C=i, kernel='linear', 
                random_state=0).fit(X_train, y_train) for i in c]

score = [cross_val_score(st, X_train, y_train, cv=30).mean() for st in svm_list]

pos_max_score = np.argmax(score)

svm_valid = SVC(C=c[pos_max_score], kernel='linear',
                random_state=0).fit(X_train, y_train)

model = svm_valid.predict(X_test)

f1 = mtr.f1_score(y_test,model)
recall = mtr.recall_score(y_test,model)
accuracy = mtr.accuracy_score(y_test,model)
precision = mtr.precision_score(y_test,model)
tn, fp, fn, tp = mtr.confusion_matrix(y_test, model).ravel()
specificity = tn/float(tn+fp)

print("\n\t   CONFUSION MATRIX")
print("         Negative     Positive")
print("Negative   {0}           {1}".format(tn,fp))
print("Positive   {0}            {1}".format(fn,tp))
print("\nF1-score: {0}\nRecall: {1}\nAccuracy: {2}\nPrecision: {3}\nSpecificity: {4}".format(f1,recall,accuracy,precision,specificity))