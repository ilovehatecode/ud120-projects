#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


from sklearn import svm
##features_train = features_train[:len(features_train)/100] 
##labels_train = labels_train[:len(labels_train)/100] 
#########################################################
### your code goes here ###
##Create classifier and fit training set
clf = svm.SVC(C=10000.0,kernel="rbf")
clf.fit(features_train, labels_train)
##Predict and compare to test set
pred = clf.predict(features_test)
accuracy = clf.score(features_test, labels_test)
print(pred[10], pred[26], pred[50])
##Number of Emails labeled Chris (1)
count = 0
for p in pred:
    if(p ==1):
        count+=1
print(count)
print(accuracy)
#########################################################


