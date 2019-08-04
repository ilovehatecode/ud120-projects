#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.model_selection import train_test_split
import math
### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
features = ["salary", "bonus"]
data_dict.pop("TOTAL",0)
data = featureFormat(data_dict, features)
target, features = targetFeatureSplit( data )
feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.8, random_state=42)

reg = LinearRegression()

### your code below

for point in data:
    salary = point[0]
    bonus = point[1]
 

    matplotlib.pyplot.scatter( salary, bonus )

bandits = list()
for key in data_dict.keys():
    if data_dict[key]['bonus'] != 'NaN' or data_dict[key]['salary'] != 'NaN':
        if data_dict[key]['bonus'] >= 5000000 and data_dict[key]['salary'] >= 1000000:
            bandits.append(key)
        

reg.fit(feature_train, target_train)
pred = reg.predict(features)
print(features[0])
print(pred[0])
errors = list()

for i in range(0, len(pred)):
    error = math.pow(target[i] - pred[i],2)
    errors.append(error)

highestError = 0
highestErrorIndex = 0
for j in range(0, len(errors)):
    if errors[j] > highestError:
        highestError = errors[j]
        highestErrorIndex = j


matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()


