##Test

#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot as plt
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from multi_testing import makeFeatureCombos
from sklearn.model_selection import train_test_split
from tester import test_classifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

    
##Add new features to data_dict

for keys in data_dict:
    if data_dict[keys]['to_messages'] != 'NaN' and data_dict[keys]['from_poi_to_this_person'] != 'NaN' :
        data_dict[keys]['percent_received_from_poi'] = np.float32(float(data_dict[keys]['from_poi_to_this_person'])/float(data_dict[keys]['to_messages']))
    else:
        data_dict[keys]['percent_received_from_poi'] = 'NaN'
    if data_dict[keys]['from_messages'] != 'NaN' and data_dict[keys]['from_this_person_to_poi'] != 'NaN':
        data_dict[keys]['percent_sent_to_poi'] =  np.float32(float(data_dict[keys]['from_this_person_to_poi'])/float(data_dict[keys]['from_messages']))
    else:
        data_dict[keys]['percent_sent_to_poi'] = 'NaN'


##Create combinations
##User Input - Put the features you want in global-features_list ##        
global_features_list = ['from_poi_to_this_person',  'from_this_person_to_poi', 'from_messages', 'to_messages','deferred_income', 'director_fees']
##User Input - change second parameter (number of elements in combos)
combos_features_list = makeFeatureCombos(global_features_list, 2, "poi")
### Store to my_dataset for easy export below.
my_dataset = data_dict


##User Input - Create different classifers here
clf1 = DecisionTreeClassifier(max_depth=4)

##User Input - Add classifiers to the list to be tested
list_classifiers = [clf1]

### Extract features and labels from dataset for local testing
for clf in list_classifiers:
    for features_list in combos_features_list:
        print features_list
##        data = featureFormat(my_dataset, features_list, sort_keys = True)
##        labels, features = targetFeatureSplit(data)
##
##        import random
##        for i in range(0, len(scaled_features)):
##     ##       plt.scatter(scaled_features[i][0], scaled_features[i][1], color=colors[int(labels[random.randint(1, len(colors))])])
##    ##    ##plt.show()

        print(test_classifier(clf, my_dataset, features_list))
        print("\n")


