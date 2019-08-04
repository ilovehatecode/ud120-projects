#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot as plt
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from multi_testing import makeFeatureCombos
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.tree import DecisionTreeClassifier
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','from_poi_to_this_person',  'from_this_person_to_poi', 'from_messages', 'to_messages','deferred_income', 'director_fees'] # You will need to use more features

##Create combinations
global_features_list = ['from_poi_to_this_person',  'from_this_person_to_poi', 'from_messages', 'to_messages','deferred_income', 'director_fees']


combos_features_list = makeFeatureCombos(global_features_list, 1, "poi")



### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)



### Task 2: Remove outliers

    
### Task 3: Create new feature(s)
##Add percentage messages to/from POI

for keys in data_dict:
    if data_dict[keys]['to_messages'] != 'NaN' and data_dict[keys]['from_poi_to_this_person'] != 'NaN' :
        data_dict[keys]['percent_received_from_poi'] = np.float32(float(data_dict[keys]['from_poi_to_this_person'])/float(data_dict[keys]['to_messages']))
    else:
        data_dict[keys]['percent_received_from_poi'] = 'NaN'
    if data_dict[keys]['from_messages'] != 'NaN' and data_dict[keys]['from_this_person_to_poi'] != 'NaN':
        data_dict[keys]['percent_sent_to_poi'] =  np.float32(float(data_dict[keys]['from_this_person_to_poi'])/float(data_dict[keys]['from_messages']))
    else:
        data_dict[keys]['percent_sent_to_poi'] = 'NaN'



### Store to my_dataset for easy export below.
my_dataset = data_dict




### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
features_list = ['poi','deferred_income', 'director_fees', 'percent_received_from_poi', 'percent_sent_to_poi'] 
clf = DecisionTreeClassifier(criterion='gini', max_depth=4, class_weight='balanced', presort=True)
##clf = GaussianNB()


from tester import test_classifier
print(test_classifier(clf, my_dataset, features_list))
print("\n")
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

