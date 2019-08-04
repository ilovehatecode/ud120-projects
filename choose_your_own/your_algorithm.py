#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
##plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary

##K-Neighbors
import time
wght = 'distance'
num = 2
print("KNNeighbors w/ " + str(num) + " neighbors, weights = " + wght)
from sklearn.neighbors import KNeighborsClassifier
t0= time.time()
clf = KNeighborsClassifier(n_neighbors=num, weights=wght)
clf.fit(features_train, labels_train)
t1= time.time()
print("Training time: " + str(t1 - t0))
pred = clf.predict(features_test)
acc = clf.score(features_test, labels_test)
print(acc)


##NK Neighbors neighbors =3 | accuracy = .936
##KNNeighbors neighbors=5 | accuracy = .92
##KNNeighbors neighbors=2 | accuracy = .928
##KNNeighbors neighbors=7 | accuracy = .936
##KNNeighbors neighbors=10 | accuracy = .932
##KNNeighbors neighbors=100 | accuracy = .928






try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
