from itertools import combinations
import sys
sys.path.append("../tools/")



def makeFeatureCombos(features_list, num_of_features, label):
    combos_of_features = combinations(features_list, num_of_features)
    combos_of_features = list(combos_of_features)
    for i in range(0, len(combos_of_features)):
        new_list = list()
        new_list.append(label)
        for j in range(0, len(combos_of_features[i])):
            new_list.append(combos_of_features[i][j])
        combos_of_features[i] = new_list
    
    return combos_of_features


##def multitester(func, combos_of_features):
