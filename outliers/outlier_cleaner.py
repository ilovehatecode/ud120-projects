#!/usr/bin/python

import math
import numpy as np
def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    num_to_remove = int(round(len(predictions)*0.10))
    error = []
    for i in range(0, len(predictions)):
        error.append(math.pow(net_worths[i] - predictions[i],2))
    ##Remove data from all lists
    for j in range(0, num_to_remove):
        index = getIndexHighestValue(error)
        ages = np.delete(ages, index, 0) 
        net_worths = np.delete( net_worths, index, 0)
        del error[index]

        
    cleaned_data = []
    ### your code goes here
    for k in range(0, len(error)):
        cleaned_data.append([ages[k], net_worths[k], error[k]])
    return cleaned_data



def getIndexHighestValue(List):
    highestValue = List[0]
    index = 0
    for i in range(1, len(List)):
        if List[i] > highestValue:
            index = i
            highestValue = List[i]
    return index



    
        
        
