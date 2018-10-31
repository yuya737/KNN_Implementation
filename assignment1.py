import numpy as np
import pandas as pd
import operator
import math


def calculateDistance(point1, point2, dimension):
    """Short summary.
        Calculate Distance between point1 and point2

    Parameters
    ----------
    point1 :
        first point to be considered
    point2 :
        second point to be considered
    dimension :
        number of predictors/dimension of the points

    Returns
    -------
    int
        the euclidean distance between the points

    """
    distance=0
    for x in range(dimension):
        distance += pow((point1[x] - point2[x]), 2)
    return math.sqrt(distance)

def getNeighbors(trainingSet, testInstance, distances):
    """Short summary.
        get a list of tuples, containing the data point and the distance from the testInstance
    Parameters
    ----------
    trainingSet : type
        training set of the implementation
    testInstance : type
        data point from which the distances will be calcualted
    distances : type
        list to which the tuples will be appeded

    Returns
    -------
    list
        list of tuples

    """
    # Empty list to store distances of between testInstance and each trainSet item
    # Number of dimensions to check
    length=len(testInstance) - 1
    # Iterate through all items in trainingSet and compute the distance, then append to the distances list
    for x in range(len(trainingSet)):
        dist=calculateDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    return distances



def getResponse(neighbors):
    """Short summary.
        return the majority vote given the neighbors
    Parameters
    ----------
    neighbors : list
        list of neighbors to be considered

    Returns
    -------
    bool
        the majority, either true or false

    """
    # Empty dictionary to hold each output and the counts
    count={True: 0, False: 0}
    # For each neighbors, retreive the class to which it belongs, and add to count
    for x in range(len(neighbors)):
        response=neighbors[x][-1]
        if response:
            count[True] += 1
        else:
            count[False] += 1
    print count
    if count[True] > count[False]:
        return True
    else:
        return False



def crossValidation(training):
    """Short summary.
        Run 5-fold cross validation
    Parameters
    ----------
    training : type
        Description of parameter `training`.

    Returns
    -------
    type
        Description of returned object.

    """
    predictions = []
    for index in range(1, 6):
        print index
        temp = list(range(1, 6))
        temp.remove(index)
        k = 3
        for x in range(len(training.get_group(index))):
            target = training.get_group(index).values.tolist()[x][-1]
            print x
            neighbors = []
            distances = []
            accuracy = []
            for validationSet in temp:
                getNeighbors(training.get_group(validationSet).values.tolist(), training.get_group(index).values.tolist()[x], distances)
            # Sort the distances list by the distance
            distances.sort(key = lambda item: item[1])
            # print distances
            # Select first k closest elements to return as the neighbors
            for x in range(k):
                neighbors.append(distances[x][0])

            result=getResponse(neighbors)

            print distances
            print neighbors
            print result
            predictions.append(result)
            # print 'result: ' + str(result)
            # print 'target: ' + str(target)
            # print 'result == target: ' + str(result == target)
            if result == target:
                accuracy.append(True)
            else:
                accuracy.append(False)

def main():
    trainingOriginal = pd.read_csv(
        filepath_or_buffer="~/Desktop/KNN Implementation/adult.train.5fold.csv")
    testOriginal = pd.read_csv(filepath_or_buffer="~/Desktop/KNN Implementation/adult.test.csv")

    # Select only the numeric data
    training = pd.DataFrame(trainingOriginal.select_dtypes(['number']))
    training = pd.concat([training.reset_index(drop=True),
                         trainingOriginal['earns'].reset_index(drop=True)], axis=1)
    # Select only the numeric data
    test = pd.DataFrame(testOriginal.select_dtypes(['number']))
    test = pd.concat([test.reset_index(drop=True),
                     testOriginal['earns'].reset_index(drop=True)], axis=1)

    training['earns'] = training['earns'] == '>50K'
    test['earns'] = test['earns'] == ' >50K'

    training = training.groupby('fold')

    crossValidation(training)

main()
