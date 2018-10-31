from __future__ import division
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
    # print 'p1: ' + str(point1) + 'p2: ' + str(point2) + str(dimension)
    for x in range(dimension - 1):
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
    # print count
    if count[True] > count[False]:
        return True
    else:
        return False



def crossValidation(training, k, performance):
    """Short summary.
        Perform 5-fold cross validation given the training set
    Parameters
    ----------
    training : type
        the training set to be considered
    k : type
        hyperparameter 'k' to be tuned
    performance : type
        dictionary to store the k-value and the respective accuracies

    Returns
    -------
    dictionary to store the k-value and the respective accuracies


    """

    predictions = []
    accuracy = []

    for index in range(1, 6):
        # print index
        temp = list(range(1, 6))
        temp.remove(index)
        print 'index: ' + str(index) + ', temp: ' + str(temp)

        for x in range(len(training.get_group(index))):
            if x % 50 != 0:
                continue
            target = training.get_group(index).values.tolist()[x][-1]
            # if x % 500 == 0:
            #     print 'index: ' + str(index) + ', x: ' + str(x)
            neighbors = []
            distances = []
            for validationSet in temp:
                getNeighbors(training.get_group(validationSet).values.tolist(), training.get_group(index).values.tolist()[x], distances)
            # Sort the distances list by the distance
            distances.sort(key = lambda item: item[1])
            # print distances
            # Select first k closest elements to return as the neighbors
            for x in range(k):
                neighbors.append(distances[x][0])

            result=getResponse(neighbors)

            # print distances
            # print neighbors
            # print result
            predictions.append(result)
            # print 'result: ' + str(result)
            # print 'target: ' + str(target)
            # print 'result == target: ' + str(result == target)
            if result == target:
                accuracy.append(True)
            else:
                accuracy.append(False)

        print 'number of instances: ' + str(len(accuracy)) + ' number correct: ' + str(sum(accuracy))

    # Add the current k-value and its accuracy for this run to dictionary
    performance[k] = sum(accuracy) / len(accuracy)

    print performance
    return performance

def main():
    """Short summary.
        main driver for this file
    """
    # Read in trainingSet and testSet as a DataFrame
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

    # Normalize the columns for training and test
    # print training['age'].min()
    # print training['age'].max()
    # print training.head()

    # Run max-min normalization on numerical columns for testing and training data
    for i in range(6):
        training.iloc[:, i] =  (training.iloc[:, i]- training.iloc[:, i].min())/(training.iloc[:, i].max() - training.iloc[:, i].min())
        test.iloc[:, i] =  (test.iloc[:, i]- test.iloc[:, i].min())/(test.iloc[:, i].max() - test.iloc[:, i].min())

    # Convert the 'earns' column to boolean as follows
    training['earns'] = training['earns'] == '>50K'
    test['earns'] = test['earns'] == ' >50K'

    # Group the training set by the fold attribute as given by the dataset
    training = training.groupby('fold')

    # Since we want to consider odd k-values from 1 to 39, construct a list with these values
    kList = []
    for i in range(40):
        if i % 2 == 1:
            kList.append(i)
    print kList

    # Empty dictionary to hold performance of each k-values and its accuracy
    performance = {}
    print performance

    # Compute the performance for each k-value
    for k in kList:
        performance = crossValidation(training, k, performance)

    # Sort the performance dictionary by its accuracy (value)
    performance = sorted(performance.items(), key=operator.itemgetter(1), reverse=True)

    # Open file to write results
    file = open('grid.results.txt', 'w')
    # Write the results to file
    file.write("K   |  Accuracy\n")
    for item in performance:
        if item[0] < 10:
            file.write(str(item[0]) + '   |  ' + str(item[1]) + '\n')
        else:
            file.write(str(item[0]) + '  |  ' + str(item[1]) + '\n')
    # Close file
    file.close()
    print performance

# Run the program
main()
