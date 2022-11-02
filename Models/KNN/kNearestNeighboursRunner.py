# This file runs KNN on a dataset.
import logging
import pandas as pd
from sklearn import model_selection as ms
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from kNearestNeighboursModel import *

def CalculateAccuracy(predictions : List, actual : List) -> float:
    
    if (len(predictions) != len(actual)):
        raise ValueError("Predictions array must be the same length as the actual array")

    # calculate total predictions and true positive predictions.
    totalPredictions = len(predictions)
    totalTruePositives = len([1 for index in range(len(predictions)) if predictions[index] == actual[index]])
    return float(totalTruePositives) / float(totalPredictions)


def printOutcomes(predictions : List, actual : List, DistanceType : str):
    print(predictions)
    print(actual)
    # Calculate how accurate the list of predictions were compared to the testing outcomes
    print(f"Model accuracy for {DistanceType} distance: {CalculateAccuracy(predictions, actual)}")
    # Generate a confustion matrix
    trueNegative, falsePositive, falseNegative, truePositive = confusion_matrix(testingOutcomes, predictions).ravel()
    print(f"Confustion Matrix:")
    print(f"\t\tPositive\tNegative")
    print(f"True {truePositive} \t {trueNegative}")
    print(f"False {falsePositive} \t {falseNegative}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    numberOfDataPoints = 3000
    data = pd.read_csv("./../../PreparedDatasets/BankInfo/bank-additional-full.csv",sep=";")[:numberOfDataPoints]

    inputs = data.iloc[:,:-1].drop(columns=['campaign'])
    outputs = data.iloc[:,-1].to_list()

    # Split the data into testing and training sets.
    trainingData, testingData, trainingOutcomes, testingOutcomes = ms.train_test_split(inputs, outputs, test_size=0.25, random_state=0)
                                                                    # test size of 0.25 = 25% of the data will be used in testing.'
    dataClassifications = [
        DistributionType.RealValued, # age
        DistributionType.Labeled, # job
        DistributionType.Labeled, # marital
        DistributionType.Labeled, # ediucation
        DistributionType.Labeled, # default
        DistributionType.Labeled, # housing
        DistributionType.Labeled, # loan
        DistributionType.Labeled, # contact
        DistributionType.Labeled, # month
        DistributionType.Labeled, # day of the week
        DistributionType.RealValued, # Duration
        DistributionType.RealValued, # pdays -> Will need cleaning for a nice gaussian
        DistributionType.RealValued, # previous
        DistributionType.Labeled, # poutcome
        DistributionType.RealValued, # emp.var.rate
        DistributionType.RealValued, # cons.price.idx
        DistributionType.RealValued, # cons.conf.idx
        DistributionType.RealValued, # euribor3m
        DistributionType.RealValued, # nr. employed
    ]
    # Fit the data to the model.
    model = KNearestNeighboursModel()
    model.fit(trainingData, dataClassifications, trainingOutcomes)
    predictions = model.predict(testingData, [DistanceMetricType.Euclidian, DistanceMetricType.Manhatten], 5)

    printOutcomes(predictions[0], testingOutcomes, "Euclidian")
    printOutcomes(predictions[1], testingOutcomes, "Manhatten")



