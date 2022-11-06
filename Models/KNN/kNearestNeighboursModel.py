import enum
from functools import reduce
import logging
import math
from statistics import mode
import threading
from typing import List, Tuple
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sortedcontainers import SortedKeyList
import numpy as np
from ..Helpers import StdDistributionType

class DistanceMetricType(enum.Enum):
    Euclidian = 1
    Minkowski = 2
    Manhatten = 3
    Cosine = 4
    Jaccard = 5

class DistanceCalculator():

    @staticmethod
    def calc(x : pd.Series, y : pd.Series, distanceMetric : DistanceMetricType) -> float:

        if distanceMetric == DistanceMetricType.Euclidian:
            return np.linalg.norm(x - y)
        if distanceMetric == DistanceMetricType.Manhatten:
            total = 0
            for index in range(len(x)):
                total += abs(x[index] - y[index])
            return total
        if distanceMetric == DistanceMetricType.Cosine:
            return -abs(float(np.dot(x, y)) / float(np.linalg.norm(x)*np.linalg.norm(y)))

class KNearestNeighboursModel():
    
    encodedData : pd.DataFrame

    columnEncoders : List[LabelEncoder] = []
    columnDistributionTypes = List[DistanceMetricType]

    classEncoder : LabelEncoder
    encodedClasses = List

    # Fits to the data, by converting all labeled data to encoded data
    def fit(self, trainingData : pd.DataFrame, trainingDataColumnDistributionTypes: List[StdDistributionType], trainingClases : List):
        
        logging.info("Creating new KNN model for:")
        logging.info(f"\n{trainingData}")

        logging.info("Encoding labeled data...")
        self.columnDistributionTypes = trainingDataColumnDistributionTypes
        self.encodedData, self.columnEncoders = self.encodeTable(trainingData, trainingDataColumnDistributionTypes)
        self.encodedData = self.encodedData.reset_index(drop=True) # reindex from 0

        logging.info("Encoding classes...")
        self.classEncoder = LabelEncoder()
        self.encodedClasses = self.classEncoder.fit_transform(trainingClases)
        
        logging.info("Successfully fit to data")

    # A function for encoding the input table's columns if they are labeled data.
    def encodeTable(self, inputTable : pd.DataFrame, columnDistributionTypes : List[StdDistributionType]) -> tuple[pd.DataFrame, List]:

        # Go through each column in the tabgle, and if columnDistrubtionTypes specifies that the data is labeled,
        # encode it.
        encodedData = inputTable
        columnEncoders = [None for x in range(len(self.columnDistributionTypes))] # Initalize columnEncoders list with none values.

        if (len(self.columnEncoders) == 0):
            logging.info("Encoding table for the first time, producing column encoders for later use")
        else :
            logging.info("Encoding table using existing column encoders ...")

        for columIndex, distributionType in enumerate(self.columnDistributionTypes):

            if (distributionType == StdDistributionType.Labeled): # Encode the column and store an encoder for use later.
                if (len(self.columnEncoders) == 0):
                    columnEncoder = LabelEncoder()
                    encodedData[encodedData.columns[columIndex]] = columnEncoder.fit_transform(encodedData[encodedData.columns[columIndex]])
                    columnEncoders[columIndex] = columnEncoder
                else:
                    encodedData[encodedData.columns[columIndex]] = self.columnEncoders[columIndex].fit_transform(encodedData[encodedData.columns[columIndex]])                    

        # Scale the data
        logging.info("Scaling data...")
        logging.debug(f"Data before : \n{encodedData}")
        dbColumns = encodedData.columns
        scaler = StandardScaler()
        encodedData = pd.DataFrame(scaler.fit_transform(encodedData), columns=dbColumns)
        logging.debug(f"Data after : \n{encodedData}")

        if (None in columnEncoders):
            return (encodedData, self.columnEncoders)
        else :
            return (encodedData, columnEncoders)

    # Outputs, for each distance metric specified a List of predictions of the class of each row in the testing data.
    def predict(self, testingData : pd.DataFrame, distanceMetrics : List[DistanceMetricType], k : int) -> List[List]:

        # Function: Make a prediction for a given row. We use the distanceMetricIndex and the row index to store the value in the predictions matrix,
        #           and use the row and distanceMetric to calculate the distance from the row to every row in the trainingDataSet.
        def makePrediction(distanceMetricIndex, rowIndex, row, distanceMetic, predictions):
            logging.debug(f"row {rowIndex}/{len(testingData)}")
            predictions[distanceMetricIndex][rowIndex] = calculateShortestDistanceClass(row, distanceMetic)

        # Function : Calculate the shortest distane between a given row and every row in the trainingDataSet.
        def calculateShortestDistanceClass(row, distanceMetric) :
            distanceData = self.encodedData.copy()
            distanceData["distance"] = distanceData.apply(lambda dataPoint: DistanceCalculator.calc(row, dataPoint, distanceMetric), axis=1)
            kSmallest = distanceData.nsmallest(k, ['distance'])
            kSmallestClasses = [self.encodedClasses[index] for index, neighbour in kSmallest.iterrows()]
            return mode(kSmallestClasses) 

        # Main Function:
        logging.info(f"Initalising a 2d array of predictions. ({len(testingData)} predictions by {len(distanceMetrics)} distane metrics)...")
        encodedTestingData, _ = self.encodeTable(testingData, self.columnDistributionTypes)
        encodedTestingData = encodedTestingData.reset_index(drop=True)
        predictions = [[-1 for dataPoint in range(len(encodedTestingData))] for distanceMetic in range(len(distanceMetrics))]
        calculationThreads = []

        logging.info(f"Looping through each datapoint and calculating the k-nearest neighbours for each distance metric using multi-threading")
        for distanceMetricIndex,distanceMetic in enumerate(distanceMetrics):
            logging.info(f"Calulating KNN for distance metric {distanceMetic}")

            for rowIndex in range(len(encodedTestingData)):

                row = encodedTestingData.iloc[rowIndex]

                # Create a thread for each calculation that needs to be performed
                calculationThreads.append(threading.Thread(target=makePrediction, args=(distanceMetricIndex, rowIndex, row, distanceMetic, predictions)))

                if (len(calculationThreads) > 100):
                    
                    for thread in calculationThreads:
                        thread.start()
                    
                    for thread in calculationThreads:
                        thread.join()
                    calculationThreads.clear()

            for thread in calculationThreads:
                thread.start()
        
            for thread in calculationThreads:
                thread.join()
            
            calculationThreads.clear()
        
        for index in range(len(predictions)):
            predictions[index] = self.classEncoder.inverse_transform(predictions[index])

        return predictions

