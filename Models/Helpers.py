import enum
import logging
import math
from typing import List, Tuple
import pandas as pd
from sklearn import model_selection as ms
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt

class Dataset(enum.Enum):
    Bank = 0
    Cancer = 1

class StdDistributionType(enum.Enum):
    Labeled = 1
    RealValued = 2

class Helpers:
    
    @staticmethod
    def PrintMetrics(predictions : List, actual : List, title : str) -> float:
        
        if (len(predictions) != len(actual)):
            raise ValueError("Predictions array must be the same length as the actual array")
        
        logging.info("_____________________________________________________________________")
        logging.info(f"Results for {title}")
        logging.info(f"\n{classification_report(actual, predictions)}")

        confustionMatrix = confusion_matrix(actual, predictions)
        sn.heatmap(confustionMatrix, annot=True)
        plt.savefig(f'{title}_confusion.png')
        plt.clf()


    @staticmethod
    # A function for encoding the input table's columns if they are labeled data.
    def EncodeTable(table : pd.DataFrame, distributionTypes : List[StdDistributionType], pregeneratedEncoders : List[LabelEncoder] = None) -> Tuple[pd.DataFrame, List[LabelEncoder]]:

        # Select pregenerated enoders if given
        logging.info("Encoding the table with Standard distribution types")
        columnEncoders : List[LabelEncoder]
        if (pregeneratedEncoders == None):
            columnEncoders = [None for x in range(len(distributionTypes))]
            logging.debug("Generating a new encoders list")
        else:
            logging.debug("Using a pregenerated encoders list")
            columnEncoders = pregeneratedEncoders

        # Go through each column and encode it if the distributionType specifies it should be encoded
        encodedData = table
        for columIndex, distributionType in enumerate(distributionTypes):

            if (distributionType == StdDistributionType.Labeled): # Encode the column and store an encoder for use later.
                
                columnName = encodedData.columns[columIndex]
                
                # If there is no a pregiven encoder, then create a new encoder and fit it.
                if (columnEncoders[columIndex] == None):
                    logging.debug(f"Creating a new label encoder for column {columnName}")
                    columnEncoder = LabelEncoder()
                    encodedData[columnName] = columnEncoder.fit_transform(encodedData[columnName])
                    columnEncoders[columIndex] = columnEncoder
                else:
                    logging.debug(f"Using a pregenerated column encoder for column {columnName}")
                    columnEncoder = columnEncoders[columIndex]
                    encodedData[columnName] = columnEncoder.transform(encodedData[columnName])

        return (encodedData, columnEncoders)

    @staticmethod
    def EncodeOutcomes(outcomes : List, pregeneratedEncoder : LabelEncoder = None) -> Tuple[List, LabelEncoder]:
        logging.debug("Encoding outcomes...")
        if pregeneratedEncoder == None:
            logging.debug("No label encoder given, creating and encoding...")
            encoder = LabelEncoder()
            encodedOutcomes = encoder.fit_transform(outcomes)
            logging.debug(f"Encoded outcomes : {encodedOutcomes}")
            return (encodedOutcomes, encoder)
        else:
            logging.debug("Using given label encoder")
            return (pregeneratedEncoder.transform(outcomes), pregeneratedEncoder)

class PreparedDatasets:

    @staticmethod
    def cancerData() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, List[StdDistributionType]] :
        """
            7. Attribute Information: (class attribute has been moved to last column)

            #  Attribute                     Domain
            -- -----------------------------------------
            1. Sample code number            id number
            2. Clump Thickness               1 - 10
            3. Uniformity of Cell Size       1 - 10
            4. Uniformity of Cell Shape      1 - 10
            5. Marginal Adhesion             1 - 10
            6. Single Epithelial Cell Size   1 - 10
            7. Bare Nuclei                   1 - 10
            8. Bland Chromatin               1 - 10
            9. Normal Nucleoli               1 - 10
            10. Mitoses                       1 - 10
            11. Class:                        (2 for benign, 4 for malignant)
        """
        
        data = pd.read_csv("./PreparedDatasets/BreastCancer/BreastCancer_Wisconsin/breast-cancer-wisconsin.data.txt", sep=",")
        data = data.replace("?", np.nan)
        indexesWithNan = data.index[data.isnull().any(axis=1)]
        data = data.drop(indexesWithNan, 0)
        inputs = data[data.columns[1:10]]
        predictions = data[data.columns[-1]]
        print(inputs)
        print(predictions)
        trainingData, testingData, trainingOutcomes, testingOutcomes = ms.train_test_split(inputs, predictions, test_size=0.25, random_state=0)

        dataClassifications = [
            StdDistributionType.Labeled, # Clump Thickness
            StdDistributionType.Labeled, # Uniformity of Cell Size 
            StdDistributionType.Labeled, # Uniformity of Cell Shape
            StdDistributionType.Labeled, # Marginal Adhesion
            StdDistributionType.Labeled, # Single Epithelial Cell Size
            StdDistributionType.Labeled, # Bare Nuclei
            StdDistributionType.Labeled, # Bland Chromatin
            StdDistributionType.Labeled, # Normal Nucleoli
            StdDistributionType.Labeled, # Mitoses
        ]
        return (trainingData, testingData, trainingOutcomes, testingOutcomes, dataClassifications)

    @staticmethod
    def get(dataset : Dataset) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, List[StdDistributionType]]:

        if (dataset == Dataset.Cancer):
            return PreparedDatasets.cancerData()