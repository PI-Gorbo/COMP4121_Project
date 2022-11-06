# This file runs the naviveBaysModel on the prepared datasets.
import logging
from operator import le
from typing import List
import pandas as pd
from sklearn import model_selection as ms
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from .logisticRegressionModel import LogisticRegressionModel, GradientAscentMethod
from ..Helpers import Helpers, Dataset, StdDistributionType
import numpy as np

class Runner:
    
    def run(self, dataset : Dataset):


        logging.basicConfig(level=logging.INFO)

        if (dataset == Dataset.Bank):

            data = pd.read_csv("./PreparedDatasets/BankInfo/bank-additional-full.csv",sep=";")
            inputs = data.iloc[:,:-1].drop(columns=['campaign'])
            outputs = data.iloc[:,-1].to_list()

            # Split the data into testing and training sets.
            trainingData, testingData, trainingOutcomes, testingOutcomes = ms.train_test_split(inputs, outputs, test_size=0.25, random_state=0)
                                                                            # test size of 0.25 = 25% of the data will be used in testing.'
            dataClassifications = [
                StdDistributionType.RealValued, # age
                StdDistributionType.Labeled, # job
                StdDistributionType.Labeled, # marital
                StdDistributionType.Labeled, # ediucation
                StdDistributionType.Labeled, # default
                StdDistributionType.Labeled, # housing
                StdDistributionType.Labeled, # loan
                StdDistributionType.Labeled, # contact
                StdDistributionType.Labeled, # month
                StdDistributionType.Labeled, # day of the week
                StdDistributionType.RealValued, # Duration
                StdDistributionType.RealValued, # pdays -> Will need cleaning for a nice gaussian
                StdDistributionType.RealValued, # previous
                StdDistributionType.Labeled, # poutcome
                StdDistributionType.RealValued, # emp.var.rate
                StdDistributionType.RealValued, # cons.price.idx
                StdDistributionType.RealValued, # cons.conf.idx
                StdDistributionType.RealValued, # euribor3m
                StdDistributionType.RealValued, # nr. employed
            ]
            # Fit the data to the model.A
            model = LogisticRegressionModel()
            model.fit(trainingData, dataClassifications, trainingOutcomes, 100, 0.5, GradientAscentMethod.Batch, 2000)
            predictions = model.predict(testingData)
            Helpers.PrintMetrics(predictions, testingOutcomes, "Logisitic Regression")
        
        if (dataset == Dataset.Cancer):

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
            # Fit the data to the model.
            model = LogisticRegressionModel()
            model.fit(trainingData, dataClassifications, trainingOutcomes, 100, 0.5, GradientAscentMethod.Batch, 2000)
            predictions = model.predict(testingData)
            Helpers.PrintMetrics(predictions, testingOutcomes, "Logistic Regression")
        



