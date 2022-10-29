# This file runs the naviveBaysModel on the prepared datasets.
import logging
from operator import le
import pandas as pd
from sklearn import model_selection as ms
from sklearn import preprocessing
from naiveBayesModel import NaiveBayesModel, DistributionType

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    data = pd.read_csv("./../PreparedDatasets/BankInfo/bank-additional-full.csv",sep=";")
    inputs = data.iloc[:,:-1]
    outputs = data.iloc[:,-1].to_list()

    # Split the data into testing and training sets.
    trainingData, testingData, trainingOutcomes, testingOutcomes = ms.train_test_split(inputs, outputs, test_size=0.25, random_state=0)
                                                                    # test size of 0.25 = 25% of the data will be used in testing.'
    dataClassifications = [
        DistributionType.GaussianDST, # age
        DistributionType.MultiModalDST, # job
        DistributionType.MultiModalDST, # marital
        DistributionType.MultiModalDST, # ediucation
        DistributionType.MultiModalDST, # default
        DistributionType.MultiModalDST, # housing
        DistributionType.MultiModalDST, # loan
        DistributionType.BinaryDST, # contact
        DistributionType.MultiModalDST, # month
        DistributionType.MultiModalDST, # day of the week
        DistributionType.GaussianDST, # Duration
        DistributionType.MultiModalDST, # Campaign
        DistributionType.GaussianDST, # pdays -> Will need cleaning for a nice gaussian
        DistributionType.GaussianDST, # previous
        DistributionType.MultiModalDST, # poutcome
        DistributionType.GaussianDST, # emp.var.rate
        DistributionType.GaussianDST, # cons.price.idx
        DistributionType.GaussianDST, # cons.conf.idx
        DistributionType.GaussianDST, # euribor3m
        DistributionType.GaussianDST, # nr. employed
    ]

    # Fit the data to the model.
    model = NaiveBayesModel()
    model.fit(trainingData, dataClassifications, trainingOutcomes)
