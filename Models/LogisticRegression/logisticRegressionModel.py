import enum
import logging
import math
import random
from typing import List
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np

class DistributionType(enum.Enum):
    Labeled = 1
    RealValued = 2

class GradientAscentMethod(enum.Enum):
    Batch = 1
    Stochastic = 2
    MiniBatch = 3

class LogisticRegressionModel():
    
    # Properties
    columnEncoders : List[LabelEncoder] = []
    columnDistributionTypes = []

    classEncoder : LabelEncoder = None
    params = None

    def encodeTable(self, inputTable : pd.DataFrame, columnDistributionTypes : List[DistributionType]) -> tuple[pd.DataFrame, List]:
        # Go through each column in the tabgle, and if columnDistrubtionTypes specifies that the data is labeled,
        # encode it.
        encodedData = inputTable
        columnEncoders = [None for x in range(len(columnDistributionTypes))] # Initalize columnEncoders list with none values.
        self.columnDistributionTypes = columnDistributionTypes

        if (len(self.columnEncoders) == 0):
            logging.info("Encoding table for the first time, producing column encoders for later use")
        else :
            logging.info("Encoding table using existing column encoders ...")

        for columIndex, distributionType in enumerate(columnDistributionTypes):

            if (distributionType == DistributionType.Labeled): # Encode the column and store an encoder for use later.

                if (len(self.columnEncoders) == 0):
                    columnEncoder = LabelEncoder()
                    encodedData[encodedData.columns[columIndex]] = columnEncoder.fit_transform(encodedData[encodedData.columns[columIndex]])
                    columnEncoders[columIndex] = columnEncoder
                else:
                    encodedData[encodedData.columns[columIndex]] = self.columnEncoders[columIndex].fit_transform(encodedData[encodedData.columns[columIndex]]) 

        # Scale the output table.
        logging.info("Scaling data...")
        logging.debug(f"Data before : \n{encodedData}")
        dbColumns = encodedData.columns
        scaler = StandardScaler()
        encodedData = pd.DataFrame(scaler.fit_transform(encodedData), columns=dbColumns)
        logging.debug(f"Data after : \n{encodedData}")
        
        # Reset the indexing to match the indexes of the outcomes.
        encodedData = encodedData.reset_index(drop=True)

        if (len(self.columnEncoders) == 0):
            return (encodedData, self.columnEncoders)
        else :
            return (encodedData, columnEncoders)                   

    def sigmoid(self, z) -> float: # z is a vector
        return 1 / (1 + np.exp(-z))

    def fit(self, trainingData : pd.DataFrame, 
            trainingDataClassifications : List[DistributionType], 
            trainingOutcomes : List, 
            MaxIterations : int,
            LearningRate : float,
            GradientAscentMethod : GradientAscentMethod,
            MiniBatchSize : int = -1):

        logging.info("Creating a new logistic regression model for:")
        logging.info(f"{trainingData}")
            
        # Transform and scale the training data:
        encodedTrainingData, encoders = self.encodeTable(trainingData, trainingDataClassifications)
        self.columnEncoders = encoders
        self.classEncoder = LabelEncoder()
        encodedTrainingOutcomes = self.classEncoder.fit_transform(trainingOutcomes)

        # Now, aim to find a list of parameters, Params such that the total likelihood is maximised.
        logging.info("Inializing all regression parameters to be 0...")
        params = [random.randint(-100,100) for x in range(len(encodedTrainingData.columns)+1)] # Initalize all parameters to be zero and add a constant term
        logging.info(f"Looping through the {MaxIterations} iterations, optimising the parameters using gradient ascent.")

        encodedTrainingData["biasTerm"] = np.ones(len(encodedTrainingData))
        for iteration in range(MaxIterations):
            logging.info(f"Iteration {iteration}/{MaxIterations}")

            if (GradientAscentMethod == GradientAscentMethod.Batch):

                # Iterate through the gradient ascent process 'MaxIterations' time(s).
                for index, row in encodedTrainingData.iterrows():
                    # logging.debug(f"Evaluating row {index}/{len(encodedTrainingData)}")

                    # We find that the partial derrivative for the Likelihood function with respect to the j'th parameter is
                    # d LLh(parameters) / d paramters[index] = (sigmoidCalculation(z[index]) - expectedOutcome) * row[index] 
                    # where z[index] is the dot product of the parameters and the values of the row.
                    inputs = row.to_numpy()
                    expectedOutcome = encodedTrainingOutcomes[index]
                    sigmoidCalculation = self.sigmoid(np.dot(params, inputs))
                    gradients = (sigmoidCalculation - expectedOutcome) * inputs # Gradients is a vector of partial derivatives.

                    # Then, modify the parameters by the gradients calculated.
                    params = params + (LearningRate * gradients)
                    
            
            if (GradientAscentMethod == GradientAscentMethod.Stochastic):

                # Choose a random row.
                rowIndex = random.randint(0, len(encodedTrainingData)- 1)
                inputs = encodedTrainingData.iloc[rowIndex]
                expectedOutcome = encodedTrainingOutcomes[rowIndex]
                sigmoidCalculation = self.sigmoid(np.dot(params, inputs))
                gradients = (sigmoidCalculation - expectedOutcome) * inputs
                params = params + (LearningRate * gradients)

            if (GradientAscentMethod == GradientAscentMethod.MiniBatch):

                # Choose a random subset of rows
                rowsChosen = encodedTrainingData.sample(n=int(len(encodedTrainingData)/MiniBatchSize))
                for index, row in encodedTrainingData.iterrows():
                    logging.debug(f"Evaluating row {index}/{len(encodedTrainingData)}")

                    # We find that the partial derrivative for the Likelihood function with respect to the j'th parameter is
                    # d LLh(parameters) / d paramters[index] = (sigmoidCalculation(z[index]) - expectedOutcome) * row[index] 
                    # where z[index] is the dot product of the parameters and the values of the row.
                    inputs = row.to_numpy()
                    expectedOutcome = encodedTrainingOutcomes[index]
                    sigmoidCalculation = self.sigmoid(np.dot(params, inputs))
                    gradients = (sigmoidCalculation - expectedOutcome) * inputs # Gradients is a vector of partial derivatives.

                    # Then, modify the parameters by the gradients calculated.
                    params = params + (LearningRate * gradients)
            
        logging.info("Finsihed optimising parameters...")
        logging.info(f"{params}")
        self.params = params

    def predict(self, testingData, precaclulatedParams : List = None):
        
        output = [None for x in range(len(testingData))]

        # Determine which parameters to use, from either given parameters or pre-calculated ones.
        params = self.params
        if (precaclulatedParams != None):
            params = precaclulatedParams
        
        # Encode the testing data.
        encodedTestingData, _ = self.encodeTable(testingData, self.columnDistributionTypes)

        # For each row in the encodedTestData, make a prediction using the sigmoid function and the parameters
        for index, row in encodedTestingData.iterrows():
            
            probability = self.sigmoid(np.dot(params, row.to_numpy()))
            if (probability > 0.5):
                output[index] = int(0)
            else:
                output[index] = int(1)
        print(output)
        print("")
        print(self.classEncoder.classes_)
        print("")
        return self.classEncoder.inverse_transform(output)
