from abc import abstractmethod
import enum
from functools import reduce
import math
from operator import index
from typing import Any, List
from sklearn import preprocessing
import pandas as pd
import logging
import numpy

class DistributionType(enum.Enum):
  BinaryDST = 1
  MultiModalDST = 2
  GaussianDST = 3

"""
# An abstraction for a distribution model applied to a specific feature.
"""
class LikelihoodModel:

  numberOfOutcomes = 0
  dataOutcomesTbl = None

  # Takes in a binary list as data, and a list of encoded outcomes for each datapoint.
  # Encodeded outcomes is a list of the classes which has a minimum value of [0,1]
  def __init__(self, inputData : List, EncodedOutcomes: List, numberOfOutcomes : int):

      if numberOfOutcomes < 2:
          raise ValueError(f"Encoded outcomes needs to be at least a list of length 2. Eg: [0,1]. What as given : {EncodedOutcomes}")

      for num in range(0,numberOfOutcomes):
          if num not in EncodedOutcomes: 
              raise ValueError("Encoded outcomes needs to be a list of integers between 0 and max, where max is an integer. Every integer between 0 and max must appear at least once in the array.")
              
      # Normalize the inputted data into a binary list.
      self.data = inputData
      self.outcomes = EncodedOutcomes
      self.numberOfOutcomes = numberOfOutcomes
      self.dataOutcomesTbl = pd.DataFrame(list(zip(self.data, self.outcomes)), columns=["data", "outcomes"])
      self.fitToData()

  @abstractmethod
  def fitToData(self):
      pass

  @abstractmethod
  def predict(self, x, cls):
      pass

class BinaryDistrubtion(LikelihoodModel):
  
  dataEncoder = None
  successProbabilites = [] 

  def __init__(self, inputData: List, EncodedOutcomes: List, numberOfOutcomes: int):

      # Convert the labeled input data into 0s and 1s.
      self.dataEncoder = preprocessing.LabelEncoder()
      self.dataEncoder.fit(inputData)
      if (len(self.dataEncoder.classes_) != 2):
          raise ValueError(f"Binary Distributions requires a binary data set. From label encoding, there are {len(self.dataEncoder.classes_)} values for this feature.")

      super().__init__(inputData, EncodedOutcomes, numberOfOutcomes)

  def fitToData(self):

      logging.info("\tBinaryDistribution: Converting data from labeled to encoded data.")
      # Encode the data into a binary 0, 1
      self.dataOutcomesTbl['data'] = pd.Series(self.dataEncoder.transform(self.dataOutcomesTbl['data']))

      # For each class, y(i), we need to find the Probability( X = 1 | y(i) )
      # We store this so we can calculate, for any input, z, P( z | y(i) ) = P( z = 1 | y(i) ) * z + (1 - P( z = 1 | y(i) ))(1 - z)
      #   Note that z is a binary int of 0 or 1.
      self.successProbabilites = [ 0 for x in range(self.numberOfOutcomes)] # Initialize a zero array of success probabilities.
                                                                          # That is the same size as the number of different outcomes.
      # Get the number of times X == 1 when y(i) is true.
      truePositiveCount = self.dataOutcomesTbl.groupby("outcomes")[['data']].agg(lambda values : len([x for x in values if x == 1]))
      truePositiveCount.columns = ["TruePositive_Count"]
      totalPositiveCount = self.dataOutcomesTbl.groupby("outcomes")[['data']].agg(lambda values : len(values))
      totalPositiveCount.columns = ["TotalPositive_Count"]
      logging.info("\tBinaryDistribution: Aggregated totalPositive and totalCounts for each class.")

      for y_i in range(len(self.successProbabilites)): # y_i is class y_i.
          # Find the probability of success for each class y_i, by dividing the total number of times (X == 1 | y_i ) / the total number of times the class y_i appears
          self.successProbabilites[y_i] = truePositiveCount.iloc[y_i][0] / totalPositiveCount.iloc[y_i][0]
      # Now, each class, i, in self.successProbabilities represents the probability that X == 1 when i was the observered outcome.
      logging.info("\tBinaryDistribution: Calculated success probabilies for each class:")
      logging.debug(f"\tBinaryDistribution: \t{self.successProbabilites}")
      logging.info("\tBinaryDistribution: Successfully fit to inputted data")

  def predict(self, x, cls):
      encodedValue = self.dataEncoder.transform([x])[0]
      logging.debug(f"\tBinaryDistrubtion : Predicting : Encoded revieved value {x} as {encodedValue}")
      # Determine the encoded value of the feature x
      probabilityOfXGivenCls = self.successProbabilites[cls]
      # P(x | cls) = (P(x == 1 | cls))(x) + (1 - P(x == 1 | cls))(1-x)
      calculatedProbabiltiy = probabilityOfXGivenCls*encodedValue + (1 - probabilityOfXGivenCls)*(1 - encodedValue)
      logging.debug(f"\tBinaryDistrubtion : Predicting : P({x}|{cls}) = {calculatedProbabiltiy}")
      return calculatedProbabiltiy

class MultiModelDistribution(LikelihoodModel):
  
  dataEncoder = None
  successProbabilites = [] 
  DAMPENING_FACTOR = 0.01

  def __init__(self, inputData: List, EncodedOutcomes: List, numberOfOutcomes: int):

      # Convert the labeled input into an enumerated list by encoding it with a label encoder.
      self.dataEncoder = preprocessing.LabelEncoder()
      self.dataEncoder.fit(inputData)
      super().__init__(inputData, EncodedOutcomes, numberOfOutcomes)

  def fitToData(self):

      logging.info(f"\tMultiModal: Converting data from labeled to encoded data with a totoal of {len(self.dataEncoder.classes_)} labels for : {self.dataEncoder.classes_}")
      # Encode the data into a binary 0, 1
      self.dataOutcomesTbl['data'] = pd.Series(self.dataEncoder.transform(self.dataOutcomesTbl['data']))

      # For each class, y(i), we find a distribution of probabilities, [p1, p2, ..., pn] where n is the number of labels.
      # self.successProbabilites = numpy.array
      self.successProbabilites = [ [0 for n in range(len(self.dataEncoder.classes_))] for x in range(self.numberOfOutcomes)]
      logging.info(f"\tMultiModal: Initalized a 2d array of success probabilities of size ({len(self.dataEncoder.classes_)} labels by {self.numberOfOutcomes} outcomes)")

      # For each label, get the number of times X == label when y(i) is true, and get the number of times y(i) is true in totoal.
      logging.info(f"\tMultiModal: Looping through each label in the dataset and calculating probability of success for each outcome...")
      for label, name in enumerate(self.dataEncoder.classes_):
          # Get the true positive for (x == label | y(i)) and the count of y(i)
          truePositiveCount = self.dataOutcomesTbl.groupby("outcomes")[['data']].agg(lambda values : len([x for x in values if x == label]))
          truePositiveCount.columns = ["TruePositive_Count"]
          totalPositiveCount = self.dataOutcomesTbl.groupby("outcomes")[['data']].agg(lambda values : len(values))
          totalPositiveCount.columns = ["TotalPositive_Count"]
          
          for outcome in range(self.numberOfOutcomes):
              self.successProbabilites[outcome][label] = (truePositiveCount.iloc[outcome][0] + self.DAMPENING_FACTOR) / (totalPositiveCount.iloc[outcome][0] + len(self.dataOutcomesTbl)*self.DAMPENING_FACTOR)
      logging.info("\tMultiModal: Calculated the success probabilities for each label for each class")
      logging.debug(f"\t{self.successProbabilites}")
      logging.info("\tMutliModal: Successfully fit to inputted data")

  def predict(self, x, cls):
      
      encodedValue = self.dataEncoder.transform([x])[0]
      logging.debug(f"\tMutliModal : Predicting : Encoded revieved value {x} as {encodedValue}")
      # Determine the encoded value of the feature x
      probabilityOfXGivenCls = self.successProbabilites[cls][encodedValue]
      # P(x | cls) = (P(x == 1 | cls))(x) + (1 - P(x == 1 | cls))(1-x)
      calculatedProbabiltiy = probabilityOfXGivenCls
      logging.debug(f"\tMutliModal : Predicting : P({x}|{cls}) = {calculatedProbabiltiy}")
      return calculatedProbabiltiy


class GaussianDistribution(LikelihoodModel):
  
  distributionStats : pd.DataFrame

  def fitToData(self):

      logging.info(f"\tGaussian: Calculating the standard devidation and mean of data for each outcome...")

      # For each class, find the standard deviation and mean of values when that class is the outcome.
      groupedData = self.dataOutcomesTbl.groupby('outcomes').agg(['std', 'mean'])
      self.distributionStats = groupedData
      logging.info("\tGuassian: Successfully calculated standard deviation and mean for each outcome")
      logging.debug(f"\t\n{self.distributionStats}")
      logging.info("\tGuassian: Successfully fit to inputted data")

  def predict(self, x, cls):
      
      # Grab the row of self.DistrubtionStats that corresponds to the given class.
      # Calulate the probability using the GaussianDistribution Formula;
      row = self.distributionStats.loc[cls]
      standardDeviation, mean = row[0], row[1]
      # P(x | y) == (1/sqrt(2 * pi * std))*exp(-(x - mean)^2/(2*std^2))
      calculatedProbability = (1/float(math.sqrt(2 * math.pi * standardDeviation**2))) * math.exp(-(float((x - mean)**2) / float(2 * (standardDeviation ** 2))))
      logging.debug(f"\tGuassian: Predicting P({x} | {cls}) = {calculatedProbability}")
      return calculatedProbability

class NaiveBayesModel:
  """
  Takes in a multi-feature dataset, where each feature can be modeled in one of three ways:
      - Binary Distrubution
      - Multi-Modal Distribution
      - Gaussian Distribution

  And, a list of outcomoes for the mutli-feature dataset.
  Stores this as a model, to use in the predict() method. 
    
  ### Agruments:
  Traning Data : A multi-feature dataframe with C columns and R Rows.
  TraningDataClassifications : A list of C Distribution types that define the distirubtion of each column.
  TraningOutcomes : A list of R labeled outcomes for each of the training data instances.
  
  ### Process : 
  1. Calcualate the prior for each class. 
  2. Calculate the Likelihood for each feature using each features's designated distribution type.
  """

  outcomeEncoder = None
  priorList = []
  likelihoodModels : List[LikelihoodModel] = []
  numberOfClasses = 0

  def fit(self, trainingData : pd.DataFrame, trainingDataDistributionTypes : List[DistributionType], trainingClasses : List) -> List :
      
      # Error checking. Ensure that the number of columns in the traning data is equal to the number of training data classifications
      if (len(trainingData.columns) != len(trainingDataDistributionTypes)):
          raise ValueError("Training data classifications must have the same number of elements as the number of columns in the training data.")

      logging.info(f"Creating a naive bayes model for:")
      logging.info(f"\n{trainingData}")

      # Encode the training outcomes using a label encoder
      self.outcomeEncoder = preprocessing.LabelEncoder()
      self.outcomeEncoder.fit(trainingClasses)
      self.numberOfClasses = len(self.outcomeEncoder.classes_)
      logging.info(f"Number of outcomes : {self.numberOfClasses}")

      # Prepare the Prior list and the likelihoods
      self.priorList = [0 for x in range(self.numberOfClasses)]

      # Encode outcomes into a list of integers starting from 0. 
      encodedOutcomes = self.outcomeEncoder.transform(trainingClasses)
      logging.info("Encoded outcomes")
      logging.info(encodedOutcomes)

      # Calculate the Prior for each outcome
      logging.info("Evaluating priors...")
      for outcome in range(0, self.numberOfClasses):
          # Prior = probability of outcome.
          self.priorList[outcome] = float(sum(1 for val in filter(lambda val : val == outcome, encodedOutcomes))) / float(len(trainingClasses))

      logging.info(f"Priors (size = {len(self.priorList)})")
      logging.info(f"{self.priorList}")

      # Calculate the likelihood model for each feature.
      for index, column in enumerate(trainingData.columns):
          
          data = trainingData[column].to_list()

          # Use the training model assigned in the list.
          model = None
          chosenDistType = trainingDataDistributionTypes[index]
          logging.info("")
          if chosenDistType == DistributionType.BinaryDST:
              logging.info(f"For feature #{index + 1} {column}, using BinaryDST Model")
              model = BinaryDistrubtion(data, encodedOutcomes, self.numberOfClasses)
          elif chosenDistType == DistributionType.GaussianDST:
              logging.info(f"For feature #{index + 1} {column}, using GuassianDST Model")
              model = GaussianDistribution(data, encodedOutcomes, self.numberOfClasses)
          else:
              logging.info(f"For feature #{index + 1} {column}, using MultiModalDST Model")
              model = MultiModelDistribution(data, encodedOutcomes, self.numberOfClasses)
          
          self.likelihoodModels.append(model)


  def predict(self, testingData : pd.DataFrame) -> List: 
      # Aim : Use the models from fitting to predict the outcome of each row in the testingData.
      # Calculation:
      #   For each row,
      #       For each possible outcome.
      #           calculate the relative probability of (row | outcome) = P(row[0] | outcome) * P(row[1] | outcome) * ... * P(row[n] | outcome) * Prior(outcome)
      #           and choose the outcome with the highest probability
      #           Store the outcome with the highest probability in the ouptut list

      output = [-1 for x in range(len(testingData))]
      logging.info("Determining the outcome with the highest probability of occuring with each row in the testingData...")
      for rowIndex in range(len(testingData)): # For each row,
          logging.debug(f"Evaluating Row {rowIndex}")
          dataRow = testingData.iloc[rowIndex] # Grab the row data.
          classChosen = None # Outcome Chosen = the outcome with the highest probability of success
          classChosenProbability = -1 
          for outcome in range(self.numberOfClasses): # For each possible outcome,
              
              # Calculate the likelihood of each row elements given the outcome
              calculatedLikelihoods = []
              for index, feature in enumerate(dataRow.values):
                  calculatedLikelihoods.append(self.likelihoodModels[index].predict(feature, outcome))
              
              # Aggregate those values together to find the calculated probability,
              # Repalce the current outcome chosen if the found probability is higher than the record.
              calculatedProbability = reduce(lambda x, y: x * y, calculatedLikelihoods) * self.priorList[outcome]
              if classChosenProbability < calculatedProbability:
                  classChosen = outcome
                  classChosenProbability = calculatedProbability
          output[rowIndex] = classChosen

          
      logging.info("Successfully determined the most likely outcome for each row in the input data.")
      return self.outcomeEncoder.inverse_transform(output)
