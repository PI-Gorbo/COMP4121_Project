'''
# To do:
    1. Create node class with two subclasses, LeafNode and DecisionNode.
    1a. Decision Nodes have the properties : featureIndex (feature splitting on) 
                                             threshold, (value splitting on)
                                             Left, Right, (left dataframe, right dataframe)
                                             InformationGain (Stores the information gained by the split.)
    1b. Leaf nodes have a single property : value (the class that has been assigned to the node.)

    2. Create the fit and predict functions
        Keep track of:
            root, minimumSamplesToSplit, maximumTreeDepth
    2a. fit(trainingData, trainingOutcomes, minimumSamplesToSplit, maximumTreeDepth)

        def recursiveBuildTree(self, dataset, outcomes, curr_depth = 0):
        - Create a recursive function called build tree which returns a the root node of a new tree.
            - To build the tree, keep splitting untill the number of samples < minimumSamplesToSplit or current depth >= max depth
                - Find the best spit
                    - The best split is the split that maximises the information gain.
                        - For each feature in the dataset, and for each value in that feature, we split on that value.
                            - To Split, group all rows that have the feature above the chosen value into one group, and then all the other rows into another.
                    - Record the information gain from the split (gini or entropy)
                    - Choose the split that maximises the information gain.
                - If the best split info gain > 0 keep splitting, (if not we have created a pure leaf node)
                    - Recurse left, (splitting data and outcomes)
                    - Recurse right, (splitting data and outcomes)
                    - return the Node with the left and right trees
            - If we get to here, then the node is a leaf node. We return the node with the mode of the outcomes we have

    2b. To predict, we now just take each row and categorize it using a recursive, makePrecition() function
        - Given a tree, and a row, we can check if the tree is a Leaf node. If it is, return the leaf node's class
        - If the tree is not a leaf node, follow its feature and value and choose the left if less than or equal, or the right if greater than or equal.
'''
import enum
import logging
import math
from typing import List
import pandas as pd
import numpy as np
from Models.Helpers import StdDistributionType, Helpers

class InformationGainMetric(enum.Enum):
    Gini = 1
    Entropy = 2

class Node:
    pass

class LeafNode(Node):
    
    def __init__(self, classValue : int) -> None:
        self.classValue = classValue

class DecisionNode(Node):
    def __init__(self, featureName : str, threshold : float, left : Node, right : Node, informationGain : float) -> None:
        self.featureName = featureName
        self.threshold = threshold
        self.left = left
        self.right = right
        self.informationGain = informationGain

class SplitInfo():

    def __init__(self, featureName : str, threshold : float, informationGain : float, left : pd.DataFrame, leftOutcomes : List, right : pd.DataFrame, rightOutcomes : List) -> None:
        self.featureName = featureName
        self.threshold = threshold
        self.informationGain = informationGain 
        self.left = left
        self.leftOutcomes = leftOutcomes
        self.right = right
        self.rightOutcomes = rightOutcomes

class DecisionTreeModel:
    
    def fit(self, trainingData : pd.DataFrame, trainingDataClasses : List[StdDistributionType], trainingOutcomes : List, minimumSamplesToSplit : int, maximumTreeDepth : int, informationGainMetric : InformationGainMetric) -> None:
        
        def calculateDescreteProbabilities(outcomeList) -> List:
            uniqueOutcomes, uniqueOutcomesCount = np.unique(outcomeList, return_counts=True)
            probabilities = uniqueOutcomesCount / (len(outcomeList)) # Devides every element in the uniqueOutcomesCount list by len(...)
            logging.debug(f"Caclulating descrete probabilities of list {uniqueOutcomes} -> {[probabilities]}")
            return probabilities

        # Transforms x into the form required to calculate entropy
        def individualElementEntropyTransform(x : float) -> float:
            return -1 * x * math.log2(x)

        # Transforms x info the form required to calculate the gini index
        def individualElementGiniTransform(x : float) -> float:
            return math.pow(x,2)

        def calculateInformationGain(dataOutcomes : List, leftOutcomes : List, rightOutcomes : List) -> float:

            # Find probability of each unique outcome.
            parentProbabilityList = calculateDescreteProbabilities(dataOutcomes)
            leftProbabilityList = calculateDescreteProbabilities(leftOutcomes)
            rightProbabilityList = calculateDescreteProbabilities(rightOutcomes)
            weightForLeft = float(len(leftOutcomes)) / float(len(dataOutcomes))
            weightForRight = float(len(rightOutcomes)) / float(len(dataOutcomes))

            # The information gain can be calculated two ways, using entropy or gini.
            if (informationGainMetric == InformationGainMetric.Entropy): #Entropy = sum of p_i * log_2(p_i) where p_i is the probability of each element in the probability list.

                entropyForData = sum(map(individualElementEntropyTransform, parentProbabilityList))
                entropyForLeft = sum(map(individualElementEntropyTransform, leftProbabilityList)) * weightForLeft
                entropyForRight = sum(map(individualElementEntropyTransform, rightProbabilityList)) * weightForRight
                logging.debug(f"Determied Entropy to be : Parent({entropyForData}) - (Left({entropyForLeft}) + Right({entropyForRight})) = {entropyForData - (entropyForLeft + entropyForRight)}")
                return entropyForData - (entropyForLeft + entropyForRight)
            else:
                
                giniForData = 1 - sum(map(individualElementGiniTransform, parentProbabilityList))
                giniForLeft = (1 - sum(map(individualElementGiniTransform, leftProbabilityList))) * weightForLeft
                giniForRight = (1 - sum(map(individualElementGiniTransform, rightProbabilityList))) * weightForRight
                return giniForData - (giniForLeft + giniForRight)

        def splitData(data : pd.DataFrame, outcomes : List, featureName : str, threshold : float) -> SplitInfo:

            # Combine data and outcomes to allow for easy splitting.
            data["Outcome"] = outcomes

            # Split the data
            leftCombined = data[data[featureName] <= threshold]
            rightCombined = data[data[featureName] > threshold]

            # Re-separate the data
            leftOutcomes = leftCombined["Outcome"]
            leftData = leftCombined.drop("Outcome", axis=1)
            rightOutcomes = rightCombined["Outcome"]
            rightData = rightCombined.drop("Outcome", axis=1)

            # Return a splitinfo object, which requires a caculation of the infromation gain
            output = SplitInfo(
                featureName=featureName,
                threshold=threshold,
                informationGain=calculateInformationGain(outcomes, leftOutcomes, rightOutcomes),
                left = leftData,
                leftOutcomes=leftOutcomes.tolist(),
                right=rightData,
                rightOutcomes=rightOutcomes.tolist()
            )
            return output

        # A function that returns a splitInfo object. 
        # The split info object contains information about the optimal split for the given dataset and outcomes.
        # A split is quantified by looking at the information gain of the split. 
        # Infromation Gain = InformationGainMetric(Parent)
        def findBestSplit(data : pd.DataFrame, outcomes : List) -> SplitInfo:
            
            logging.debug("Finding best split")
            bestSplit : SplitInfo = None
            bestSplitInformationGain = -float("inf")

            # Loop through all the features in the data, and all the unique values in those features.
            # and find the spilt with the maximum information gain
            for featureName in data.columns:
                # Loop through every value in the feature
                valueSet = data[featureName].unique()
                for threshold in valueSet:
                    logging.debug(f"Splitting on feature {featureName} on threshold {threshold} (out of {len(valueSet)} unique values)")

                    # Split on the threshold of the feature, and find the information gain of the split
                    split : SplitInfo = splitData(data, outcomes, featureName, threshold)
                    if (split.informationGain > bestSplitInformationGain):
                        bestSplit = split
                        bestSplitInformationGain = split.informationGain
            
            return bestSplit


        def buildTree(data : pd.DataFrame, outcomes : List, currentDepth : int = 0) -> Node :
            
            # Base Case, data is under the minimum conditions to split, so we return a leaf node with the most common outcome.
            if (len(data) < minimumSamplesToSplit or currentDepth > maximumTreeDepth):
                logging.debug("Reached minimum sample limit or depth limit.")
                MostCommonClass = max(outcomes, key=outcomes.count)
                logging.debug(f"Seting most common class for this node to be {MostCommonClass}")
                return LeafNode(MostCommonClass)
            
            # Recursive case, the data meets the conditions to split.
            bestSplit : SplitInfo = findBestSplit(data, outcomes)
            if (bestSplit.informationGain > 0):
                
                # If the information gain > 0, then we have found a split that improves the information gain.
                # We should recurse on the left and right halves
                logging.debug("Splitting...")
                outputDecisionNode = DecisionNode(
                    bestSplit.featureName,
                    bestSplit.threshold,
                    buildTree(bestSplit.left, bestSplit.leftOutcomes, currentDepth + 1),
                    buildTree(bestSplit.right, bestSplit.rightOutcomes, currentDepth + 1),
                    bestSplit.informationGain
                )
                return outputDecisionNode
            
            else: # If the information gain is equal to zero, then we should just return a leaf node as there is no point in splitting further.
                outcomesAsList = list(outcomes)
                MostCommonClass = max(outcomesAsList, key=outcomesAsList.count)
                return LeafNode(MostCommonClass)
                
        
        # Store important variables for later.
        self.trainingDataClasses = trainingDataClasses
        self.minimumSamplesToSplit = minimumSamplesToSplit
        self.maximumTreeDepth = maximumTreeDepth

        # Encode the database and the possible outcomes
        self.encodedTable, self.columnEncoders = Helpers.EncodeTable(trainingData, trainingDataClasses)
        _encodedOutcomes, _outcomeEncoder = Helpers.EncodeOutcomes(trainingOutcomes)
        self.encodedOutcomes = _encodedOutcomes
        self.outcomeEncoder = _outcomeEncoder
        logging.debug("Constructing Tree...")
        self.constructedTree = buildTree(self.encodedTable, self.encodedOutcomes)
        logging.debug("Successfully constructed tree")
        self.printTree(self.constructedTree)

    def printTree(self, tree = None, indent = " "):

        if tree == None:
            tree = self.constructedTree
        
        if isinstance(tree, LeafNode):
            logging.debug(f"{indent}Leaf: ")
            logging.debug(f"{indent}{tree.classValue}")
        else:
            decisionNode : DecisionNode = tree
            logging.debug(f"{indent}Decision")
            logging.debug(f"{indent}X[{decisionNode.featureName}] <= {decisionNode.threshold} produces a gain of {decisionNode.informationGain}")
            logging.debug(f"{indent}Left: ")
            self.printTree(decisionNode.right, indent + indent)
            logging.debug(f"{indent}Right: ")
            self.printTree(decisionNode.left, indent + indent)

    def predict(self, testingData : pd.DataFrame) -> List:
        
        def calculatePredictedClass(parentNode : Node, row : pd.Series) -> int:

            # Base case
            if isinstance(parentNode, LeafNode):
                return parentNode.classValue
            elif isinstance(parentNode, DecisionNode):
                
                if (float(row[parentNode.featureName]) <= parentNode.threshold):
                    return calculatePredictedClass(parentNode.left, row)
                else:
                    return calculatePredictedClass(parentNode.right, row)


        # For each row in the etsing data, we need to find where on the decision tree the data sits.
        predictedValues = []
        for index, row in testingData.iterrows():

            # Find the value of the row.
            predictedValues.append(calculatePredictedClass(self.constructedTree, row))
        
        return self.outcomeEncoder.inverse_transform(predictedValues)