
from Models.KNN.kNearestNeighboursRunner import Runner as KNN
from Models.LogisticRegression.logisticRegressionRunner import Runner as LogisticRegression
from Models.NaiveBayes.naiveBayesRunner import Runner as NaiveBayes
from Models.Helpers import Dataset
from Models.DecisionTree.DecisionTreeRunner import Runner as DecisionTree

if __name__ == '__main__':
    # runner = LogisticRegression()
    # runner.run(dataset=Dataset.Cancer)

    # runner = KNN()
    # runner.run(dataset=Dataset.Cancer)

    # runner = NaiveBayes()
    # runner.run(dataset=Dataset.Cancer)

    runner = DecisionTree()
    runner.run(dataset=Dataset.Cancer)