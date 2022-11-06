import logging
from Models.DecisionTree.DecisionTreeModel import DecisionTreeModel, InformationGainMetric
from Models.Helpers import Dataset, Helpers, PreparedDatasets

class Runner:

    def run(self, dataset : Dataset):
        logging.basicConfig(level=logging.INFO)
        trainingData, testingData, trainingOutcomes, testingOutcomes, dataclassification = PreparedDatasets.get(dataset)
        model = DecisionTreeModel()
        model.fit(trainingData, dataclassification, trainingOutcomes, 2, 30, InformationGainMetric.Entropy)
        predictions = model.predict(testingData)
        Helpers.PrintMetrics(predictions, testingOutcomes, "Decision Tree")