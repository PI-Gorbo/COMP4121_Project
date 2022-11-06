import logging
from Models.Helpers import Dataset, Helpers, PreparedDatasets
from SvmModel import SvmModel
class Runner:

    @staticmethod
    def run(dataset : Dataset):
        logging.basicConfig(level=logging.INFO)
        trainingData, testingData, trainingOutcomes, testingOutcomes, dataclassification = PreparedDatasets.get(dataset)
        model = SvmModel()
        model.fit(trainingData, dataclassification, trainingOutcomes)
        # predictions = model.predict(testingData)
        Helpers.PrintMetrics(predictions, testingOutcomes, "Suppport Vector Machine")