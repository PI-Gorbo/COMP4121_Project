from typing import List


class RunnerHelpers:
    
    @staticmethod
    def CalculateAccuracy(predictions : List, actual : List) -> float:
        
        if (len(predictions) != len(actual)):
            raise ValueError("Predictions array must be the same length as the actual array")

        # calculate total predictions and true positive predictions.
        totalPredictions = len(predictions)
        totalTruePositives = len([1 for index in range(len(predictions)) if predictions[index] == actual[index]])
        return float(totalTruePositives) / float(totalPredictions)