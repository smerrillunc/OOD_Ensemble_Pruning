from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    log_loss,
    confusion_matrix
)
import numpy as np

class EnsembleMetrics:
    def __init__(self, true_labels, predictions, probabilities=None):
        self.true_labels = true_labels
        self.predictions = predictions
        self.probabilities = probabilities

    def accuracy(self):
        return accuracy_score(self.true_labels, self.predictions)

    def precision(self):
        return precision_score(self.true_labels, self.predictions)

    def recall(self):
        return recall_score(self.true_labels, self.predictions)

    def f1(self):
        return f1_score(self.true_labels, self.predictions)

    def auc(self):
        if self.probabilities is not None:
            try:
                return roc_auc_score(self.true_labels, self.probabilities)
            except:
                return np.nan
        else:
            raise ValueError("Probabilities must be provided for AUC calculation.")

    def mean_absolute_error(self):
        return mean_absolute_error(self.true_labels, self.predictions)

    def mean_squared_error(self):
        return mean_squared_error(self.true_labels, self.predictions)

    def log_loss(self):
        if self.probabilities is not None:
            try:
                return log_loss(self.true_labels, self.probabilities)
            except:
                return np.nan
        else:
            raise ValueError("Probabilities must be provided for log loss calculation.")

    def confusion_matrix(self):
        return confusion_matrix(self.true_labels, self.predictions)

if __name__ == '__main__':
    # Usage Example
    true_labels = [0, 1, 1, 0, 1]  # Actual labels
    predictions = [0, 1, 0, 0, 1]  # Predicted labels
    probabilities = [0.2, 0.8, 0.4, 0.1, 0.9]  # Predicted probabilities

    metrics = EnsembleMetrics(true_labels, predictions, probabilities)

    print("Accuracy:", metrics.accuracy())
    print("Precision:", metrics.precision())
    print("Recall:", metrics.recall())
    print("F1 Score:", metrics.f1())
    print("AUC:", metrics.auc())
    print("Mean Absolute Error:", metrics.mean_absolute_error())
    print("Mean Squared Error:", metrics.mean_squared_error())
    print("Log Loss:", metrics.log_loss())
    print("Confusion Matrix:\n", metrics.confusion_matrix())
