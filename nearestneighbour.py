import numpy as np

class NearestNeighbourClassifier:
    def __init__(self):
        self.x = np.array([])
        self.y = np.array([])

    def fit(self, x, y):
        """ Fit the training data to the classifier.

        Args:
        x (np.ndarray): Instances, numpy array with shape (N,K)
        y (np.ndarray): Class labels, numpy array with shape (N,)
        """

        self.x = x  # Stores training data
        self.y = y

    def predict(self, x):
        """ Perform prediction given some examples.

        Args:
        x (np.ndarray): Instances, numpy array with shape (N,K)

        Returns:
        y (np.ndarray): Predicted class labels, numpy array with shape (N,)
        """

        predictions = np.zeros(len(x), dtype=self.y.dtype)

        for index, row in enumerate(x):
            distances = np.sqrt(np.sum((row - self.x) ** 2, axis=1))
            min_index = np.argmin(distances)
            predictions[index] = self.y[min_index]

        return predictions

