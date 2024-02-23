import numpy as np
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt

class SingleNeuron:
    """
    Represents a single artificial neuron.

    Attributes
    ----------
    activation_function : callable
        The function applied to the neuron's output (pre-activation).
        Example: sigmoid, tanh, relu, etc.

    cost_function : callable
        The loss function used to measure model performance during training.
        Example: mean squared error, cross entropy, etc.

    w_ : numpy.ndarray
        Weights vector for the neuron, with the last entry being the bias.

    errors_ : list of float
        Mean squared error for each epoch during training.

    Methods
    -------
    train(X, y, alpha=0.005, epochs=50) -> "SingleNeuron":
        Trains the neuron using stochastic gradient descent.

    predict(X: numpy.ndarray) -> numpy.ndarray:
        Predicts the output using the trained neuron.

    plot_cost_function():
        Plots the cost function across epochs.

    plot_decision_boundary(X, y, xstring="x", ystring="y"):
        Plots the decision boundary if the neuron is used for binary classification.
    """

    def __init__(self, activation_function: callable, cost_function: callable):
        self.activation_function = activation_function
        self.cost_function = cost_function
        self.w_ = None
        self.errors_ = []

    def train(self, X: np.ndarray, y: np.ndarray, alpha: float = 0.005, epochs: int = 50) -> "SingleNeuron":
        """
        Trains the neuron on the provided data.

        Parameters
        ----------
        X : numpy.ndarray
            The feature matrix with shape (number_samples, number_features).

        y : numpy.ndarray
            The target vector with shape (number_samples,).

        alpha : float, optional
            The learning rate. Default is 0.005.

        epochs : int, optional
            Number of times the entire dataset is shown to the model. Default is 50.

        Returns
        -------
        SingleNeuron
            The trained neuron.
        """
        self.w_ = np.random.randn(1 + X.shape[1])
        N = X.shape[0]

        for _ in range(epochs):
            errors = 0
            for xi, target in zip(X, y):
                update = alpha * (self.predict(xi) - target)
                self.w_[:-1] -= update * xi
                self.w_[-1] -= update
                errors += self.cost_function(self.predict(xi), target)
            self.errors_.append(errors / N)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the output for the given input data.

        Parameters
        ----------
        X : numpy.ndarray
            The feature matrix to predict on.

        Returns
        -------
        numpy.ndarray
            Predictions for the input samples.
        """
        z = np.dot(X, self.w_[:-1]) + self.w_[-1]
        return self.activation_function(z)

    def plot_cost_function(self):
        """Plots the cost across epochs."""
        fig, axs = plt.subplots(figsize=(10, 8))
        axs.plot(range(1, len(self.errors_) + 1), self.errors_, label="Cost function")
        axs.set_xlabel("Epochs", fontsize=15)
        axs.set_ylabel("Cost", fontsize=15)
        axs.legend(fontsize=15)
        axs.set_title("Cost Across Epochs During Training", fontsize=18)
        plt.show()

    def plot_decision_boundary(self, X, y, xstring="x", ystring="y"):
        """
        Plots the decision boundary for 2D data.

        Parameters
        ----------
        X : numpy.ndarray
            The feature matrix.

        y : numpy.ndarray
            The target vector.

        xstring : str, optional
            Label for the x-axis. Default is "x".

        ystring : str, optional
            Label for the y-axis. Default is "y".
        """

        plt.figure(figsize=(10, 8))
        plot_decision_regions(X, y, clf=self)
        plt.title("Neuron Decision Boundary", fontsize=18)
        plt.xlabel(xstring, fontsize=15)
        plt.ylabel(ystring, fontsize=15)
        plt.legend(loc='upper left')
        plt.show()
