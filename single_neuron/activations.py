import numpy as np

__all__ = ['linear_regression_activation', 'sigmoid_activation']

def linear_regression_activation(z):
    return z

def sigmoid_activation(z):
    return 1.0/(1.0 + np.exp(-z))
