import numpy as np
from sklearn.utils import shuffle

__all__= ['cross_entropy_loss', 'train_test_split']

def cross_entropy_loss(y_hat, y):
 return - y*np.log(y_hat) - (1 - y)*np.log(1 - y_hat)

def train_test_split(X, y, percent = .15):
 X, y = shuffle(X, y)
 split_index = np.int(np.round(X.shape[0])*.15)
 X_train, y_train = X[:split_index], y[:split_index]
 X_test, y_test = X[-split_index:], y[-split_index:]
 return X_train, y_train, X_test, y_test