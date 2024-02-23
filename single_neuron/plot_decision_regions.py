import numpy as np
import matplotlib.pyplot as plt

def plot_decision_regions(X, y, clf, resolution=0.02):
    """
    Plot decision regions of a classifier.

    Parameters
    ----------
    X : numpy.ndarray
        The feature matrix.

    y : numpy.ndarray
        The target vector.

    clf : object
        Classifier object with a 'predict' method.

    resolution : float, optional
        The resolution of the grid for plotting. Default is 0.02.
    """

    # Define markers and colors for the plot
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = plt.cm.jet

    # Extract min and max values for the two features and create a meshgrid
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))

    # Predict the class labels for each combination in the grid
    Z = clf.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    # Plot the regions
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # Plot the data points
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')
    
    plt.show()
