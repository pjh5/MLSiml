import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def summarize(X, y, plot=True, decimals=4):
    """Prints summary statistics for every column of X, split into 2 classes

    X is the data, y is the class labels, assumed to be 0 or 1. This will also
    plot the data.

    Arguments:
    X           An N x K numpy ndarray of N data vectors, each of dimension K.
    y           A numpy array or size N, the class labels for X. In this case,
                y is assumed to be either 1 or 0. Summary statistics will be
                printed separately for both class labels.

    plot        If set, this will also plot a scatterplot of the data (or the
                first three principal components of the data).

    decimals    How many decimals to print every value with.


    There is no return value.
    """

    # Assumes y is either 1 or 0
    pos_idxs = np.where(y == 1)[0]
    neg_idxs = np.where(y == 0)[0]

    # Divide dataset into positive and negatives
    Xs = (X[neg_idxs, :], X[pos_idxs, :])
    Ys = (y[neg_idxs], y[pos_idxs])

    # Make format string
    numstr = ", ".join(["{" + str(i) + ":10." + str(decimals) + "f}" for i
                                                        in range(X.shape[1])])

    # Output results
    print("Total number of samples: " + str(len(y)))
    print()
    print(str(len(Ys[1])) + " Positive Samples:")
    print("\tMin   : " + numstr.format( *np.min(Xs[1], axis=0)))
    print("\tMean  : " + numstr.format(*np.mean(Xs[1], axis=0)))
    print("\tMax   : " + numstr.format( *np.max(Xs[1], axis=0)))
    print()
    print("\tStdev : " + numstr.format(*np.sqrt(np.var(Xs[1], axis=0))))
    print("\tVar   : " + numstr.format( *np.var(Xs[1], axis=0)))
    print()

    print(str(len(Ys[0])) + " Negative Samples:")
    print("\tMin   : " + numstr.format( *np.min(Xs[0], axis=0)))
    print("\tMean  : " + numstr.format(*np.mean(Xs[0], axis=0)))
    print("\tMax   : " + numstr.format( *np.max(Xs[0], axis=0)))
    print()
    print("\tStdev : " + numstr.format(*np.sqrt(np.var(Xs[0], axis=0))))
    print("\tVar   : " + numstr.format( *np.var(Xs[0], axis=0)))

    # Plot if requested
    if plot:
        plot_data(X, y)


def plot_data(X, y):
    """Shows a simple scatterplot of X, colored by the classes in y.

    Technically, this shows the 1st three principal components of X if X has
    more than 3 dimensions. If X only has 2 dimensions, then just a
    2-dimensional scatterplot is returned. This will not produce a plot for 1
    dimensional data.
    
    Arguments:
    X       An N x K numpy ndarray of N data vectors, each of dimension K
    y       A numpy array or size N, the class labels for X

    There is no return value.
    """

    x_dim = X.shape[1]

    # Ignore 1 dimensional data
    if x_dim == 1:
        print("plot_data not gonna bother with 1 dimensional data")
        return

    # For 2 dimensional data, just plot it
    if x_dim == 2:
        plt.scatter(X[:,0], X[:,1], c=y)
        plt.show()
        return

    # For at least 3 dimensions, do PCA
    pca = PCA(n_components=3)
    pca.fit(X)
    plot_x = pca.transform(X)

    # Plot the now 3 dimensional data
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(plot_x[:,0], plot_x[:,1], plot_x[:,2], c=y)
    ax.set_title("PCA of Generated Data")
    ax.set_xlabel("1st Principal Component")
    ax.set_ylabel("2nd Principal Component")
    ax.set_zlabel("3rd Principal Component")
    plt.show()


