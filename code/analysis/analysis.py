import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def summarize(X, y, decimals=4):

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

    #2 dimensional
    # pca = PCA(n_components=2)
    # pca.fit(X)
    # pca_trans = pca.transform(X)
    

    # plt.scatter(pca_trans[:,0], pca_trans[:,1], c=y)
    # plt.title("PCA")
    # plt.show()

    #plt.scatter(X[:,0], X[:,1], c=y)
    #plt.show()
    #return

    #three dimensional
    pca = PCA(n_components=3)
    pca.fit(X)
    pca_trans = pca.transform(X)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(pca_trans[:,0], pca_trans[:,1], pca_trans[:,2], c=y)
    plt.title("PCA")
    plt.show()


