import numpy as np


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


