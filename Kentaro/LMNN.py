import numpy as np
import matplotlib.pyplot as plt
from metric_learn import LMNN
from sklearn.datasets import load_iris


iris_data = load_iris()
X = iris_data['data'] #Data vector
Y = iris_data['target'] #Class Labels

lmnn = LMNN(k=5, learn_rate=1e-6)
lmnn.fit(X, Y)
X_t = lmnn.transform(X=X)
print( "Average Change per Dimension:", sum(abs(X-X_t))/len(X))


