#Space for Kentaro to try things. Do not run.
import numpy as np
import matplotlib.pyplot as plt
from metric_learn import LMNN
from metric_learn import LSML_Supervised
from sklearn.datasets import load_iris
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA


iris_data = load_iris()
X = iris_data['data'] #Data vector
Y = iris_data['target'] #Class Labels

lmnn = LMNN(k=5, learn_rate=1e-6)
lmnn.fit(X, Y)
#print(lmnn.metric())

X_t = lmnn.transform(X=X)

#print(X[1,])
#print(X_t[1,])
#diff = abs(X - X_t)
#print(sum(sum(diff)))

'''
fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
X_reduced = PCA(n_components=3).fit_transform(X)
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=Y,
           cmap=plt.cm.Paired)
ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])

plt.show()
'''

from sklearn.neighbors import KNeighborsClassifier
np.random.seed(0)
indices = np.random.permutation(len(X))
iris_X_train = X[indices[:-50]]
iris_y_train = Y[indices[:-50]]
iris_X_test  = X[indices[-50:]]
iris_y_test  = Y[indices[-50:]]

knn = KNeighborsClassifier()
knn.fit(iris_X_train, iris_y_train) 

Guess = knn.predict(iris_X_test)
print("accuracy for original:")
print(sum(abs(Guess-iris_y_test))/len(iris_y_test))

np.random.seed(0)
indices = np.random.permutation(len(X))
iris_X_train = X_t[indices[:-50]]
iris_y_train = Y[indices[:-50]]
iris_X_test  = X_t[indices[-50:]]
iris_y_test  = Y[indices[-50:]]

knn = KNeighborsClassifier()
knn.fit(iris_X_train, iris_y_train) 

Guess = knn.predict(iris_X_test)
print("accuracy for transofrmed")
print(sum(abs(Guess-iris_y_test))/len(iris_y_test))


from sklearn.datasets import load_digits
digits = load_digits()



import numpy as np
import matplotlib.pyplot as plt
from metric_learn import LMNN
from metric_learn import LSML_Supervised
from sklearn.datasets import load_iris
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA


iris_data = load_iris()
X = iris_data['data'] #Data vector
Y = iris_data['target'] #Class Labels

lmnn = LMNN(k=5, learn_rate=1e-6)
lmnn.fit(X, Y)
#print(lmnn.metric())

X_t = lmnn.transform(X=X)

#print(X[1,])
#print(X_t[1,])
#diff = abs(X - X_t)
#print(sum(sum(diff)))

'''
fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
X_reduced = PCA(n_components=3).fit_transform(X)
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=Y,
           cmap=plt.cm.Paired)
ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])

plt.show()
'''

from sklearn.neighbors import KNeighborsClassifier
np.random.seed(0)
indices = np.random.permutation(len(X))
iris_X_train = X[indices[:-50]]
iris_y_train = Y[indices[:-50]]
iris_X_test  = X[indices[-50:]]
iris_y_test  = Y[indices[-50:]]

knn = KNeighborsClassifier()
knn.fit(iris_X_train, iris_y_train) 

Guess = knn.predict(iris_X_test)
print("accuracy for original:")
print(sum(abs(Guess-iris_y_test))/len(iris_y_test))

np.random.seed(0)
indices = np.random.permutation(len(X))
iris_X_train = X_t[indices[:-50]]
iris_y_train = Y[indices[:-50]]
iris_X_test  = X_t[indices[-50:]]
iris_y_test  = Y[indices[-50:]]

knn = KNeighborsClassifier()
knn.fit(iris_X_train, iris_y_train) 

Guess = knn.predict(iris_X_test)
print("accuracy for transofrmed")
print(sum(abs(Guess-iris_y_test))/len(iris_y_test))


from sklearn.datasets import load_digits
digits = load_digits()



