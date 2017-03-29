from metric_learn import LSML_Supervised
from sklearn.datasets import load_iris

iris_data = load_iris()
X = iris_data['data']
Y = iris_data['target']

lsml = LSML_Supervised(num_constraints=200)
lsml.fit(X, Y)
X_t = lsml.transform(X=X)
print( "Average Change per Dimension:", sum(abs(X-X_t))/len(X))