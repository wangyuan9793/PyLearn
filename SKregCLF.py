import sklearn
from sklearn import datasets
from sklearn import linear_model
import numpy as np
import subprocess


boston_x = datasets.load_boston().data
boston_y = datasets.load_boston().target


indices = np.random.permutation(len(boston_y))
# iris_X_train = iris_X[indices[:-10]]
# iris_y_train = iris_y[indices[:-10]]
# iris_X_test  = iris_X[indices[-10:]]
# iris_y_test  = iris_y[indices[-10:]]

train_x = boston_x[indices[:-100]]
train_y = boston_y[indices[:-100]]

test_x = boston_x[indices[-100:]]
test_y = boston_y[indices[-100:]]

regr = linear_model.LinearRegression()

regr.fit(train_x,train_y)

MSE = np.mean(np.square(regr.predict(test_x)-test_y))



print(regr.score(train_x,train_y))
print(regr.score(test_x,test_y))


from sklearn import tree
data_X = datasets.load_breast_cancer().data
data_Y = datasets.load_breast_cancer().target

train_x = data_X[indices[:-100]]
train_y = data_Y[indices[:-100]]

test_x = data_X[indices[-100:]]
test_y = data_Y[indices[-100:]]

# print(np.unique(train_x),train_x.shape)
dt = tree.DecisionTreeClassifier(min_samples_split=10,max_depth=5)
dt.fit(train_x,train_y)



print(dt.score(test_x,test_y))

from os import system

tree.export_graphviz(dt,"dt.dot")
system("dot -Tpng dt.dot -o dt.png")


# command = ["./dt.dot"]

# subprocess.check_call(command)