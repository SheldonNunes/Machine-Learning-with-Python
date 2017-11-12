from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
import numpy as np
import mglearn
from sklearn.model_selection import train_test_split
def printScores(model_type, training_score, testing_score):
	print("Results from {}".format(model_type))
	print("Training set score: {:.2f}".format(training_score))
	print("Testing set score: {:.2f}".format(testing_score))

X, y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

lr = LinearRegression().fit(X_train, y_train)
printScores("Linear Regression", lr.score(X_train, y_train), lr.score(X_test, y_test))
print("\n")

ridge = Ridge().fit(X_train, y_train)
printScores("Ridge Regression", ridge.score(X_train, y_train), ridge.score(X_test, y_test))
print("\n")

ridge10 = Ridge(alpha=10).fit(X_train, y_train)
printScores("Ridge Regression with 10 alpha", ridge10.score(X_train, y_train), ridge.score(X_test, y_test))
print("\n")

ridge01 = Ridge(alpha=0.1).fit(X_train, y_train)
printScores("Ridge Regression with 0.1 alpha", ridge01.score(X_train, y_train), ridge01.score(X_test, y_test))
print("\n")

lasso = Lasso().fit(X_train, y_train)
printScores("Lasso Regression", lasso.score(X_train, y_train), lasso.score(X_test, y_test))
print("Number of features used: {}".format(np.sum(lasso.coef_ != 0)))
print("\n")

lasso001 = Lasso(alpha=0.01, max_iter=100000).fit(X_train, y_train)
printScores("Lasso with 0.01 alpha", lasso001.score(X_train, y_train), lasso001.score(X_test, y_test))
print("Number of features used: {}".format(np.sum(lasso001.coef_ != 0)))
print("\n")

lasso00001 = Lasso(alpha=0.0001, max_iter=100000).fit(X_train, y_train)
printScores("Lasso with 0.0001 alpha", lasso00001.score(X_train, y_train), lasso00001.score(X_test, y_test))
print("Number of features used: {}".format(np.sum(lasso00001.coef_ != 0)))
print("\n")
