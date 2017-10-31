import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def plotDataSet()
{
        iris_dataset = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

        # create dataframe from data in X_train
        # label the columns using the string in iris_dataset.feature_names
        iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
        # create a scatter matrix from the dataframe, colour by y_train
        ts = pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15,15), marker='o', hist_kwds={'bins' : 20}, s=60, alpha=.8, cmap=mglearn.cm3)
        plt.show()
}

plotDataSet()

knn = KNeighborsClassifier(n_neighbors=1)
# train algorithm
knn.fit(X_train, y_train)

KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, n_jobs=1, n_neighbors=1, p=2, weights='uniform')

# run predictions agains the test set
y_pred = knn.predict(X_test)
percentage_correct = knn.score(X_test, y_test) * 100

print('The algorithm is predicting with {}%  accuracy'.format(percentage_correct))
