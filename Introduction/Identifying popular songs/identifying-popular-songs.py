import matplotlib.pyplot as plt
import numpy as np
import dataset_reader as dr
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

data_entries = dr.get_data_entries()

X_train, X_test, y_train, y_test = train_test_split(data_entries[0], data_entries[1], random_state=0)
X_train = np.asarray(X_train)
knn = KNeighborsClassifier(n_neighbors=1)
np_training = np.array(X_train)
np_labels = np.array(y_train)

knn.fit(X_train, y_train)

KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, n_jobs=1, n_neighbors=1, p=2, weights='uniform')

y_pred = knn.predict(X_test)

track_listens_prediction = [x[1] for x in y_pred]
track_listens_actual = [x[1] for x in y_test]

correctResults = 0
for index in range(len(track_listens_prediction)):
    if(abs(int(track_listens_prediction[index])-int(track_listens_actual[index]) < 300)):
        correctResults += 1

print("Result: {:.2f}".format(correctResults/len(track_listens_prediction)))
