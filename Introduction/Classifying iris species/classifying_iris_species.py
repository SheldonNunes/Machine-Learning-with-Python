import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'], random_state=0)

fig, ax = plt.subplots(3, 3, figsize=(15,15))
plt.suptitle("iris_pairplot")

for i in range(3):
	for j in range(3):
		ax[i, j].scatter(X_train[:,j], X_train[:, i+1], c=y_train, s=60)
		ax[i, j].set_xticks(())
		ax[i, j].set_yticks(())
		if i == 2:
			ax[i, j].set_xlabel(iris['feature_names'][j])
		if j == 0:
			ax[i, j].set_ylabel(iris['feature_names'][i+1])
		if j > i:
			ax[i, j].set_visible(False)

plt.show()

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, n_jobs=1, n_neighbors=1, p=2, weights='uniform')

X_new = np.array([[5, 2.9, 1, 0.2]])

prediction = knn.predict(X_new)
result = iris['target_names'][prediction]

print('The first test prediction is for a setosa. The machine predicts ' + result[0])

y_pred = knn.predict(X_test)
percentage_correct = knn.score(X_test, y_test) * 100

print('The machine is predicting with {}%  accuracy'.format(percentage_correct))
