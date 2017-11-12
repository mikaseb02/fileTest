print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

n_neighbors = 15

# import some data to play with
# iris = datasets.load_iris()

# we only take the first two features. We could avoid this ugly
# slicing by using a two-dim dataset
# X = iris.data[:, :2]
# y = iris.target

from start_app_exercise import prepareTrainData

X, y = prepareTrainData()

from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

reduced_data = PCA(n_components=20).fit_transform(X.todense())
print ("shape(reducedData)" + str(reduced_data.shape))
print ("shape(y)" + str(y.shape))
h = .02  # step size in the mesh

from start_app_exercise import getDimensions

n_samples, n_features = reduced_data.shape
n_precategories = len(np.unique(y))
print("n_digits: %d, \t n_samples %d, \t n_features %d"
      % (n_precategories, n_samples, n_features))

# Create color maps
# cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
# cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

from sklearn.model_selection import cross_val_score

# for weights in ['uniform', 'distance']:
#     # we create an instance of Neighbours Classifier and fit the data.
#     # clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
#     # clf.fit(reduced_data, y)
#     scores = cross_val_score(knn, reduced_data, y, cv=10, scoring='accuracy')
#     cv_scores.append(scores.mean())

from sklearn.neighbors import KNeighborsClassifier

# creating odd list of K for KNN
myList = list(range(1, 50))

# subsetting just the odd ones
neighbors = filter(lambda x: x % 2 != 0, myList)

# empty list that will hold cv scores
cv_scores = {}

# perform 10-fold cross validation
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=10, scoring="accuracy")
    cv_scores[k] = scores.mean()
    print("K:%i,%f", k, scores.mean())
import operator
optim = max(cv_scores.iteritems(), key=operator.itemgetter(1))
print ("optimal (k,accuracy) :" + str(optim))


#
#     # Plot the decision boundary. For that, we will assign a color to each
#     # point in the mesh [x_min, x_max]x[y_min, y_max].
#     x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
#     y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
#                          np.arange(y_min, y_max, h))
#     Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
#
#     # Put the result into a color plot
#     Z = Z.reshape(xx.shape)
#     plt.figure()
#     # plt.pcolormesh(xx, yy, Z , cmap=cmap_light)
#     plt.pcolormesh(xx, yy, Z)
#     # Plot also the training points
#     # plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
#     #             edgecolor='k', s=20)
#     plt.scatter(X[:, 0], X[:, 1], c=y,
#                 edgecolor='k', s=20)
#     plt.xlim(xx.min(), xx.max())
#     plt.ylim(yy.min(), yy.max())
#     plt.title("3-Class classification (k = %i, weights = '%s')"
#               % (n_neighbors, weights))
#
# plt.show()
