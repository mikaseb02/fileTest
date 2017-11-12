from __future__ import print_function

import logging
import numpy as np
from optparse import OptionParser
import sys
from time import time
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn import metrics

from nltk.corpus import stopwords
from nltk import wordpunct_tokenize
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

from langdetect import detect

mypath = "/home/user/Documents/startapp/"
filename = "appDescriptions.xlsx"
trainpagename = "Examples"
classifypagename = "Classify"

import nltk

engwords = set(nltk.corpus.words.words('en'))
from nltk.corpus import stopwords

stopWords = set(stopwords.words('english'))

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


# parse commandline arguments
def getOptionParser():
    op = OptionParser()
    op.add_option("--report",
                  action="store_true", dest="print_report",
                  help="Print a detailed classification report.")
    op.add_option("--chi2_select",
                  action="store", type="int", dest="select_chi2",
                  help="Select some number of features using a chi-squared test")
    op.add_option("--confusion_matrix",
                  action="store_true", dest="print_cm",
                  help="Print the confusion matrix.")
    op.add_option("--top10",
                  action="store_true", dest="print_top10",
                  help="Print ten most discriminative terms per class"
                       " for every classifier.")
    op.add_option("--all_categories",
                  action="store_true", dest="all_categories",
                  help="Whether to use all categories or not.")
    op.add_option("--use_hashing",
                  action="store_true",
                  help="Use a hashing vectorizer.")
    op.add_option("--n_features",
                  action="store", type=int, default=2 ** 16,
                  help="n_features when using the hashing vectorizer.")
    return op


def is_interactive():
    return not hasattr(sys.modules['__main__'], '__file__')


# work-around for Jupyter notebook and IPython console
# argv = [] if is_interactive() else sys.argv[1:]
def getOpts(op, argv):
    # argv = ["--report"]
    (opts, args) = op.parse_args(argv)
    # print(opts)
    # print(args)
    if len(args) > 0:
        op.error("this script takes no arguments.")
        sys.exit(1)

    # print(__doc__)
    # op.print_help()
    print()
    return opts


def getTokenizeCleanData(mypath, filename, pagename):
    data = pd.read_excel(mypath + filename, pagename)
    data['desc_tokens'] = data['description'].map(tokenizeForEnglishOnly)
    # data['probableLang'] = data['description'].map(detect)
    return data


def filterDataWithNoEngDesc(data):
    data_desc_token_len = data['desc_tokens'].map(len)  # .desc_tokens
    data_filtered = data[(data_desc_token_len > 0)]
    return data_filtered

def filterDatawithEngDesc(data):
    data_desc_token_len = data['desc_tokens'].map(len)  # .desc_tokens
    data_filtered = data[(data_desc_token_len == 0)]
    return data_filtered


def prepareTrainData():
    # preparing the data

    data_examples = filterDataWithNoEngDesc(getTokenizeCleanData(mypath, filename, trainpagename))
    y_examples = data_examples['segment']
    data_examples.data = data_examples['desc_tokens']
    data_examples_size_kb = size_kb(data_examples)

    print("%d documents - %0.3fKB (examples set)" % (
        len(data_examples.data), data_examples_size_kb))

    argv = ["--report"]
    op = getOptionParser()
    opts = getOpts(op, argv)
    print("Extracting features from the training data using a sparse vectorizer")
    t0 = time()
    if opts.use_hashing:
        vectorizer = HashingVectorizer(stop_words='english', alternate_sign=False,
                                       n_features=opts.n_features)
        X_examples = vectorizer.transform(data_examples.data)
    else:
        vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                     stop_words='english')
        X_examples = vectorizer.fit_transform(data_examples.data)
    duration = time() - t0
    print("done in %fs at %0.3fkB/s" % (duration, data_examples_size_kb / duration))
    print("n_samples: %d, n_features: %d" % X_examples.shape)
    print()

    # print(type(X_examples))
    # print(type(X_examples.todense()))
    return [X_examples, y_examples, vectorizer]


def tokenizeForEnglishOnly(sentence):
    filt_text = " ".join(w.lower() for w in nltk.wordpunct_tokenize(sentence) \
                         if (w.lower() in engwords and len(w) > 2 and w.lower() not in stopWords))
    return filt_text


def size_kb(docs):
    return sum(len(s.encode('utf-8')) for s in docs) / 1e3


from sklearn.model_selection import cross_val_score


# retransform X_examples in non sparse data
# then fit the PCA

# dimensions
def getDimensions(examples):
    n_samples, n_features = examples.shape
    n_precategories = len(np.unique(examples.segment))
    print("n_digits: %d, \t n_samples %d, \t n_features %d"
          % (n_precategories, n_samples, n_features))
    return n_samples, n_features, n_precategories


# K Means with PCA
def KmeansProc(X_examples, y_examples):
    from sklearn import metrics

    print(82 * '_')
    print('init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette')

    sample_size = 300
    labels = y_examples
    scaledData = scale(X_examples.todense())
    # scaledSparse = scaledData.tosparse()
    bench_k_means(KMeans(init='k-means++', n_clusters=15, n_init=15),
                  name="k-means++", data=scaledData, labels=labels, sample_size=sample_size)

    bench_k_means(KMeans(init='random', n_clusters=15, n_init=15),
                  name="random", data=scaledData, labels=labels, sample_size=sample_size)
    pca = PCA(n_components=15).fit(scaledData)
    bench_k_means(KMeans(init=pca.components_, n_clusters=15, n_init=1),
                  name="PCA-based",
                  data=scaledData,
                  labels=labels,
                  sample_size=sample_size)
    print(82 * '_')

    # #############################################################################
    # Visualize the results on PCA-reduced data
    myPCA = PCA(n_components=2).fit(X_examples.todense())
    reduced_data = myPCA.fit_transform(X_examples.todense())
    print("myPCA.explained_variance_")
    print(myPCA.explained_variance_)
    # scaledReducedData = scale(reduced_data)
    kmeans = KMeans(init='k-means++', n_clusters=15, n_init=15)
    kmeans.fit(reduced_data)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .05  # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - h, reduced_data[:, 0].max() + h
    y_min, y_max = reduced_data[:, 1].min() - h, reduced_data[:, 1].max() + h
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')

    plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
              'Centroids are marked with white cross')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()


def bench_k_means(estimator, name, data, labels, sample_size):
    t0 = time()
    estimator.fit(data)
    print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, (time() - t0), estimator.inertia_,
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels, estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean',
                                      sample_size=sample_size)))




    # in this case the seeding of the centers is deterministic, hence we run the


# kmeans algorithm only once with n_init=1

# # Benchmark classifiers
def benchmark(clf, X, y):
    print('_' * 80)
    print(clf)
    t0 = time()

    scores = cross_val_score(clf, X, y, cv=10)
    cross_val_time = time() - t0
    print("cross val time: %0.3fs" % cross_val_time)
    print("cross val average score: %0.3f" % scores.mean())

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))
    return scores.mean()


def compareClassifiers(X, y):
    results = {}
    for clf, name in (
            (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
            (Perceptron(n_iter=50), "Perceptron"),
            (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
            (KNeighborsClassifier(n_neighbors=10), "kNN"),
            (RandomForestClassifier(n_estimators=100), "Random forest")):
        print('=' * 80)
        print(name)
        results[name] = (clf, benchmark(clf, X, y))
    return results


# print(data_toclassify.head(3))


# trial to optimize the whole process of calibration at all levels
def testGridSearch(data, target):
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.linear_model import SGDClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline

    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier()),
    ])

    # uncommenting more parameters will give better exploring power but will
    # increase processing time in a combinatorial way
    parameters = {
        'vect__max_df': (0.5, 0.75, 1.0),
        # 'vect__max_features': (None, 5000, 10000, 50000),
        'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
        # 'tfidf__use_idf': (True, False),
        # 'tfidf__norm': ('l1', 'l2'),
        'clf__alpha': (0.00001, 0.000001),
        'clf__penalty': ('l2', 'elasticnet'),
        # 'clf__n_iter': (10, 50, 80),
    }

    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)
    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    print(parameters)
    t0 = time()
    grid_search.fit(data, target)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
    return best_parameters


def optimizePipeVectTfIdfSgdClassifier(X, y):
    bestParametersPipeline = testGridSearch(X, y)
    vectorizer = CountVectorizer(stop_words=stopWords,
                                 max_df=bestParametersPipeline('vect_max_df'),
                                 ngram_range=bestParametersPipeline('vect__ngram_range'))
    tfidftr = TfidfTransformer()
    clf = SGDClassifier(alpha=bestParametersPipeline('clf__alpha'),
                        penalty=bestParametersPipeline('clf__penalty')
                        )

    # let s say that we consider this as the best
    pipeline = Pipeline([
        ('vect', vectorizer()),
        ('tfidf', tfidftr),
        ('clf', clf),
    ])
    print(bestParametersPipeline)
    scores = cross_val_score(pipeline, X, y)
    perf = scores.mean
    print(perf)


# now to predict we will take

if __name__ == '__main__':
    # prepare the Data
    X, y,vectorizer = prepareTrainData()
    # visualize with K-means if we can create better clusters (labels)
    # KmeansProc(X, y)
    # compare Classifiers
    results = compareClassifiers(X, y)
    import operator

    (name, (bestClf, accuracy)) = max(results.iteritems(), key=operator.itemgetter(1))
    bestClf.fit(X.todense(), y)
    pipeline = Pipeline([('tfIdfVectorizer', vectorizer), ('clf', bestClf)])
    from sklearn.externals import joblib

    joblib.dump(pipeline, mypath + 'pipeLineClassifier' + '.pkl')
    print(name + "saved in " + mypath + 'pipeLineClassifier' + '.pkl')
    newClf = joblib.load(mypath + 'pipeLineClassifier' + '.pkl')
    print(newClf)
    # optimize Pipeline
    # optimizePipeVectTfIdfSgdClassifier
