import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split, KFold, cross_validate, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegressionCV
from xgboost import XGBClassifier
from sklearn.utils import shuffle
from sklearn import metrics

h = .02  # step size in the mesh

#def computeMetrics(y_true, y_pred):
#    return {"accuracy_score":metrics.accuracy_score(y_true, y_pred),
#            }

classifiers = [
    {"model":KNeighborsClassifier(3), "name":"Nearest Neighbors"},
    {"model":SVC(kernel="linear", C=0.025), "name":"Linear SVM"},
    {"model":SVC(kernel='rbf', gamma="scale", C=1), "name":"RBF SVM"}]
    # {"model":GaussianProcessClassifier(1.0 * RBF(1.0)), "name":"Gaussian Process"},
    # {"model":LogisticRegressionCV(Cs=20), "name":"Logistic regression"},
    # {"model":KNeighborsClassifier(3), "name":"Nearest Neighbors"},
    # {"model":RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1), "name":"Random Forest"},
    # {"model":GaussianNB(), "name":"Naive Bayes"},
    # {"model":QuadraticDiscriminantAnalysis(), "name":"QDA"}] #,
    #(DecisionTreeClassifier(max_depth=5), "Decision Tree"),
    #(MLPClassifier(alpha=1, max_iter=1000), "Neural Net"),
    #(AdaBoostClassifier(), "AdaBoost"),
    #(XGBClassifier(use_label_encoder=False, eval_metric='logloss'), "XGBoost")]  # these are default parameters, but by explictly including them we avoid warning messages

scoring = [
    'accuracy',
    'precision',
    'recall',
    'f1',
    'roc_auc']

n_samples = 100

#np.random.seed(246845)

X, y = make_classification(n_samples=n_samples, n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1)
#rng = np.random.RandomState(2)
#X += 2 * rng.uniform(size=X.shape)
X += 2 * np.random.uniform(size=X.shape)
linearly_separable = (X, y)

datasets = [make_moons(n_samples=n_samples, noise=0.3), #, random_state=0),
            make_circles(n_samples=n_samples, noise=0.2, factor=0.5), #, random_state=1),
            linearly_separable
            ]

figure = plt.figure(figsize=(27, 9))
i = 1
# iterate over datasets
score_cv = []
for ds_cnt, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.4) #, random_state=42)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    if ds_cnt == 0:
        ax.set_title("Input data")
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
               edgecolors='k')
    # Plot the testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
               edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    # use same CV splits for all models
    rskf = RepeatedStratifiedKFold(n_splits=3, n_repeats=2)

    # iterate over classifiers
    scores = []
    for classifier in classifiers:
        clf = classifier["model"]
        name = classifier["name"]

        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)

        clf_cv = []
        score_cv = []
        for train_index, test_index in rskf.split(X, y):
            X_train_cv, X_test_cv = X[train_index], X[test_index]
            y_train_cv, y_test_cv = y[train_index], y[test_index]
            clf.fit(X_train_cv, y_train_cv)
            clf_cv.append(clf)
            score_cv.append(clf.score(X_test_cv, y_test_cv))
        scores.append({'name':name, 'score':score_cv})
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                   edgecolors='k')
        # Plot the testing points
        #ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
        #           edgecolors='k', alpha=1)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        if ds_cnt == 0:
            ax.set_title(name)
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score),
                size=15, horizontalalignment='right')
        cvL = np.percentile(score_cv,10)
        cvU = np.percentile(score_cv,90)
        ax.text(xx.min() + 2, yy.min() + .3, ('%.2f' % cvL) + (', %.2f' % cvU),
                size=15, horizontalalignment='right')
        i += 1

plt.tight_layout()
plt.show()
