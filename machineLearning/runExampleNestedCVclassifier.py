import numpy as np
from sklearn.datasets import load_iris, make_moons
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_validate, KFold, RepeatedKFold, StratifiedKFold, RepeatedStratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, make_scorer, matthews_corrcoef, roc_curve, plot_roc_curve
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from NestedCVclassifier import nestedCVclassification, plot_roc_cv

np.random.seed(2345)

X_data, y_data = make_moons(n_samples=153, noise=0.2)

cm_bright = ListedColormap(['#FF0000', '#0000FF'])
plt.scatter(X_data[:, 0], X_data[:, 1], c=y_data, edgecolors='k', cmap=cm_bright)
plt.show()

# Set up model and parameter grid

estimators = []
# estimators.append({"model": LogisticRegression(solver="liblinear"),                        "name": "Logistic",          "p_grid": {"C": np.logspace(-4,4,20), "penalty": ["l1", "l2"]},     "result": {}})
# estimators.append({"model": SVC(kernel="rbf"),                                             "name": "SVM-RBF",           "p_grid": {"C": np.logspace(-1,3,5), "gamma": np.logspace(-3,1,5)}, "result": {}})
# estimators.append({"model": SVC(kernel="linear"),                                          "name": "SVM-linear",        "p_grid": {"C": np.logspace(-1,3,15)},                              "result": {}})
# estimators.append({"model": GaussianProcessClassifier(),                                   "name": "GaussianProcess",   "p_grid": {"kernel": [RBF(l) for l in np.logspace(-1, 1, 10)]},     "result": {}})
estimators.append({"model": GaussianNB(),                                                  "name": "GaussianNB",        "p_grid": {},                                                       "result": {}})
estimators.append({"model": KNeighborsClassifier(),                                        "name": "KNN",               "p_grid": {"n_neighbors":[2, 3, 4, 5, 6, 7, 8]},                    "result": {}})
# estimators.append({"model": RandomForestClassifier(max_features=1),                        "name": "RF",                "p_grid": {"max_depth":[2, 3, 4, 5, 6, 7, 8]},                      "result": {}})
#estimators.append({"model": XGBClassifier(use_label_encoder=False, eval_metric='logloss'), "name": "XGB",               "p_grid": {},                                                       "result": {}})
# estimators.append({"model": QuadraticDiscriminantAnalysis(),                               "name": "QDA",               "p_grid": {},                                                       "result": {}})
# estimators.append({"model": PassiveAggressiveClassifier(),                                 "name": "PassiveAggressive", "p_grid": {},                                                       "result": {}})
# estimators.append({"model": AdaBoostClassifier(),                                          "name": "AdaBoost",          "p_grid": {},                                                       "result": {}})
# estimators.append({"model": LinearDiscriminantAnalysis(),                                  "name": "LDA",               "p_grid": {},                                                       "result": {}})

estimators = nestedCVclassification(X_data, y_data, estimators)
