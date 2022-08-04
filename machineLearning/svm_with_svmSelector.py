from sklearn.model_selection import GridSearchCV, cross_validate, RepeatedStratifiedKFold, cross_val_predict, KFold
from sklearn.linear_model import LogisticRegressionCV
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, make_scorer, roc_auc_score, roc_curve
from sklearn.datasets import make_classification
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification

X, y = make_classification(n_features=10, n_informative=2, n_clusters_per_class=2, random_state=0, class_sep=2)

pipe = Pipeline(steps=[('scaler', StandardScaler()), ('svm', LinearSVC(penalty='l1', dual=False, random_state=0, tol=1e-5))])

clf = GridSearchCV(estimator=pipe, param_grid={'svm__C':np.logspace(-3,2,20)}, cv=10, refit=True, verbose=0, scoring='roc_auc', n_jobs=-1)

clf.fit(X, y)