import numpy as np
from sklearn.datasets import load_iris, make_moons
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_validate, StratifiedKFold, RepeatedStratifiedKFold, cross_val_predict
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
import sys


# Plot all ROC curves from cross-validation, and also the averaged curve.
# There are three ways to compute the averaged curve:
#   (i)   interpolating fpr and tpr as a function of the threshold and averaging both curves over the cv repeats,
#   (ii)  interpolating tpr as a function of fpr and averaging the tpr over the cv repeats,
#   (iii) interpolating fpr as a function of tpr and averaging the fpr over the cv repeats.
# Method (i) seems most sensible to me since the ROC is an intrinsic function
# averaging is one of 'threshold', 'fpr', 'tpr'
def plot_roc_cv(X, y, cv, estimator, ax, smoothing=100, plot_individuals=True, titleStr='', staircase=True, linewidth=2, color=None):

    fpr_g = np.linspace(0, 1, smoothing)
    tpr_i = []

    # get roc curve for each cv fold
    for n, tt in enumerate(cv.split(X, y)):
        test_index = tt[1]
        clf = estimator["estimator"][n]
        if hasattr(clf, "decision_function"):
            y_pred = clf.decision_function(X[test_index])
        else:
            y_pred = clf.predict_proba(X[test_index])[:, 1]

        # for some classifiers y_pred will contain only a few distinct values (e.g. KNN)
        # in this case the roc_curve function generates a piecewise linear curve, rather than
        # the usual staircase.  This code adds a tiny bit of noise to the values to force roc_curve
        # to plot a staircase
        if len(set(y_pred)) < len(y_pred):
            # delta is smallest difference between distinct values
            delta = np.min(np.diff(np.sort(np.array(list(set(y_pred))))))
            # add uniform noise that is 1000x smaller than delta to force distinct values
            y_pred = y_pred + 0.001*delta*np.random.uniform(size=y_pred.shape)

        fp, tp, th = roc_curve(y[test_index], y_pred, drop_intermediate=False)
        # interpolate tpr as a function of fpr
        tpr_i.append(np.interp(fpr_g, fp, tp))
        if plot_individuals:
            if staircase:
                ax.plot(fp, tp, color='silver')
            else:
                # draw lines from the middle of the treads and risers of the original staircase
                fpd = (0.5*(fp[0:-1] + fp[1:]))[0::2]
                fpd = np.append(np.insert(fpd, 0, 0), 1)
                tpd = (0.5*(tp[0:-1] + tp[1:]))[0::2]
                tpd = np.append(np.insert(tpd, 0, 0), 1)
                #ax.plot(fpd, tpd, color='black', linewidth=10, alpha=0.005)
                #ax.plot(fpd, tpd, color='black', linewidth=2, alpha=0.3)
                ax.plot(fpd, tpd, color='silver')

    fpr_g = np.append(np.insert(fpr_g, 0, 0), 1)
    tpr_g = np.mean(tpr_i, axis=0)
    tpr_g = np.append(np.insert(tpr_g, 0, 0), 1)
    ax.plot(fpr_g, tpr_g, label=titleStr, linewidth=linewidth, color=color)
    ax.set_title(titleStr)