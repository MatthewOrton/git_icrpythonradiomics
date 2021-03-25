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


def nestedCVclassification(X, y, estimators, *scoring, n_splits_inner=5, n_splits_outer=5, n_repeats=2):

    # default scoring functions
    if not scoring:
        scoring = {'roc_auc': 'roc_auc',
                   'accuracy': 'accuracy',
                   'MCC': make_scorer(matthews_corrcoef),
                   'precision': 'precision',
                   'recall': 'recall',
                   'f1':'f1'}

    print(' ')
    print('Performing nested cross-validation classification with:')
    print('    ' + str(n_splits_inner) + ' inner folds for parameter tuning')
    print('    ' + str(n_splits_outer) + ' outer folds repeated ' + str(n_repeats)+ ' times for performance evaluation')
    print(' ')
    print('Fitting classification models:')
    {print('    ' + x["name"]) for x in estimators}
    print(' ')

    # Compact way of doing nested cross-validation.
    inner_cv = StratifiedKFold(n_splits=n_splits_inner, shuffle=True, random_state=1234)
    outer_cv = RepeatedStratifiedKFold(n_splits=n_splits_outer, n_repeats=n_repeats, random_state=1234)

    for n, estimator in enumerate(estimators):
        clf = GridSearchCV(estimator=estimator["model"], param_grid=estimator["p_grid"], cv=inner_cv, refit=True, verbose=0)
        cv_result = cross_validate(clf, X=X, y=y, cv=outer_cv, scoring=scoring, return_estimator=True, verbose=0, n_jobs=-1)
        estimators[n]["result"] = cv_result

        # draw roc from outer cv
        fig, ax = plt.subplots()
        plot_roc_cv(X, y, outer_cv, cv_result, ax, plot_individuals=True, smoothing=500, titleStr=estimator["name"])
        ax.legend()
        plt.show()

        # tidy up nested_scores and key names
        cv_result.pop('fit_time', None)
        cv_result.pop('score_time', None)
        cv_result.pop('estimator', None)
        cv_result = {key.replace('test_',''): value for key, value in cv_result.items()}

        # output main result
        nested_scores_mean = {key: np.mean(value) for key, value in cv_result.items()}
        nested_scores_std = {key: np.std(value) for key, value in cv_result.items()}
        print(estimator["name"])
        print(nested_scores_mean)
        print(nested_scores_std)
        print(' ')

    plt.show()

    return estimators



# Plot all ROC curves from cross-validation, and also the averaged curve.
# There are three ways to compute the averaged curve:
#   (i)   interpolating fpr and tpr as a function of the threshold and averaging both curves over the cv repeats,
#   (ii)  interpolating tpr as a function of fpr and averaging the tpr over the cv repeats,
#   (iii) interpolating fpr as a function of tpr and averaging the fpr over the cv repeats.
# Method (i) seems most sensible to me since the ROC is an intrinsic function
# averaging is one of 'threshold', 'fpr', 'tpr'
def plot_roc_cv(X, y, cv, estimator, ax, smoothing=100, plot_individuals=True, titleStr=''):

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

        fp, tp, th = roc_curve(y[test_index], y_pred)
        # interpolate tpr as a function of fpr
        tpr_i.append(np.interp(fpr_g, fp, tp))
        if plot_individuals:
            #ax.plot(fp, tp, color='silver')
            # draw lines from the middle of the treads and risers of the original staircase
            fpd = (0.5*(fp[0:-1] + fp[1:]))[0::2]
            fpd = np.append(np.insert(fpd, 0, 0), 1)
            tpd = (0.5*(tp[0:-1] + tp[1:]))[0::2]
            tpd = np.append(np.insert(tpd, 0, 0), 1)
            #ax.plot(fpd, tpd, color='black', linewidth=10, alpha=0.005)
            ax.plot(fpd, tpd, color='black', linewidth=2, alpha=0.3)

    fpr_g = np.append(np.insert(fpr_g, 0, 0), 1)
    tpr_g = np.mean(tpr_i, axis=0)
    tpr_g = np.append(np.insert(tpr_g, 0, 0), 1)
    ax.plot(fpr_g, tpr_g, label=titleStr) #, color='green')
    ax.set_title(titleStr)