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
from sklearn.utils import resample
import sys
from plot_roc_cv import plot_roc_cv

def nestedCVclassification(X, y, estimators, *scoring, n_splits_inner=5, n_splits_outer=5, n_repeats=2, verbose=0, staircase=True, linewidth=2, color=None, plot_individuals=False):

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

    # use all the processors unless we are in debug mode
    n_jobs = -1
    if getattr(sys, 'gettrace', None)():
        n_jobs = 1

    for n, estimator in enumerate(estimators):
        clf = GridSearchCV(estimator=estimator["model"], param_grid=estimator["p_grid"], cv=inner_cv, refit=True, verbose=0, scoring=estimator["scoring"])
        cv_result = cross_validate(clf, X=X, y=y, cv=outer_cv, scoring=scoring, return_estimator=True, verbose=verbose, n_jobs=n_jobs)
        estimators[n]["result"] = cv_result

        # fig0, ax0 = plt.subplots(3, 1)
        # for ax, res in zip(ax0.reshape(-1), cv_result["estimator"]):
        #     testScore = np.ma.getdata(res.cv_results_['mean_test_score']).astype('float')
        #     # ax.plot(np.linspace(-4,4,20), testScore, color='blue')
        #     # ax.plot(res.cv_results_['param_C'][res.cv_results_['param_penalty']=='l1'], testScore[res.cv_results_['param_penalty']=='l1'], color='blue')
        #     # ax.plot(res.cv_results_['param_C'][res.cv_results_['param_penalty']=='l2'], testScore[res.cv_results_['param_penalty']=='l2'], color='red')
        #     ax.plot(res.cv_results_['param_n_neighbors'], testScore)
        #     #ax.set_xscale("log")
        #     #ax.set_title(str(res.best_params_["C"]))
        # plt.show()

        # draw roc from outer cv
        fig, ax = plt.subplots()
        plot_roc_cv(X, y, outer_cv, cv_result, ax, plot_individuals=plot_individuals, smoothing=500, titleStr=estimator["name"], staircase=staircase, linewidth=linewidth, color=color)
        ax.text(0, 1, 'AUC = ' + str(np.mean(cv_result['test_roc_auc']).round(3)) + ' \u00B1 ' + str(np.std(cv_result['test_roc_auc']).round(3)))
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

    return estimators

