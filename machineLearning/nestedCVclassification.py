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
import copy
from sklearn.pipeline import Pipeline
import random

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

    # seed these manually if you want the same outcome for repeated calls to this function
    innerSeed = 12345 #random.randint(1,100000)
    outerSeed = 12345 #random.randint(1, 100000)

    # Compact way of doing nested cross-validation.
    inner_cv = StratifiedKFold(n_splits=n_splits_inner, shuffle=True, random_state=innerSeed)

    # need to use same seed so that all uses of this splitter will result in the same split
    outer_cv = RepeatedStratifiedKFold(n_splits=n_splits_outer, n_repeats=n_repeats, random_state=outerSeed)

    # use all the processors unless we are in debug mode
    n_jobs = -1
    if getattr(sys, 'gettrace', None)():
        n_jobs = 1

    for n, estimator in enumerate(estimators):

        # get best tuning parameters from the whole data set
        clf_best_params = GridSearchCV(estimator=estimator["model"], param_grid=estimator["p_grid"], cv=inner_cv, refit=True, verbose=0, scoring=estimator["scoring"])
        clf_best_params.fit(X, y)
        # make a model that uses the best tuning parameters  ...
        model_best_params = copy.deepcopy(estimator["model"])
        if isinstance(model_best_params, Pipeline):
            for key in estimator["p_grid"].keys():
                pp = key.split('__')
                setattr(model_best_params.named_steps[pp[0]], pp[1], clf_best_params.best_params_[key])
        else:
            for key in estimator["p_grid"].keys():
                setattr(model_best_params, key, clf_best_params.best_params_[key])
        # ... and cross validate this model
        # The purpose of this cross-validated fit is to see how much impact the parameter tuning is having on the performance, compared with the (correct) use
        # of cross-validation for performance evaluation.  This fit does not use the nested cv loop for parameter tuning, so the variability caused by the different
        # tuned parameters on the full nested cross-validation will not be included in this fit.
        cv_result_best_params = cross_validate(model_best_params, X=X, y=y, cv=outer_cv, scoring=scoring, return_estimator=True, verbose=verbose, n_jobs=n_jobs)

        # nested cross validation - the correct way to do things
        clf = GridSearchCV(estimator=estimator["model"], param_grid=estimator["p_grid"], cv=inner_cv, refit=True, verbose=0, scoring=estimator["scoring"])
        cv_result = cross_validate(clf, X=X, y=y, cv=outer_cv, scoring=scoring, return_estimator=True, verbose=verbose, n_jobs=n_jobs)
        estimators[n]["result"] = cv_result

        print(' ')
        print(' ')
        print('Nested CV tuning parameters')
        cv_result["best_params"] = {}
        for key in cv_result["estimator"][0].best_params_.keys():
            cv_result["best_params"][key] = []
        for res in cv_result["estimator"]:
            for key, value in res.best_params_.items():
                cv_result["best_params"][key].append(value)
        for key, value in cv_result["best_params"].items():
            print(' ')
            print(key+ ':')
            print('   Median = ' + str(np.median(value)))
            print('   Min = ' + str(np.min(value)))
            print('   Max = ' + str(np.max(value)))
        print(' ')


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
        # fig0, ax0 = plt.subplots(3, 2)
        # for ax, res in zip(ax0.reshape(-1), cv_result["estimator"]):
        #     testScore = np.ma.getdata(res.cv_results_['mean_test_score']).astype('float')
        #     # ax.plot(np.linspace(-4,4,20), testScore, color='blue')
        #     # ax.plot(res.cv_results_['param_C'][res.cv_results_['param_penalty']=='l1'], testScore[res.cv_results_['param_penalty']=='l1'], color='blue')
        #     # ax.plot(res.cv_results_['param_C'][res.cv_results_['param_penalty']=='l2'], testScore[res.cv_results_['param_penalty']=='l2'], color='red')
        #     ax.plot(np.array(res.cv_results_['param_s__n_features_to_select']), testScore)
        #     # ax.set_xscale("log")
        #     # ax.set_title(str(res.best_params_["C"]))
        # plt.show()

        # draw roc from outer cv
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        plot_roc_cv(X, y, outer_cv, cv_result, ax1, plot_individuals=plot_individuals, smoothing=500, titleStr=estimator["name"]+' nested CV', staircase=staircase, linewidth=linewidth, color=color)
        ax1.text(0, 1, 'AUC = ' + str(np.mean(cv_result['test_roc_auc']).round(3)) + ' \u00B1 ' + str(np.std(cv_result['test_roc_auc']).round(3)))
        plot_roc_cv(X, y, outer_cv, cv_result_best_params, ax2, plot_individuals=plot_individuals, smoothing=500, titleStr=estimator["name"] + ' CV with best tuning param', staircase=staircase, linewidth=linewidth, color=color)
        ax2.text(0, 1, 'AUC = ' + str(np.mean(cv_result_best_params['test_roc_auc']).round(3)) + ' \u00B1 ' + str(np.std(cv_result['test_roc_auc']).round(3)))
        plt.show()

        # tidy up nested_scores and key names
        cv_result.pop('fit_time', None)
        cv_result.pop('score_time', None)
        cv_result.pop('estimator', None)
        cv_result.pop('best_params', None)
        cv_result = {key.replace('test_',''): value for key, value in cv_result.items()}

        # output main result
        nested_scores_mean = {key: np.mean(value) for key, value in cv_result.items()}
        nested_scores_std = {key: np.std(value) for key, value in cv_result.items()}
        print(' ')
        print(estimator["name"])
        print(nested_scores_mean)
        print(nested_scores_std)
        print(' ')
        print('Resubstitution performance using parameters tuned on whole data set:')
        print(clf_best_params.scoring + ' = ' + str(clf_best_params.score(X, y)))
        print(' ')
        print ('Best parameters tuned using whole data set:')
        print(clf_best_params.best_params_)

        # tidy up nested_scores and key names
        cv_result_best_params.pop('fit_time', None)
        cv_result_best_params.pop('score_time', None)
        cv_result_best_params.pop('estimator', None)
        cv_result_best_params = {key.replace('test_',''): value for key, value in cv_result_best_params.items()}

        print(' ')
        print('Cross-validated performance using the best tuning parameters based on GridSearchCV of whole data set:')
        best_params_scores_mean = {key: np.mean(value) for key, value in cv_result_best_params.items()}
        best_params_scores_std = {key: np.std(value) for key, value in cv_result_best_params.items()}
        print(best_params_scores_mean)
        print(best_params_scores_std)

    return estimators

