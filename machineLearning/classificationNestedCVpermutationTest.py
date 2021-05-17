import numpy as np
from sklearn.model_selection import GridSearchCV, cross_validate, StratifiedKFold, RepeatedStratifiedKFold, permutation_test_score
from sklearn.metrics import roc_auc_score, confusion_matrix

def classificationNestedCVpermutationTest(X, y, estimator, param_grid, scoring, n_repeats=100, n_permutations=None, verbose=0, n_jobs=-1):

    print('Model                              = ' + str(estimator))

    # fit to all data using CV for tuning parameter optimisation
    inner_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5)
    clfAll = GridSearchCV(estimator=estimator,
                          param_grid=param_grid,
                          cv=inner_cv,
                          refit=True,
                          verbose=0,
                          scoring=scoring)
    clfAll.fit(X, y)
    if hasattr(clfAll, 'decision_function'):
        resubAUROC = roc_auc_score(y, clfAll.decision_function(X))
    else:
        resubAUROC = roc_auc_score(y, clfAll.predict_proba(X)[:, 1])
    print('AUCROC (resub)                     = ' + str(np.round(resubAUROC, 3)))

    inner_cv = StratifiedKFold(n_splits=5, shuffle=True)
    clf = GridSearchCV(estimator=estimator,
                          param_grid=param_grid,
                          cv=inner_cv,
                          refit=True,
                          verbose=0,
                          scoring=scoring)

    # Use method suggested in "Permutation Tests for Studying Classifier Performance" by Markus Ojala for dealing with repeated CV.

    outer_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats)
    cv_result = cross_validate(clf,
                               X=X,
                               y=y,
                               cv=outer_cv,
                               scoring='roc_auc',
                               return_estimator=True,
                               verbose=verbose,
                               n_jobs=n_jobs)

    # get scores for each repeat, averaged over the CV-folds
    scores = np.mean(np.reshape(cv_result['test_score'], (n_repeats, -1)), axis=1)
    print('AUCROC (CV)                        = \033[1m' + str(np.mean(scores).round(3)) + '\033[0m' + ' \u00B1 ' + str(np.std(scores).round(3)))

    # permutation test needs to use the same type of splitter as for outer_cv, but only needs to use one repeat
    outer_cv.n_repeats = 1
    if n_permutations is None:
        n_permutations = n_repeats
    _, perm_scores, _ = permutation_test_score(clf,
                                               X,
                                               y,
                                               scoring='roc_auc',
                                               cv=outer_cv,
                                               n_permutations=n_permutations,
                                               verbose=0,
                                               n_jobs=n_jobs)

    p_values = []
    for score in scores:
        p_values.append((np.count_nonzero(perm_scores >= score) + 1) / (n_permutations + 1))
    print('AUCROC (perm)                      = ' + str(np.mean(perm_scores).round(3)) + ' \u00B1 ' + str(np.std(perm_scores).round(3)))
    print('\033[4mp-value                            = \033[1m' + str(np.mean(p_values).round(4)) + '\033[0m\033[0m')  # + ' (' + str(np.quantile(p_values, 0.025).round(4)) + ', ' + str(np.quantile(p_values, 0.975).round(4)) + ')')
    print(' ')
    # print('coeff  ICC   AUROC featureName')
    # coef = np.squeeze(clfAll.best_estimator_.coef_)
    # I = np.argsort(np.abs(coef))[::-1]
    # coef = coef[I]
    # colNames = [colNames[n] for n in I]
    # X = X[:, I]
    # for n in np.nonzero(coef)[0]:
    #     rocauc = roc_auc_score(y, X[:, n])
    #     if rocauc < 0.5:
    #         rocauc = 1 - rocauc
    #     print(str('{:.3f}'.format(round(coef[n], 3))).rjust(6) + ' ' +
    #           str('{:.3f}'.format(round(iccValue[colNames[n]], 3))).ljust(5) + ' ' +
    #           str('{:.3f}'.format(round(rocauc, 3))).ljust(5) + ' ' +
    #           colNames[n])
    #
    # print(' ')