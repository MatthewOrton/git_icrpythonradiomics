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

        if hasattr(clf, "decision_function"):
            y_pred = estimator["estimator"][n].decision_function(X[test_index])
        else:
            y_pred = estimator["estimator"][n].predict_proba(X[test_index])[:, 1]

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
            ax.plot(fpd, tpd, color='black', linewidth=10, alpha=0.005)
            #ax.plot(fpd, tpd, color='black', linewidth=2, alpha=0.3)

    fpr_g = np.append(np.insert(fpr_g, 0, 0), 1)
    tpr_g = np.mean(tpr_i, axis=0)
    tpr_g = np.append(np.insert(tpr_g, 0, 0), 1)
    ax.plot(fpr_g, tpr_g, label=titleStr) #, color='green')
    #ax.set_title(titleStr)



np.random.seed(2345)

# # Load the dataset
# iris = load_iris()
# X_data = iris.data
# y_data = iris.target
# 
# # make it a binary problem
# idx = np.logical_or(y_data==1, y_data==2)
# X_data = X_data[idx,:]
# y_data = y_data[idx]-1

X_data, y_data = make_moons(n_samples=153, noise=0.2)

cm_bright = ListedColormap(['#FF0000', '#0000FF'])
plt.scatter(X_data[:, 0], X_data[:, 1], c=y_data, edgecolors='k', cmap=cm_bright)
plt.show()

# Set up model and parameter grid

estimators = []
estimators.append({"model": LogisticRegression(solver="liblinear"),                        "name": "Logistic",          "p_grid": {"C": np.logspace(-4,4,20), "penalty": ["l1", "l2"]},     "result": {}})
estimators.append({"model": SVC(kernel="rbf"),                                             "name": "SVM-RBF",           "p_grid": {"C": np.logspace(-1,3,5), "gamma": np.logspace(-3,1,5)}, "result": {}})
estimators.append({"model": SVC(kernel="linear"),                                          "name": "SVM-linear",        "p_grid": {"C": np.logspace(-1,3,15)},                              "result": {}})
estimators.append({"model": GaussianProcessClassifier(),                                   "name": "GaussianProcess",   "p_grid": {"kernel": [RBF(l) for l in np.logspace(-1, 1, 10)]},     "result": {}})
estimators.append({"model": GaussianNB(),                                                  "name": "GaussianNB",        "p_grid": {},                                                       "result": {}})
estimators.append({"model": KNeighborsClassifier(),                                        "name": "KNN",               "p_grid": {"n_neighbors":[2, 3, 4, 5, 6, 7, 8]},                    "result": {}})
estimators.append({"model": RandomForestClassifier(max_features=1),                        "name": "RF",                "p_grid": {"max_depth":[2, 3, 4, 5, 6, 7, 8]},                      "result": {}})
#estimators.append({"model": XGBClassifier(use_label_encoder=False, eval_metric='logloss'), "name": "XGB",               "p_grid": {},                                                       "result": {}})
estimators.append({"model": QuadraticDiscriminantAnalysis(),                               "name": "QDA",               "p_grid": {},                                                       "result": {}})
# estimators.append({"model": PassiveAggressiveClassifier(),                                 "name": "PassiveAggressive", "p_grid": {},                                                       "result": {}})
# estimators.append({"model": AdaBoostClassifier(),                                          "name": "AdaBoost",          "p_grid": {},                                                       "result": {}})
# estimators.append({"model": LinearDiscriminantAnalysis(),                                  "name": "LDA",               "p_grid": {},                                                       "result": {}})

# list of scoring functions to use
scoring = {'roc_auc': 'roc_auc',
           'accuracy': 'accuracy',
           'MCC': make_scorer(matthews_corrcoef),
           'precision': 'precision',
           'recall': 'recall',
           'f1':'f1'}

# Compact way of doing nested cross-validation.
inner_cv = KFold(n_splits=5, shuffle=True, random_state=1234) #StratifiedKFold
outer_cv = RepeatedKFold(n_splits=5, n_repeats=100, random_state=1234) #RepeatedStratifiedKFold

for n, estimator in enumerate(estimators):
    clf = GridSearchCV(estimator=estimator["model"], param_grid=estimator["p_grid"], cv=inner_cv, refit=True, verbose=0)
    cv_result = cross_validate(clf, X=X_data, y=y_data, cv=outer_cv, scoring=scoring, return_estimator=True, verbose=0, n_jobs=-1)
    estimators[n]["result"] = cv_result

    # draw roc from outer cv
    fig, ax = plt.subplots()
    plot_roc_cv(X_data, y_data, outer_cv, cv_result, ax, plot_individuals=True, smoothing=500, titleStr=estimator["name"])
    ax.legend()
    plt.show()

    # Use of return_estimator=True in call to cross_validate() means cv_result includes a list containing all the estimators
    # from the outer CV loop.  Unfortunately, these don't come packaged with their corresponding data (i.e. the test and
    # training split for that fold).  However, the outer_cv object can be re-used to apply the same split again, and we can
    # use the predict function on the test data to re-generate the predictions.  This is useful if we want to plot ROCs that
    # are averaged over the CV folds.

    # check that re-using the outer_cv object to regenerate the test and training data does indeed generate the same result
    if False:
        predictions_Equal = []
        for n, tt in enumerate(outer_cv.split(X_data, y_data)):
            train_index = tt[0]
            test_index = tt[1]
            X_train, X_test = X_data[train_index], X_data[test_index]
            y_train, y_test = y_data[train_index], y_data[test_index]

            # use clf to fit the training data for this fold and predict the test data
            clf.fit(X_train, y_train)
            y_pred1 = clf.predict(X_test)

            # use the corresponding fitted estimator from the call to cross_validate() to predict the same test data
            y_pred2 = cv_result["estimator"][n].predict(X_test)

            # check the results of the two approaches are the same
            predictions_Equal.append(np.all(y_pred1==y_pred2))


        print('Are predictions identical for folds? '+str(np.all(predictions_Equal)))

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

#ax.legend()
#plt.show()
