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

# Plot all ROC curves from cross-validation, and also the averaged curve.
# There are three ways to compute the averaged curve:
#   (i)   interpolating fpr and tpr as a function of the threshold and averaging both curves over the cv repeats,
#   (ii)  interpolating tpr as a function of fpr and averaging the tpr over the cv repeats,
#   (iii) interpolating fpr as a function of tpr and averaging the fpr over the cv repeats.
# Method (i) seems most sensible to me since the ROC is an intrinsic function
# averaging is one of 'threshold', 'fpr', 'tpr'
def plot_roc_cv(X, y, cv, estimator, ax, averaging='threshold', smoothing=100, plot_individuals=True):

    fpr_g, tpr_g = np.linspace(0, 1, smoothing), np.linspace(0, 1, smoothing)
    fpr, tpr, thr, tpr_i, fpr_i = [], [], [], [], []

    # get roc curve for each cv fold
    for n, tt in enumerate(cv.split(X, y)):
        test_index = tt[1]

        if hasattr(clf, "decision_function"):
            y_pred = estimator["estimator"][n].decision_function(X[test_index])
        else:
            y_pred = estimator["estimator"][n].predict_proba(X[test_index])[:, 1]

        fp, tp, th = roc_curve(y[test_index], y_pred)
        if plot_individuals:
            ax.plot(fp, tp, color='silver')
            # draw lines from the middle of the treads and risers of the original staircase
            # fpd = (0.5*(fp[0:-1] + fp[1:]))[0::2]
            # fpd = np.append(np.insert(fpd, 0, 0), 1)
            # tpd = (0.5*(tp[0:-1] + tp[1:]))[0::2]
            # tpd = np.append(np.insert(tpd, 0, 0), 1)
            # ax.plot(fpd, tpd, color='black', linewidth=15, alpha=0.005)
            # ax.plot(fpd, tpd, color='black', linewidth=2, alpha=0.3)
        fpr.append(fp)
        tpr.append(tp)
        thr.append(th)
        # interpolate tpr as a function of fpr
        tpr_i.append(np.interp(fpr_g, fp, tp))
        # interpolate fpr as a function of tpr
        fpr_i.append(np.interp(tpr_g, tp, fp))

    # to interpolate on the threshold we need to know the limits of the threshold to
    # get the grid of values over which to interpolate
    th_min = np.min([np.min(x) for x in thr])
    th_max = np.max([np.max(x) for x in thr])
    th_g = np.linspace(th_min, th_max, smoothing)
    tpr_i2 = []
    fpr_i2 = []
    for fp, tp, th in zip(fpr, tpr, thr):
        ind = np.argsort(th)
        tpi = np.interp(th_g, th[ind], tp[ind])
        tpr_i2.append(tpi)
        fpi = np.interp(th_g, th[ind], fp[ind])
        fpr_i2.append(fpi)
        ax.plot(th_g, tpi, color='blue')
        ax.plot(th_g, fpi, color='green')

    # make it a one item list if it isn't already
    if isinstance(averaging, str):
        averaging = [averaging]

    if any(x == 'tpr' for x in averaging):
        tprm = np.mean(tpr_i, axis=0)
        ax.plot(fpr_g, tprm, color='green')
    if any(x == 'fpr' for x in averaging):
        fprm = np.mean(fpr_i, axis=0)
        ax.plot(fprm, tpr_g, color='blue')
    if any(x == 'threshold' for x in averaging):
        fprm = np.mean(fpr_i2, axis=0)
        tprm = np.mean(tpr_i2, axis=0)
        ax.plot(fprm, tprm, color='red')



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

X_data, y_data = make_moons(n_samples=413, noise=0.1)

cm_bright = ListedColormap(['#FF0000', '#0000FF'])
plt.scatter(X_data[:, 0], X_data[:, 1], c=y_data, edgecolors='k', cmap=cm_bright)
plt.show()

# Set up model and parameter grid

# estimator = SVC(kernel="rbf")
# p_grid = {"C": np.logspace(-1,3,8), "gamma": np.logspace(-3,1,8)}

estimator = GaussianProcessClassifier()
p_grid = {"kernel": [RBF(l) for l in np.logspace(-1, 1, 2)]}

# estimator = GaussianNB()
# p_grid = {}

# list of scoring functions to use
scoring = {'accuracy': 'accuracy',
           'MCC': make_scorer(matthews_corrcoef),
           'precision': 'precision',
           'recall': 'recall',
           'roc_auc': 'roc_auc',
           'f1':'f1'}

# Most compact way of doing nested cross-validation.
inner_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=1234)
outer_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=1234)
clf = GridSearchCV(estimator=estimator, param_grid=p_grid, cv=inner_cv, refit=True)
cv_result = cross_validate(clf, X=X_data, y=y_data, cv=outer_cv, scoring=scoring, return_estimator=True)

# draw roc from outer cv
fig, ax = plt.subplots()
plot_roc_cv(X_data, y_data, outer_cv, cv_result, ax, plot_individuals=True, smoothing=500) #, averaging=['threshold', 'fpr', 'tpr'])
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
print(nested_scores_mean)
print(nested_scores_std)
