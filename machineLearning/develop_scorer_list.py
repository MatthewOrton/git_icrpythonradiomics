from sklearn.model_selection import GridSearchCV, cross_validate, RepeatedStratifiedKFold, cross_val_predict, KFold
from sklearn.linear_model import LogisticRegressionCV
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, make_scorer, roc_auc_score, roc_curve
import pandas as pd
import seaborn as sns
import copy

# Note that this version of the function has the threshold as a keyword (pt), and expects pt to be a single (non-array) value
def calculate_net_benefit_score(y_true, y_pred, pt=0):
    _, fp, _, tp = confusion_matrix(y_true, y_pred > pt).ravel()
    net_benefit = (tp - fp * (pt / (1 - pt))) / len(y_true)
    return net_benefit

# Note that this version of the function has the threshold as a keyword (pt), and expects pt to be a single (non-array) value
def calculate_roc_point(y_true, y_pred, pt=0, axis='x'):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred > pt).ravel()
    if axis=='x':
        # 1 - specificity
        return fp / (fp + tn)
    if axis=='y':
        # sensitivity
        return tp / (tp + fn)


np.random.seed(1234)

# Toy data
nData = 100
w = 0.15
X = np.vstack((np.random.multivariate_normal([0, 1, 2], [[1, 0, 0],[0, 1, 0],[0, 0, 1]], int(w*nData)), np.random.multivariate_normal([2, 0, 1], [[2, 0, 0],[0, 2, 0],[0, 0, 2]], int((1-w)*nData))))
y = np.hstack((np.ones(int(w*nData)), np.zeros(int((1-w)*nData))))

# Make dictionary of scorers, each of which will compute one point on the DCA curve.
# The dictionary key is used to keep track of the threshold value that was used.
scorers = {}
# Don't use 0 and 1 as endpoints as this causes numerical underflow.
ptArr = np.linspace(np.finfo(float).eps, 1-np.finfo(float).eps, 100)
scoreName = 'DCA_'
for pt in ptArr:
    scorers[scoreName + str(pt)] = make_scorer(calculate_net_benefit_score, pt = pt, needs_proba=True)

scoreName = 'ROC_x_'
for pt in ptArr:
    scorers[scoreName + str(pt)] = make_scorer(calculate_roc_point, pt = pt, axis='x', needs_proba=True)

scoreName = 'ROC_y_'
for pt in ptArr:
    scorers[scoreName + str(pt)] = make_scorer(calculate_roc_point, pt = pt, axis='y', needs_proba=True)

# add another scorer to show we can add others if we want to
scorers['roc_auc'] = make_scorer(roc_auc_score)

# Estimator is logistic regression with in-built CV tuning
clf = LogisticRegressionCV(Cs=10, cv=5, penalty='l1', scoring='neg_log_loss', solver='liblinear', max_iter=10000)

# Fit to whole data set
clf.fit(X, y)

# Apply all scorers to the fitted model
netBenefit = []
for key, scorer in scorers.items():
    if 'DCA_' in key:
        netBenefit.append(scorer(clf, X, y))

# Cross-validate the fitting using the dictionary of scorers
n_splits = 10
n_repeats = 10
cv_split = RepeatedStratifiedKFold(n_splits=n_splits,  n_repeats=n_repeats)
cv = cross_validate(clf, X, y, scoring=scorers, cv=cv_split, return_estimator=True)

# Process results of cross-validation to extract the dca curves for each data split
dca_cv = {float(k.replace('test_DCA_','')):np.mean(v) for k, v in cv.items() if 'DCA' in k}
roc_x_cv = {float(k.replace('test_ROC_x_','')):np.mean(v) for k, v in cv.items() if 'ROC_x' in k}
roc_y_cv = {float(k.replace('test_ROC_y_','')):np.mean(v) for k, v in cv.items() if 'ROC_y' in k}

# main plots
plt.plot(ptArr, netBenefit, label='Resub')
plt.plot(list(dca_cv.keys()), list(dca_cv.values()), label='CV')
ylim = plt.gca().get_ylim()

# treat all and treat none
TP = np.sum(y==1)/nData
plt.plot(ptArr, TP - (1 - TP) * ptArr / (1 - ptArr), color='k', linestyle='-.', label='Treat all')
plt.plot(ptArr, 0*ptArr, color='k', linestyle=':', label='Treat none')

plt.ylim(ylim)
plt.legend(loc=3)
plt.title('AUROC = ' + str(np.round(np.mean(cv['test_roc_auc']),3)))

plt.show()

# CV ROC plot
fpr, tpr, _ = roc_curve(y, clf.predict_proba(X)[:,1])
plt.plot(fpr, tpr)
plt.plot(list(roc_x_cv.values()), list(roc_y_cv.values()))
plt.show()
