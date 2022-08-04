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
pred = cross_val_predict(clf, X, y, cv=KFold(n_splits=n_splits), method='predict_proba')

# Process results of cross-validation to extract the dca curves for each data split
dca_cv = np.array([])
pt_cv = np.array([])
roc_cv_x = np.array([])
roc_cv_y = np.array([])
for k, v in cv.items():
    if 'DCA_' in k:
        ptHere = float(k.replace('test_DCA_', ''))
        pt_cv = np.append(pt_cv, ptHere*np.ones(len(v)))
        dca_cv = np.append(dca_cv, v)

    if 'ROC_x_' in k:
        roc_cv_x = np.append(roc_cv_x, v)

    if 'ROC_y_' in k:
        roc_cv_y = np.append(roc_cv_y, v)

roc_cv_x = np.reshape(roc_cv_x, (len(ptArr), n_splits*n_repeats))
roc_cv_y = np.reshape(roc_cv_y, (len(ptArr), n_splits*n_repeats))

# main plots
plt.plot(ptArr, netBenefit, label='Resub')
snsData = pd.DataFrame({'Probability threshold':pt_cv, 'Net benefit':dca_cv})
sns.lineplot(data=snsData, x='Probability threshold', y='Net benefit', ci='none', label='CV')
ylim = plt.gca().get_ylim()

# treat all and treat none
TP = np.sum(y==1)/nData
plt.plot(ptArr, TP - (1 - TP) * ptArr / (1 - ptArr), color='k', linestyle='-.', label='Treat all')
plt.plot(ptArr, 0*ptArr, color='k', linestyle=':', label='Treat none')

plt.ylim(ylim)
plt.legend(loc=3)
plt.title('AUROC = ' + str(np.round(np.mean(cv['test_roc_auc']),3)))

plt.show()
#plt.savefig('/Users/morton/Desktop/DCA_CV.pdf',  orientation='landscape', format='pdf')

fpr, tpr, _ = roc_curve(y, clf.predict_proba(X)[:,1])
plt.plot(fpr, tpr)
plt.plot(np.mean(roc_cv_x, axis=1), np.mean(roc_cv_y, axis=1))
plt.show()

from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of `x` and `y`

    Parameters
    ----------
    x, y : array_like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    Returns
    -------
    matplotlib.patches.Ellipse

    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

fig, ax = plt.subplots(figsize=(6, 6))
for n in range(len(ptArr)):
    confidence_ellipse(roc_cv_x[n,:], roc_cv_y[n,:], ax, n_std=0.5, alpha=0.2, facecolor='pink', edgecolor='none')
plt.plot(np.mean(roc_cv_x, axis=1), np.mean(roc_cv_y, axis=1))
plt.show()