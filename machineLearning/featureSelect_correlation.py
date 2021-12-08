from scipy.stats import spearmanr
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection.base import SelectorMixin
from sklearn.utils.validation import check_is_fitted


class featureSelect_correlation(BaseEstimator, SelectorMixin):

    def __init__(self, threshold=0.75, exact=False):
        self.threshold = threshold
        self.exact = exact

    def fit(self, X, y=None):

        if self.threshold==1 or X.shape[1]==1:
            self.maxCorrelation_ = np.ones(X.shape[1])
            return self

        # make sure no zero-variance columns
        colInd = np.nonzero(np.var(X, axis=0)>0)[0]

        # compute correlation matrix
        xCorr = np.abs(spearmanr(X[:,colInd]).correlation)

        if self.exact:

            diagZeros = 1 - np.diag(np.ones(xCorr.shape[0]))
            xCorr = np.abs(xCorr) * diagZeros
            xCorrMax = np.max(xCorr)

            while (xCorrMax > self.threshold) and (xCorr.shape[0]>2):
                # find row column index of max correlated pair
                idx = np.nonzero(xCorr == xCorrMax)
                i0 = idx[0][0]
                i1 = idx[1][0]
                # get column for i0 and i1, and remove the diagonal terms and the terms for (i0, i1) or (i1, i0)
                x0 = np.delete(xCorr[:, i0], [i0, i1])
                x1 = np.delete(xCorr[:, i1], [i0, i1])
                if np.nanmean(x0) > np.nanmean(x1):
                    # remove feature i0
                    colInd = np.delete(colInd, i0)
                    xCorr = np.delete(xCorr, i0, axis=0)
                    xCorr = np.delete(xCorr, i0, axis=1)
                else:
                    # remove feature i1
                    colInd = np.delete(colInd, i1)
                    xCorr = np.delete(xCorr, i1, axis=0)
                    xCorr = np.delete(xCorr, i1, axis=1)
                xCorrMax = np.nanmax(xCorr)

            # at this stage there will be two or more features left.  If there are two features left then leave them both if their correlation
            # is low enough, but keep only one if not
            if (xCorr.shape[0]==2) and (xCorr[0,1]>self.threshold):
                # set to artifically keep the second feature over the first
                xCorr[0, 1] = 0
                xCorr[1, 0] = 1

        else: # faster approximate method that does not re-compute the average correlation rankings at each step

            xCorrMean = np.mean(np.abs(xCorr), axis=0)

            xCorr = np.triu(xCorr, 1)
            xCorr[np.tril_indices(xCorr.shape[0], 0)] = np.nan
            elToCheck = np.nonzero(np.abs(np.nan_to_num(xCorr)) > self.threshold)

            colsToDiscard = xCorrMean[elToCheck[1]] > xCorrMean[elToCheck[0]]
            rowsToDiscard = np.logical_not(colsToDiscard)

            deletecol = np.concatenate((elToCheck[1][colsToDiscard], elToCheck[0][rowsToDiscard]))
            deletecol = np.unique(deletecol)
            xCorr = np.delete(xCorr, deletecol, axis=0)
            xCorr = np.delete(xCorr, deletecol, axis=1)
            xCorr = np.triu(xCorr, 1)
            xCorr = xCorr + xCorr.T
            colInd = np.delete(colInd, deletecol)

        # get maximum correlation for each column
        # set this to 1 for any features that are already above self.threshold
        # so that they won't get selected by _get_support_mask()
        self.maxCorrelation_ = np.ones(X.shape[1])
        self.maxCorrelation_[colInd] = np.max(xCorr, axis=0)


        return self

    def _get_support_mask(self):
        check_is_fitted(self, 'maxCorrelation_')
        return self.maxCorrelation_ <= self.threshold
