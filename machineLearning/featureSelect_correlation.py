from scipy.stats import spearmanr
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection.base import SelectorMixin
from sklearn.utils.validation import check_is_fitted
import pandas as pd


class featureSelect_correlation(BaseEstimator, SelectorMixin):

    def __init__(self, threshold=0.75, exact=False, keepFirstColumn=False, namedColumnsKeep=[], featureGroupHierarchy=[]):
        self.threshold = threshold
        self.exact = exact
        self.keepFirstColumn = keepFirstColumn  # this input forces algorithm to keep the first column - useful if we know that a particular feature (e.g. tumour size) is likely to be informative and interpretable, so we want to keep it all the time and discard and features that are correlated with it
        self.featureGroupHierarchy = featureGroupHierarchy

        # make sure self.ignoreColumns is a list, even if it only has one element
        if isinstance(namedColumnsKeep, list):
            self.namedColumnsKeep = namedColumnsKeep
        else:
            self.namedColumnsKeep = [namedColumnsKeep]

    def fit(self, Xall, y=None):

        if self.namedColumnsKeep and not isinstance(Xall, pd.DataFrame):
            raise Exception('To keep named columns data input needs to be a DataFrame')

        if self.featureGroupHierarchy and not isinstance(Xall, pd.DataFrame):
            raise Exception('To apply hierarchical feature group selection the data input needs to be a DataFrame')

        # extract the features we need to apply feature reduction to into an array, and keep track of the features we will keep using keepMask
        keepMask = np.zeros(Xall.shape[1], dtype=bool)
        if isinstance(Xall, pd.DataFrame):
            if self.namedColumnsKeep:
                for namedColumn in self.namedColumnsKeep:
                    keepMask[list(Xall.columns).index(namedColumn)] = True
            X = Xall.loc[:, np.logical_not(keepMask)]
        else:
            X = Xall

        # feature selection should not remove any features
        if self.threshold==1 or X.shape[1]==1:
            self.mask_ = np.ones(Xall.shape[1], dtype=bool)
            return self

        if self.featureGroupHierarchy:
            xCorr = np.abs(spearmanr(np.array(X)).correlation)
            featuresAtRisk = np.ones(X.shape[1], dtype=bool)
            interGroupDiscardMask = np.zeros(X.shape[1], dtype=bool)
            intraGroupDiscardMask = np.zeros(X.shape[1], dtype=bool)
            for featureGroup in self.featureGroupHierarchy:
                thisFeatureGroup = np.logical_and(featuresAtRisk, X.columns.str.contains(featureGroup))
                intraGroupDiscardMask[thisFeatureGroup] = np.logical_not(self.getNonCorrelatedMask_(X.loc[:,thisFeatureGroup]))
                featuresAtRisk[thisFeatureGroup] = False
                xCorrMax = np.max(xCorr[np.logical_not(featuresAtRisk), :][:, featuresAtRisk], axis=0)
                interGroupDiscardMask[featuresAtRisk] = np.logical_or(interGroupDiscardMask[featuresAtRisk], xCorrMax >= self.threshold)
            # get intraGroupDisdardMask for remaining features
            intraGroupDiscardMask[featuresAtRisk] = np.logical_not(self.getNonCorrelatedMask_(X.loc[:, featuresAtRisk]))
            discardMask = np.logical_or(interGroupDiscardMask, intraGroupDiscardMask)
            keepMask[np.logical_not(keepMask)] = np.logical_not(discardMask)

            featuresAtRisk = np.ones(X.shape[1], dtype=bool)
            for featureGroup in self.featureGroupHierarchy:
                thisFeatureGroup = np.logical_and(featuresAtRisk, X.columns.str.contains(featureGroup))
                featuresAtRisk[thisFeatureGroup] = False
                print(featureGroup + ' : ' + str(np.sum(np.logical_not(discardMask[thisFeatureGroup]))) + '/' + str(np.sum(thisFeatureGroup)))
            print('remainder' + ' : ' + str(np.sum(np.logical_not(discardMask[featuresAtRisk]))) + '/' + str(np.sum(featuresAtRisk)))

        else:
            keepMask[np.logical_not(keepMask)] = self.getNonCorrelatedMask_(X)

        self.mask_ = keepMask

        return self



    def _get_support_mask(self):
        check_is_fitted(self, 'mask_')
        return self.mask_



    # copied from '/Users/morton/anaconda3/lib/python3.6/site-packages/sklearn/feature_selection/_base.py'
    # changed output to make sure DataFrame output if DataFrame input
    def fit_transform(self, X, y=None, **fit_params):
        """
        Fit to data, then transform it.

        Fits transformer to X and y with optional parameters fit_params
        and returns a transformed version of X.

        Parameters
        ----------
        X : {array-like, sparse matrix, dataframe} of shape \
                (n_samples, n_features)

        y : ndarray of shape (n_samples,), default=None
            Target values.

        **fit_params : dict
            Additional fit parameters.

        Returns
        -------
        X_new : ndarray array of shape (n_samples, n_features_new)
            Transformed array.
        """
        # non-optimized default implementation; override when a better
        # method is possible for a given clustering algorithm
        if isinstance(X, pd.DataFrame):
            if y is None:
                # fit method of arity 1 (unsupervised transformation)
                self.fit(X, **fit_params)
            else:
                # fit method of arity 2 (supervised transformation)
                self.fit(X, y, **fit_params)
            return X.loc[:,self.mask_]
        else:
            if y is None:
                # fit method of arity 1 (unsupervised transformation)
                return self.fit(X, **fit_params).transform(X)
            else:
                # fit method of arity 2 (supervised transformation)
                return self.fit(X, y, **fit_params).transform(X)

    def getNonCorrelatedMask_(self, X):

        if X.shape[1]==1:
            return np.ones(1, dtype=bool)

        X = np.array(X)

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

                if self.keepFirstColumn and (i0==0 or i1==0):
                    if i1==0:
                        # remove feature i0
                        colInd = np.delete(colInd, i0)
                        xCorr = np.delete(xCorr, i0, axis=0)
                        xCorr = np.delete(xCorr, i0, axis=1)
                    else:
                        # remove feature i1
                        colInd = np.delete(colInd, i1)
                        xCorr = np.delete(xCorr, i1, axis=0)
                        xCorr = np.delete(xCorr, i1, axis=1)
                else:
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

        # mask has same columns as X (which doesn't include the features we will keep), but output mask needs to relate to Xall (which has all features)
        mask = np.zeros(X.shape[1], dtype=bool)
        mask[colInd] = True
        return mask