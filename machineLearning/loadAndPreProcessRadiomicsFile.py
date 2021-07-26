import sys
sys.path.append('/Users/morton/Documents/GitHub/icrpythonradiomics/machineLearning')

import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from pyirr import intraclass_correlation
from featureSelect_correlation import featureSelect_correlation
from scipy.stats import spearmanr, skew
from re import search


def loadAndPreProcessRadiomicsFile(fileName, index_col=None, featureRemoveStr='source|diagnostics', featureSelectStr='original', correlation_threshold=0.9, iccThreshold = None, followupStr='_followup', reproducibilityStr='_repro', logTransform=False):

    print('File                               = ' + fileName)
    print('Feature filter string              = ' + featureSelectStr)

    # read data from file
    df = pd.read_csv(fileName, index_col=index_col)

    # remove unwanted columns
    df = df.loc[:, ~df.columns.str.contains(featureRemoveStr)]

    # Select chosen features (can be a regex, e.g. 'original|wavelet'
    include = featureSelectStr
    df = df.loc[:, df.columns.str.contains(include)]
    print('Initial number of features         = ' + str(df.shape[1]))

    # Most of the shape features are (non-linear) functions of MeshVolume and SurfaceArea.
    # These extra features are useful for univariate feature discovery, but not for multivariate modelling as they are degenerate, and in a known way.
    # MeshVolume and SurfaceArea are typically highly correlated, and so later correlation-based feature reductions would remove one of them, which is undesirable.
    # Sphericity is derived from MeshVolume and SurfaceArea, is dimensionless, is interpretable, and tends to be uncorrelated with MeshVolume.
    # This suggests keeping MeshVolume and Sphericity.
    exclude = 'SurfaceArea|VoxelVolume|SurfaceVolumeRatio|Compactness1|Compactness2|SphericalDisproportion'
    df = df.loc[:, ~df.columns.str.contains(exclude)]

    # A similar argument applies to Major/Least/MinorAxisLength, Elongation and Flatness.
    # In this case Elongation and Flatness are dimensionless so we discard MinorAxisLength and LeastAxisLength to remove the redundancy.
    # Elongation and Flatness are often correlated with Sphericity, and the correlation feature reduction below will always remove Elongation and Flatness is this is the case.
    exclude = 'MinorAxisLength|LeastAxisLength'
    df = df.loc[:, ~df.columns.str.contains(exclude)]

    # Use log transform on any features that contain all positive (or all negative) values, and where the skewness is lower on taking logs.
    # Add '_log' subscript to column names so we can see which have been transformed.
    if logTransform:
        logTransformedCount = 0
        columnNames = df.columns
        for column in columnNames:
            # don't do log-transform on certain features or feature types
            if search('Skew|Kurtosis', column) or is_string_dtype(df[column]) or ('source' in column) or ('diagnostic' in column):
                continue
            # Some columns may have NaNs etc, so leave these as they are, and just look at the finite values since
            # we don't know here if we are going to use row/column deletion or data imputation to handle NaNs.
            thisColumnFinite = np.isfinite(df[column])
            thisColumn = df[column][thisColumnFinite]
            sign = np.sign(thisColumn[0])
            if np.all(np.sign(thisColumn) == sign):
                logThisColumn = sign * np.log(sign * thisColumn)
                # include sign so that all negative data (should it exist) will have the same ordering sense after taking logs
                if np.abs(skew(logThisColumn)) < np.abs(skew(thisColumn)):
                    df[column][thisColumnFinite] = logThisColumn
                    columnNames = [x + '_log' if x == column else x for x in columnNames]
                    logTransformedCount += 1
        df.columns = columnNames
        print('Number of log-transformed features = ' + str(logTransformedCount))

    # split off rows for any followup scans
    ifu = df.index.str.endswith(followupStr)
    if any(ifu):
        dfu = df.loc[ifu, :]
        dfu = dfu.rename(index=lambda x: x.replace(followupStr, ''))
        dfu.sort_index(inplace=True)
        # remove from main data frame
        df = df.loc[~ifu, :]
    else:
        dfu = []

    # split off rows for any reproducibility data
    ir = df.index.str.endswith(reproducibilityStr)
    if any(ir):
        dfr = df.loc[ir, :]
        dfr = dfr.rename(index=lambda x: x.replace(reproducibilityStr, ''))
        dfr.sort_index(inplace=True)
        # remove from main data frame
        df = df.loc[~ir, :]
        # get rows of df corresponding to dfr
        dfrr = df.loc[df.index.isin(dfr.index),:]
        dfrr.sort_index(inplace=True)

    print('No. of subjects                    = ' + str(df.shape[0]))
    if any(ir):
        print('No. of reproducibility subjects    = ' + str(dfr.shape[0]))
    if any(ifu):
        print('No. of follow-up subjects          = ' + str(dfu.shape[0]))

    # MeshVolume and Sphericity are the two principle shape features that are not removed in a previous step.
    # If any other features are correlated with these two features then remove the other features in order
    # to ensure MeshVolume and Sphericity remain as these are the easiest features to interpret.
    if any(df.columns.str.contains('original_shape_MeshVolume')) and any(df.columns.str.contains('original_shape_Sphericity')):
        corrMat = np.abs(spearmanr(np.array(df)).correlation)
        iMeshVolume = np.where(df.columns.str.contains('original_shape_MeshVolume'))[0]
        iSphericity = np.where(df.columns.str.contains('original_shape_Sphericity'))[0]
        ind = np.logical_and(corrMat[:, iMeshVolume] < correlation_threshold, corrMat[:, iSphericity] < correlation_threshold)
        ind[iMeshVolume] = True
        ind[iSphericity] = True
        df = df.loc[:, ind[:, 0]]

    # Check if MajorAxisLength and Maximum3DDiameter are correlated, and force MajorAxisLength to be removed.
    # Both or either feature may already have been removed if it is correlated with MeshVolume
    if any(df.columns.str.contains('original_shape_MajorAxisLength')) and any(df.columns.str.contains('original_shape_Maximum3DDiameter')):
        corrMat = np.abs(spearmanr(np.array(df)).correlation)
        iMajorAxisLength = np.where(df.columns.str.contains('original_shape_MajorAxisLength'))[0]
        iMaximum3DDiameter = np.where(df.columns.str.contains('original_shape_Maximum3DDiameter'))[0]
        if np.abs(corrMat[iMajorAxisLength, iMaximum3DDiameter])>correlation_threshold:
            df.drop(df.columns[iMajorAxisLength], axis=1, inplace=True)

    # Apply generic pair-wise correlation feature reduction.
    # Since we previously removed features correlated with MeshVolume and Sphericity, these two features
    # are guaranteed to survive this step.
    fsc = featureSelect_correlation(threshold=correlation_threshold, exact=True)
    idxCorrFeatures = ~df.columns.str.contains('source|diagnostic')
    fsc.fit(np.array(df.loc[:, idxCorrFeatures]))
    idxFeatures = df.columns.str.contains('source|diagnostic')
    idxFeatures[idxCorrFeatures] = fsc._get_support_mask()
    df = df.loc[:, idxFeatures]

    print('Correlation threshold              = ' + str(correlation_threshold))
    print('No. of feat. after corr selection  = ' + str(df.shape[1]))

    # compute reproducibility statistics and remove non-reproducible features
    icc = {}
    if 'dfr' in  locals():
        Iuse = np.ones(len(df.columns)).astype(bool)
        for n, col in enumerate(df.columns):
            data = np.stack((dfr[col], dfrr[col]), axis=1)
            icc[col] = intraclass_correlation(data, "twoway", "agreement").value
            if iccThreshold is not None:
                Iuse[n] = icc[col] > iccThreshold
        df = df.loc[:, Iuse]
        print('ICC threshold                      = ' + str(iccThreshold))
        print('No. of feat. after icc selection   = ' + str(df.shape[1]))

    # make columns of dfu match df
    if len(dfu)>0:
        dfu = dfu[df.columns]

    print('\033[4mNo. of features for modelling      = ' + str(df.shape[1]) + '\033[0m')

    # if (not np.all(np.isfinite(df))) or (not np.all(np.isfinite(dfu))):
    #     print(' ')
    #     print('\033[1;31mNon-valid input data, please investigate!\033[0;0m')
    #     print(' ')

    return df, dfu, icc
