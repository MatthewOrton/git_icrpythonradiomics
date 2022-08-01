import sys
sys.path.append('/Users/morton/Documents/GitHub/icrpythonradiomics')

import numpy as np
from dataLoader import dataLoader
from makeThumbnails import makeThumbnails
from getSeriesUIDFolderDict import getSeriesUIDFolderDict

rtsFile = '/Users/morton/Dicom Files/BRC_GCT/XNAT/assessors/BRC_GCT_RMH_002__II__20110608_132500_BrightSpeed__II__2__II__AIM_20220307_085805.dcm'

seriesFolderDict = getSeriesUIDFolderDict('/Users/morton/Dicom Files/BRC_GCT/XNAT/referencedScans')

data = dataLoader(rtsFile, seriesFolderDict, verbose=True, roiShift={'row':-1, 'col':-1}) # roiShift needed to account for bug in XNAT_COLLABORATIONS viewer

data_Lesion1 = data.getNamedROI('Lesion1')
data_Lesion1_hole = data.getNamedROI('Lesion1_hole')
data_Lesion2 = data.getNamedROI('Lesion2')

data_Lesion1['mask']['array'] = np.logical_and(data_Lesion1['mask']['array'], np.logical_not(data_Lesion1_hole['mask']['array']))

makeThumbnails([data_Lesion1, data_Lesion2])

