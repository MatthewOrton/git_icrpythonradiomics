import sys
sys.path.append('/Users/morton/Documents/GitHub/icrpythonradiomics')
from loadDicomSeries import loadDicomSeries, getNamedROI, readRTSdata
import pydicom
import numpy as np
from skimage import draw
import matplotlib.pyplot as plt
from dataLoader import dataLoader



rtsFile = '/Users/morton/Dicom Files/BRC_GCT/XNAT/assessors/BRC_GCT_RMH_002__II__20110608_132500_BrightSpeed__II__2__II__AIM_20220307_085805.dcm'

seriesFolderDict = {'1.2.752.24.3.327953459.6497310.59202072':'/Users/morton/Dicom Files/BRC_GCT/XNAT/referencedScans/BRC_GCT_RMH_002__II__20110608_132500_BrightSpeed'}

data = dataLoader(rtsFile, seriesFolderDict)

image, mask_Lesion1 = data.getNamedROI('Lesion1')
_, mask_Lesion1_hole = data.getNamedROI('Lesion1_hole')
_, mask_Lesion2 = data.getNamedROI('Lesion2')

mask_Lesion1 = np.logical_and(mask_Lesion1, np.logical_not(mask_Lesion1_hole))


plt.imshow(image[300,:,:], vmin=-100, vmax=200)
plt.show()

plt.imshow(mask_Lesion1[300,:,:])
plt.show()