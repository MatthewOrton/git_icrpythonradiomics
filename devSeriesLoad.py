import sys
sys.path.append('/Users/morton/Documents/GitHub/icrpythonradiomics')
from loadDicomSeries import loadDicomSeries, getNamedROI, readRTSdata
import pydicom
import numpy as np
from skimage import draw
import matplotlib.pyplot as plt





rts = pydicom.dcmread('/Users/morton/Dicom Files/BRC_GCT/XNAT/assessors/BRC_GCT_RMH_002__II__20110608_132500_BrightSpeed__II__2__II__AIM_20220307_085805.dcm')

rfors = rts.ReferencedFrameOfReferenceSequence[0]
rtrss = rfors.RTReferencedStudySequence[0]
ReferencedSeriesUID = rtrss.RTReferencedSeriesSequence[0].SeriesInstanceUID

folder = '/Users/morton/Dicom Files/BRC_GCT/XNAT/referencedScans/BRC_GCT_RMH_002__II__20110608_132500_BrightSpeed'

series = loadDicomSeries(ReferencedSeriesUID, folder, maxSpatiallyNonCompatibleInstances=1)

series = readRTSdata(rts, series)

image, mask_Lesion1 = getNamedROI(series, 'Lesion1')
_, mask_Lesion1_hole = getNamedROI(series, 'Lesion1_hole')

mask_Lesion1 = np.logical_and(mask_Lesion1, np.logical_not(mask_Lesion1_hole))


plt.imshow(image[300,:,:], vmin=-100, vmax=200)
plt.show()

plt.imshow(mask_Lesion1[300,:,:])
plt.show()