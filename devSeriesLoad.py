import sys, glob, traceback, os
sys.path.append('/Users/morton/Documents/GitHub/icrpythonradiomics')

import numpy as np
from dataLoader import dataLoader
from saveThumbnail import saveThumbnail
from getSeriesUIDFolderDict import getSeriesUIDFolderDict

# Some patients have multiple lesions stored in separate assessor files so get unique patient IDs
rtsFolder = '/Users/morton/Dicom Files/BRC_GCT/XNAT/assessors'
rtsFiles = glob.glob(os.path.join(rtsFolder, '*.dcm'))
patientIDs = list(set([os.path.split(x)[1].split('__II__')[0] for x in rtsFiles]))
patientIDs.sort()
patientIDs = [patientIDs[17]]

# dictionary to locate the images referenced by the rts files
seriesFolderDict = getSeriesUIDFolderDict('/Users/morton/Dicom Files/BRC_GCT/XNAT/referencedScans')

# TO DO:
# Warning if mask not on contiguous slices - display also?

# roiShift needed to account for bug in XNAT_COLLABORATIONS viewer
roiShift = {'row': -1, 'col': -1}

# some DICOM series have an extra coronal reformat image as part of the series that we will discard up to this max limit
maxNonCompatibleInstances = 1

patientIDsWithWarnings = []

for patientID in patientIDs:

    print('Processing ' + patientID)

    try:
        assessors = glob.glob(os.path.join(rtsFolder, patientID + '*.dcm'))

        # get list of Lesions and remove any holes from masks
        lesions = []

        for assessor in assessors:

            data = dataLoader(assessor, seriesFolderDict, roiShift=roiShift, maxNonCompatibleInstances=maxNonCompatibleInstances)

            for roiName in data.seriesData['ROINames']:
                # get ROIs that are not holes
                if 'hole' not in roiName:
                    roiInfo = [roiName]
                    thisLesion = data.getNamedROI(roiName, sliceSpacingUniformityThreshold=0.005)
                    # find any ROIs that are holes linked to the current ROI
                    for roiNameInner in data.seriesData['ROINames']:
                        if ('hole' in roiNameInner) and (roiName in roiNameInner):
                            roiInfo.append(roiNameInner)
                            thisHole = data.getNamedROI(roiNameInner)
                            thisLesion['mask']['array'] = np.logical_and(thisLesion['mask']['array'], np.logical_not(thisHole['mask']['array']))
                    lesions.append(thisLesion)
                    print(roiInfo)
        print(' ')

        # sort based on ROIName
        lesions = [x for _, x in sorted(zip([t['ROIName'] for t in lesions], lesions))]

        # format title string
        titleStr = os.path.split(assessors[0])[1].split('__II__')
        titleStr = r'$\bf{' + titleStr[0].replace('_', '\_') + '}$   ' + '  '.join(titleStr[1:2])
        titleStr += '\n' + '  '.join([os.path.split(x)[1].split('__II__')[3] for x in assessors])
        if any([not x['maskContiguous'] for x in lesions]):
            titleStr += '\n' + r'$\bf{WARNING}$: contains volumes with missing slices'
            patientIDsWithWarnings.append(patientID)
        else:
            titleStr += '\n '

        # save one pdf per patient
        saveThumbnail(lesions, assessors[0].replace('assessors', 'newThumbnails').replace('dcm','pdf'), titleStr=titleStr)

    except:
        print('\033[1;31;48m'+'_'*50)
        traceback.print_exc(file=sys.stdout)
        print('_'*50 + '\033[0;30;48m')

print('Patient IDs with warnings:')
print(patientIDsWithWarnings)
