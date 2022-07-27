import os, glob, pydicom
import numpy as np
from skimage import draw


def __getScaledSlice(dcm):

    # Extract pixel data and scale it, using default values if necessary.

    if 'RescaleSlope' in dcm:
        RescaleSlope = dcm.RescaleSlope
    else:
        RescaleSlope = 1.0
    if 'RescaleIntercept' in dcm:
        RescaleIntercept = dcm.RescaleIntercept
    else:
        RescaleIntercept = 0.0
    return RescaleSlope * dcm.pixel_array + RescaleIntercept

def __getListOfNonMatchingSopInstancesToDiscard(sopInstDict, tagsToMatch, maxSpatiallyNonCompatibleInstances):

    # Ideally all sopInstances in a series will have certain matching attributes, e.g. ImageOrientationPatient, PixelSpacing etc.
    #
    # Sometimes there are a small (usually 1) number that don't, e.g. one image that is a coronal reformat that shows position of axial slices.
    #
    # If the number of non-matching images is below maxSpatiallyNonCompatibleInstances then get output a list of the sopInstances to discard.

    # Generate a hash of tagsToMatch for each sopInstance and find the unique values.
    hashMeta = {key: hash(np.hstack([value[tag] for tag in tagsToMatch]).data.tobytes()) for key, value in sopInstDict.items()}
    hashMetaUnique = list(set([v for _, v in hashMeta.items()]))

    # If too many non-matching sopInstances then return - calling function will handle displaying the error
    if maxSpatiallyNonCompatibleInstances==0 and len(hashMetaUnique)>1:
        return None

    # If more than one unique hash with count above maxSpatiallyNonCompatibleInstances then return - calling function will handle error
    hashMetaKeep = [x for x in hashMetaUnique if list(hashMeta.values()).count(x) > maxSpatiallyNonCompatibleInstances]
    if len(hashMetaKeep)>1:
        return None

    # Get hash values that have fewer than maxSpatiallyNonCompatibleInstances instances and can therefore be discarded
    hashMetaUniqueDiscard = [x for x in hashMetaUnique if list(hashMeta.values()).count(x) <= maxSpatiallyNonCompatibleInstances]

    # Get SopInstances to discard
    sopInstDiscard = []
    for key, value in hashMeta.items():
        if value in hashMetaUniqueDiscard:
            sopInstDiscard.append(key)

    return sopInstDiscard

def __processContour(contourData, sopInstance, series, roiShift={'row':0, 'col':0}):

    # Process contour data to get contour in pixel coordinates and as a mask.
    #
    # Also get the contour area (in mm^2) so we can detect and remove contours that are too small, in particular
    # we occasionally get contours that only consist of a single point.

    coords = np.array([float(x) for x in contourData])
    polygonPatient = coords.reshape((int(len(coords) / 3), 3))

    # https://en.wikipedia.org/wiki/Shoelace_formula
    # this formula should work for planar polygon with arbitrary orientation
    crossSum = np.sum(np.cross(polygonPatient[0:-2, :], polygonPatient[1:-1, :]), axis=0)
    crossSum += np.cross(polygonPatient[-1, :], polygonPatient[0, :])
    contourArea = 0.5 * np.linalg.norm(crossSum)

    # Transform contour to pixel coordinates
    origin = np.reshape(sopInstance['ImagePositionPatient'], (1,3))
    spacing = series['PixelSpacing']
    colNorm = np.reshape(series['ImageOrientationPatient'][0:3], (3,1))
    rowNorm = np.reshape(series['ImageOrientationPatient'][3:6], (3,1))
    colPixCoord = np.dot(polygonPatient - origin, colNorm) / spacing[0]
    rowPixCoord = np.dot(polygonPatient - origin, rowNorm) / spacing[1]

    # according to https://scikit-image.org/docs/stable/api/skimage.draw.html?highlight=skimage%20draw#module-skimage.draw
    # there is a function polygon2mask, but this doesn't seem to be actually present in the library I have.
    # Since draw.polygon2mask is just a wrapper for draw.polygon I'm using the simpler function directly here.
    mask = np.zeros((series['Rows'], series['Columns'])).astype(bool)
    fill_row_coords, fill_col_coords = draw.polygon(rowPixCoord + roiShift['row'], colPixCoord + roiShift['col'], (series['Columns'], series['Rows']))
    mask[fill_row_coords, fill_col_coords] = True

    return colPixCoord, rowPixCoord, contourArea, mask

def getNamedROI(series, ROIName):

    if ROIName not in series['ROINames']:
        print(ROIName + ' not found!')
        return None

    # Find which AcquisitionNumber the named ROI is in
    AcquisitionList = []
    for k, v in series['SOPInstanceDict'].items():
        if len(v['ContourList'])>0:
            for contour in v['ContourList']:
                if contour['ROIName'] == ROIName:
                    AcquisitionList.append(v['AcquisitionNumber'])
    AcquisitionNumber = list(set(AcquisitionList))
    if len(AcquisitionNumber)>1:
        print(ROIName + ' spans more than one Acquisition!')
        return None
    AcquisitionNumber = AcquisitionNumber[0]

    # Get SopInstances from this Acquisition and sort on InstanceNumber
    SopInstanceList = [x for x in series['SOPInstanceDict'].values() if x['AcquisitionNumber'] == AcquisitionNumber]
    # get list of InstanceNumber values and check it is consistent
    InstanceNumberList = [x['InstanceNumber'] for x in SopInstanceList]
    if len(InstanceNumberList) != len(set(InstanceNumberList)) or min(InstanceNumberList) != 1 or max(InstanceNumberList) != len(set(InstanceNumberList)):
        print('InstanceNumber values are not consequtive in for AcquisitionNumber ' + str(AcquisitionNumber))
        return None
    SopInstanceList = [x for _, x in sorted(zip(InstanceNumberList, SopInstanceList))]

    # put data into image and mask arrays
    image = np.zeros((len(SopInstanceList), series['Columns'], series['Rows']))
    mask = np.zeros((len(SopInstanceList), series['Columns'], series['Rows'])).astype(bool)
    for n, sopInst in enumerate(SopInstanceList):
        image[n,:,:] = sopInst['PixelData']
        for contour in sopInst['ContourList']:
            if contour['ROIName'] == ROIName:
                mask[n,:,:] = np.logical_or(mask[n,:,:], contour['Mask'])


    return image, mask


def loadDicomSeries(seriesUID, folder, maxSpatiallyNonCompatibleInstances=0):

    # output variable to store image and metadata for this series
    series = {}
    series['SOPInstanceDict'] = {}

    sopInstanceCount = 0
    for file in glob.glob(os.path.join(folder, '**'), recursive=True):
        if not os.path.isdir(file) and pydicom.misc.is_dicom(file):

            dcm = pydicom.dcmread(file)

            if dcm.SeriesInstanceUID == seriesUID:
                sopInstanceCount += 1
            else:
                continue

            # keep one dcm for getting tags common to all instances in series
            if 'dcmKeep' not in locals():
                dcmKeep = pydicom.dcmread(file)

            thisSopInst = {}
            thisSopInst['PixelData'] = __getScaledSlice(dcm)
            thisSopInst['InstanceNumber'] = int(dcm.InstanceNumber)
            if hasattr(dcm, 'AcquisitionNumber'):
                thisSopInst['AcquisitionNumber'] = int(dcm.AcquisitionNumber)
            else:
                thisSopInst['AcquisitionNumber'] = 0
            if hasattr(dcm, 'SliceLocation'):
                thisSopInst['SliceLocation'] = float(dcm.SliceLocation)
            else:
                thisSopInst['SliceLocation'] = 0.0
            thisSopInst['ImagePositionPatient'] = np.array([float(x) for x in dcm.data_element('ImagePositionPatient')])
            thisSopInst['ImageOrientationPatient'] = np.array([float(x) for x in dcm.data_element('ImageOrientationPatient')])
            thisSopInst['PixelSpacing'] = np.array([float(x) for x in dcm.data_element('PixelSpacing')])
            thisSopInst['Rows'] = int(dcm.Rows)
            thisSopInst['Columns'] = int(dcm.Columns)
            if hasattr(dcm, 'SliceThickness'):
                thisSopInst['SliceThickness'] = float(dcm.SliceThickness)
            else:
                thisSopInst['SliceThickness'] = 0.0

            series['SOPInstanceDict'][dcm.SOPInstanceUID] = thisSopInst

    if sopInstanceCount==0:
        print('loadDicomSeries() error: SeriesInstanceUID ' + seriesUID + ' not found in folder ' + folder)
        return None

    # extract metadata that should be common to all instances in series
    if 'dcmKeep' in locals():
        series['SeriesInstanceUID'] = dcmKeep.SeriesInstanceUID
        series['StudyInstanceUID'] = dcmKeep.StudyInstanceUID
        series['PatientID'] = dcmKeep.PatientID
    else:
        print('loadDicomSeries() error - problem reading dicom files!')
        return None

    # get any Modality-specific parameters that might be useful
    sopClassUid_CTImageStorage = '1.2.840.10008.5.1.4.1.1.2'
    sopClassUid_MRImageStorage = '1.2.840.10008.5.1.4.1.1.4'
    if dcm.SOPClassUID == sopClassUid_CTImageStorage:
        parameterList = ['KVP',
                         'XRayTubeCurrent',
                         'ConvolutionKernel',
                         'ScanOptions',
                         'SliceThickness',
                         'SpacingBetweenSlices',
                         'ContrastBolusAgent',
                         'ContrastBolusVolume',
                         'PatientWeight']
        for parameter in parameterList:
            if hasattr(dcm, parameter):
                series[parameter] = getattr(dcm, parameter)
            else:
                series[parameter] = None

    # use listed tags to check if there are any non-matching sopInstances, and get list of these if there are fewer than maxSpatiallyNonCompatibleInstances
    tagsToMatch = ['Columns', 'Rows', 'ImageOrientationPatient', 'PixelSpacing', 'SliceThickness']
    sopInstDiscard = __getListOfNonMatchingSopInstancesToDiscard(series['SOPInstanceDict'], tagsToMatch, maxSpatiallyNonCompatibleInstances)

    # this is an error state that occurs if there are too many non-matching sopInstances
    if sopInstDiscard is None:
        print('\nloadDicomSeries() found images with more than one Orientation/Pixel spacing in series ' + seriesUID + ' ' + folder + '\n')
        return None

    # remove unwanted sopInstances
    for sop in sopInstDiscard:
        series['SOPInstanceDict'].pop(sop, None)

    # check that the collection of sopInstances is now compatible
    sopInstDiscard = __getListOfNonMatchingSopInstancesToDiscard(series['SOPInstanceDict'], tagsToMatch, maxSpatiallyNonCompatibleInstances)
    if len(sopInstDiscard)>0:
        print('\nloadDicomSeries() found images with more than one Orientation/Pixel spacing in series ' + seriesUID + ' ' + folder + '\n')
        return None

    # move the tags that match to the top level of the series dictionary
    for tag in tagsToMatch:
        series[tag] = series['SOPInstanceDict'][next(iter(series['SOPInstanceDict']))][tag]

    # delete tags from each sopInstance
    for k, v in series['SOPInstanceDict'].items():
        for tag in tagsToMatch:
            v.pop(tag)
        v['ContourList'] = []
        v['MaskList'] = []

    return series


def readRTSdata(rts, series):

    roiNameDict = {}
    roiNames = []
    for ssr in rts.StructureSetROISequence:
        roiNameDict[ssr.ROINumber] = ssr.ROIName
        roiNames.append(ssr.ROIName)
    series['ROINames'] = roiNames

    for rcs in rts.ROIContourSequence:
        for cs in rcs.ContourSequence:

            # extract information from ContourSequence items
            thisContour = {'ContourGeometricType':cs.ContourGeometricType,
                           'ContourNumber':cs.ContourNumber,
                           'ContourData':cs.ContourData,
                           'ROIName':roiNameDict[rcs.ReferencedROINumber],
                           'ROINumber':rcs.ReferencedROINumber}

            # get referenced SopInstance item from series
            thisSopInstance = series['SOPInstanceDict'][cs.ContourImageSequence[0].ReferencedSOPInstanceUID]

            # process contour data to get contour pixel coordinates, contour area and the mask
            colCoord, rowCoord, contourArea, mask = __processContour(cs.ContourData, thisSopInstance, series)
            thisContour['ColumnPixelCoordinates'] = colCoord
            thisContour['RowPixelCoordinates'] = rowCoord
            thisContour['ContourArea'] = contourArea
            thisContour['Mask'] = mask

            thisSopInstance['ContourList'].append(thisContour)

    return series