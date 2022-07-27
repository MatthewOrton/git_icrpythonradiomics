import os, glob, pydicom
import numpy as np
from skimage import draw

class dataLoader:

    def __init__(self, assessorFile, seriesFolderDict, maxNonCompatibleInstances=0):

        self.assessorFile = assessorFile
        self.seriesFolderDict = seriesFolderDict

        # This is a list of dicom tags that are checked to see if they match for all sopInstances in the series
        self.tagsToMatch = ['Columns', 'Rows', 'ImageOrientationPatient', 'PixelSpacing', 'SliceThickness']

        # This is to allow series to be loaded that have given number of instances with tags that do not match self.tagsToMatch
        # This is because some series have a localizer in the same series, so this will be detected and removed
        self.maxNonCompatibleInstances = maxNonCompatibleInstances

        self.assessor = pydicom.dcmread(assessorFile)
        self.ReferencedSeriesUID = self.__getReferencedSeriesUID()

        self.seriesData = None
        self.__loadImageSeries()

        self.__loadRTSdata()


    def __getReferencedSeriesUID(self):
        rfors = self.assessor.ReferencedFrameOfReferenceSequence[0]
        rtrss = rfors.RTReferencedStudySequence[0]
        return rtrss.RTReferencedSeriesSequence[0].SeriesInstanceUID


    def __getScaledSlice(self, dcm):

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

    def __getNonMatchingSopInstances(self, sopInstDict):

        # Ideally all sopInstances in a series will have certain matching attributes, e.g. ImageOrientationPatient, PixelSpacing etc.
        #
        # Sometimes there are a small (usually 1) number that don't, e.g. one image that is a coronal reformat that shows position of axial slices.
        #
        # If the number of non-matching images is below maxSpatiallyNonCompatibleInstances then get output a list of the sopInstances to discard.

        # Generate a hash of tagsToMatch for each sopInstance and find the unique values.
        hashMeta = {key: hash(np.hstack([value[tag] for tag in self.tagsToMatch]).data.tobytes()) for key, value in sopInstDict.items()}
        hashMetaUnique = list(set([v for _, v in hashMeta.items()]))

        # Find number of sopInstances with each unique hash, where the number of instances is above self.maxNonCompatibleInstances
        hashMetaKeep = [x for x in hashMetaUnique if list(hashMeta.values()).count(x) > self.maxNonCompatibleInstances]
        if len(hashMetaKeep)>1:
            raise Exception('More than one set of compatible SOPInstances found')

        # Get hash values that have fewer than maxSpatiallyNonCompatibleInstances instances and can therefore be discarded
        hashMetaUniqueDiscard = [x for x in hashMetaUnique if list(hashMeta.values()).count(x) <= self.maxNonCompatibleInstances]

        # Get SopInstances to discard
        sopInstDiscard = []
        for key, value in hashMeta.items():
            if value in hashMetaUniqueDiscard:
                sopInstDiscard.append(key)

        return sopInstDiscard



    def getNamedROI(self, ROIName):

        if ROIName not in self.seriesData['ROINames']:
            raise Exception(ROIName + ' not found!')

        # Find which AcquisitionNumber the named ROI is in
        AcquisitionList = []
        for k, v in self.seriesData['SOPInstanceDict'].items():
            if len(v['ContourList'])>0:
                for contour in v['ContourList']:
                    if contour['ROIName'] == ROIName:
                        AcquisitionList.append(v['AcquisitionNumber'])
        AcquisitionNumber = list(set(AcquisitionList))
        if len(AcquisitionNumber)>1:
            raise Exception(ROIName + ' spans more than one Acquisition!')
        AcquisitionNumber = AcquisitionNumber[0]

        # Get SopInstances from this Acquisition
        SopInstanceList = [x for x in self.seriesData['SOPInstanceDict'].values() if x['AcquisitionNumber'] == AcquisitionNumber]

        # Get list of InstanceNumbers and check values are contiguous
        InstanceNumberList = [x['InstanceNumber'] for x in SopInstanceList]
        if len(InstanceNumberList) != len(set(InstanceNumberList)) or min(InstanceNumberList) != 1 or max(InstanceNumberList) != len(set(InstanceNumberList)):
            raise Exception('InstanceNumber values are not consecutive for AcquisitionNumber = ' + str(AcquisitionNumber))

        # Sort on InstanceNumber
        SopInstanceList = [x for _, x in sorted(zip(InstanceNumberList, SopInstanceList))]

        # Copy image data and make mask
        image = np.zeros((len(SopInstanceList), self.seriesData['Columns'], self.seriesData['Rows']))
        mask = np.zeros((len(SopInstanceList), self.seriesData['Columns'], self.seriesData['Rows'])).astype(bool)
        for n, sopInst in enumerate(SopInstanceList):
            image[n,:,:] = sopInst['PixelData']
            for contour in sopInst['ContourList']:
                if contour['ROIName'] == ROIName:
                    mask[n,:,:] = np.logical_or(mask[n,:,:], contour['Mask'])

        return image, mask


    def __loadImageSeries(self):

        series = {}
        series['SOPInstanceDict'] = {}

        files = glob.glob(os.path.join(self.seriesFolderDict[self.ReferencedSeriesUID], '**'), recursive=True)
        for file in files:
            if not os.path.isdir(file) and pydicom.misc.is_dicom(file):

                dcm = pydicom.dcmread(file)

                if dcm.SeriesInstanceUID != self.ReferencedSeriesUID:
                    continue

                # keep one dcm for getting tags common to all instances in series
                if 'dcmCommon' not in locals():
                    dcmCommon = pydicom.dcmread(file)

                thisSopInst = {}
                thisSopInst['PixelData'] = self.__getScaledSlice(dcm)
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

        if 'dcmCommon' not in locals():
            raise Exception('SeriesInstanceUID ' + seriesUID + ' not found in folder ' + folder)

        # extract metadata that should be common to all instances in series
        series['SeriesInstanceUID'] = dcmCommon.SeriesInstanceUID
        series['StudyInstanceUID'] = dcmCommon.StudyInstanceUID
        series['PatientID'] = dcmCommon.PatientID

        # get any Modality-specific parameters that might be useful
        sopClassUid_CTImageStorage = '1.2.840.10008.5.1.4.1.1.2'
        sopClassUid_MRImageStorage = '1.2.840.10008.5.1.4.1.1.4'
        if dcmCommon.SOPClassUID == sopClassUid_CTImageStorage:
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
                if hasattr(dcmCommon, parameter):
                    series[parameter] = getattr(dcmCommon, parameter)
                else:
                    series[parameter] = None

        # Remove non-matching sopInstances (limited to groups that are smaller than self.maxNonCompatibleInstances)
        for sop in self.__getNonMatchingSopInstances(series['SOPInstanceDict']):
            series['SOPInstanceDict'].pop(sop, None)

        # Double check that the collection of sopInstances is now compatible
        sopInstDiscard = self.__getNonMatchingSopInstances(series['SOPInstanceDict'])
        if len(sopInstDiscard)>0:
            raise Exception('Series has SOPInstance(s) with non-compatible metadata')

        # Move the tags that do match to the top level of the series dictionary
        for tag in self.tagsToMatch:
            series[tag] = series['SOPInstanceDict'][next(iter(series['SOPInstanceDict']))][tag]

        # Delete the tags from each sopInstance
        for k, v in series['SOPInstanceDict'].items():
            for tag in self.tagsToMatch:
                v.pop(tag)
            v['ContourList'] = []
            v['MaskList'] = []

        self.seriesData = series


    def __loadRTSdata(self):

        roiNameDict = {}
        roiNames = []
        for ssr in self.assessor.StructureSetROISequence:
            roiNameDict[ssr.ROINumber] = ssr.ROIName
            roiNames.append(ssr.ROIName)
        self.seriesData['ROINames'] = roiNames

        for rcs in self.assessor.ROIContourSequence:
            for cs in rcs.ContourSequence:

                # extract information from ContourSequence items
                thisContour = {'ContourGeometricType':cs.ContourGeometricType,
                               'ContourNumber':cs.ContourNumber,
                               'ContourData':cs.ContourData,
                               'ROIName':roiNameDict[rcs.ReferencedROINumber],
                               'ROINumber':rcs.ReferencedROINumber}

                # get referenced SopInstance item from series
                thisSopInstance = self.seriesData['SOPInstanceDict'][cs.ContourImageSequence[0].ReferencedSOPInstanceUID]

                # process contour data to get contour pixel coordinates, contour area and the mask
                colCoord, rowCoord, contourArea, mask = self.__processContour(cs.ContourData, thisSopInstance)
                thisContour['ColumnPixelCoordinates'] = colCoord
                thisContour['RowPixelCoordinates'] = rowCoord
                thisContour['ContourArea'] = contourArea
                thisContour['Mask'] = mask

                thisSopInstance['ContourList'].append(thisContour)


    def __processContour(self, contourData, sopInstance, roiShift={'row':0, 'col':0}):

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
        spacing = self.seriesData['PixelSpacing']
        colNorm = np.reshape(self.seriesData['ImageOrientationPatient'][0:3], (3,1))
        rowNorm = np.reshape(self.seriesData['ImageOrientationPatient'][3:6], (3,1))
        colPixCoord = np.dot(polygonPatient - origin, colNorm) / spacing[0]
        rowPixCoord = np.dot(polygonPatient - origin, rowNorm) / spacing[1]

        # according to https://scikit-image.org/docs/stable/api/skimage.draw.html?highlight=skimage%20draw#module-skimage.draw
        # there is a function polygon2mask, but this doesn't seem to be actually present in the library I have.
        # Since draw.polygon2mask is just a wrapper for draw.polygon I'm using the simpler function directly here.
        mask = np.zeros((self.seriesData['Rows'], self.seriesData['Columns'])).astype(bool)
        fill_row_coords, fill_col_coords = draw.polygon(rowPixCoord + roiShift['row'], colPixCoord + roiShift['col'], (self.seriesData['Columns'], self.seriesData['Rows']))
        mask[fill_row_coords, fill_col_coords] = True

        return colPixCoord, rowPixCoord, contourArea, mask