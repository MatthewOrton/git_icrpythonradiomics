import os, glob, pydicom
import numpy as np
from skimage import draw

class dataLoader:

    def __init__(self, assessorFile, seriesFolderDict, maxNonCompatibleInstances=0, verbose=False, roiShift={'row':0, 'col':0}, sliceLocationFromImagePositionPatient=True):

        if verbose:
            print('Processing ' + assessorFile)

        self.assessorFile = assessorFile
        self.seriesFolderDict = seriesFolderDict
        self.roiShift = roiShift

        # This is a list of dicom tags that are checked to see if they match for all sopInstances in the series
        self.tagsToMatch = ['Columns', 'Rows', 'ImageOrientationPatient', 'PixelSpacing', 'SliceThickness']

        # This is to allow series to be loaded that have given number of instances with tags that do not match self.tagsToMatch
        # This is because some series have a localizer in the same series, so this will be detected and removed
        self.maxNonCompatibleInstances = maxNonCompatibleInstances

        self.assessor = pydicom.dcmread(assessorFile)
        self.ReferencedSeriesUID = self.__getReferencedSeriesUID()

        # this is to switch between getting sliceLocation from dicom tag with same name, or by calculating it from ImagePositionPatient tag
        self.sliceLocationFromImagePositionPatient = sliceLocationFromImagePositionPatient

        self.seriesData = {}
        if verbose:
            print('Loading images ...', end=' ')

        self.__loadImageSeries()
        if not any(self.seriesData):
            print('Scan data not found!\n')
            return

        if verbose:
            print('loading contours ...', end=' ')

        if self.assessor.SOPClassUID == '1.2.840.10008.5.1.4.1.1.481.3':
            self.__loadRTSdata()

        if self.assessor.SOPClassUID == '1.2.840.10008.5.1.4.1.1.66.4':
            self.__loadSEGdata()

        if verbose:
            print('complete.')


    def __getReferencedSeriesUID(self):

        if self.assessor.SOPClassUID == '1.2.840.10008.5.1.4.1.1.481.3':
            # RT-STRUCT
            rfors = self.assessor.ReferencedFrameOfReferenceSequence[0]
            rtrss = rfors.RTReferencedStudySequence[0]
            return rtrss.RTReferencedSeriesSequence[0].SeriesInstanceUID

        if self.assessor.SOPClassUID == '1.2.840.10008.5.1.4.1.1.66.4':
            # SEG
            return self.assessor.ReferencedSeriesSequence[0].SeriesInstanceUID

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

        # Some scans have ImageOrientationPatient values that are the same to a tolerance (rather than exactly equal)
        # This causes problems when performing the hash of the tags to match.
        # Solution is to find groups of ImageOrientationPatient values that have directions within a tolerance of each other, and use
        # the matching values locally
        if 'ImageOrientationPatient' in self.tagsToMatch:
            imOriList = [value['ImageOrientationPatient'] for value in sopInstDict.values()]

            # initialise with first element as first element of unique list
            imOriTest = imOriList[0]
            todoList = [True]*len(imOriList)
            todoList[0] = False
            imOriTest_updated = False
            numUniqueValues = 1

            # set ImageOrientationPatient arrays exactly equal if they are within a tolerance
            while any(todoList):
                for n, imOri in enumerate(imOriList):
                    if todoList[n]:
                        dc0 = np.abs(np.dot(imOri[0:3], imOriTest[0:3])-1)
                        dc1 = np.abs(np.dot(imOri[3:], imOriTest[3:])-1)
                        tol = 1e-6
                        if dc0<tol and dc1<tol:
                            imOriList[n] = imOriTest
                            todoList[n] = False
                        elif not imOriTest_updated:
                            imOriTestNew = imOri.copy()
                            imOriTest_updated = True
                            todoList[n] = False
                            numUniqueValues += 1
                if imOriTest_updated:
                    imOriTest = imOriTestNew.copy()
                    imOriTest_updated = False

            # replace values into the original dictionary
            for n, key in enumerate(sopInstDict.keys()):
                sopInstDict[key]['ImageOrientationPatient'] = imOriList[n]

        # Generate a hash of tagsToMatch for each sopInstance and find the unique values.
        hashMeta = {key: hash(np.hstack([value[tag] for tag in self.tagsToMatch]).data.tobytes()) for key, value in sopInstDict.items()}
        hashMetaUnique = list(set([v for _, v in hashMeta.items()]))

        # Find number of sopInstances with each unique hash, where the number of instances is above self.maxNonCompatibleInstances
        hashMetaKeep = [x for x in hashMetaUnique if list(hashMeta.values()).count(x) > self.maxNonCompatibleInstances]
        if len(hashMetaKeep)>1:
            raise Exception(str(len(hashMetaKeep)) + ' sets of compatible SOPInstances found!')

        # Get hash values that have fewer than maxSpatiallyNonCompatibleInstances instances and can therefore be discarded
        hashMetaUniqueDiscard = [x for x in hashMetaUnique if list(hashMeta.values()).count(x) <= self.maxNonCompatibleInstances]

        # Get SopInstances to discard
        sopInstDiscard = []
        for key, value in hashMeta.items():
            if value in hashMetaUniqueDiscard:
                sopInstDiscard.append(key)

        return sopInstDiscard



    def getNamedROI(self, ROIName, minContourArea=0, checkMaskPresentOnContiguousSlices=True, sliceSpacingUniformityThreshold=1e-4, contiguousInstanceNumberCheck=True):

        output = {'ROIName':ROIName}

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
        InstanceNumbers = [x['InstanceNumber'] for x in SopInstanceList]
        instanceListSteps = np.unique(np.diff(np.sort(np.array(InstanceNumbers))))
        if contiguousInstanceNumberCheck and (len(instanceListSteps) != 1 or instanceListSteps[0] != 1):
            raise Exception('InstanceNumber values are not consecutive for AcquisitionNumber = ' + str(AcquisitionNumber))

        # Sort on InstanceNumber if possible and on SliceLocation if not
        if len(set(InstanceNumbers)) == len(InstanceNumbers):
            SopInstanceList = [x for _, x in sorted(zip(InstanceNumbers, SopInstanceList))]
            InstanceNumbers.sort()
        elif len(set(InstanceNumbers)) == 1:
            SliceLocationList = [x['SliceLocation'] for x in SopInstanceList]
            SopInstanceList = [x for _, x in sorted(zip(SliceLocationList, SopInstanceList))]
            SliceLocationList.sort()

        # Check SliceLocations are uniformly spaced (to a tolerance)
        SliceLocationList = [x['SliceLocation'] for x in SopInstanceList]
        SliceSpacing = np.diff(SliceLocationList)
        sliceRatioCheck = np.std(SliceSpacing) / np.abs(np.mean(SliceSpacing))
        if sliceRatioCheck > sliceSpacingUniformityThreshold:
            raise Exception('Non uniform slice spacing: slice ratio = ' + str(sliceRatioCheck))

        # Check other tags are the same for all SOPInstances
        if any([len(set([x['ImageOrientationPatient'][n] for x in SopInstanceList]))!=1 for n in range(6)]):
            raise Exception('SopInstances with non-matching ImageOrientationPatient')
        if len(set([x['PixelSpacing'][0] for x in SopInstanceList])) != 1 or len(set([x['PixelSpacing'][1] for x in SopInstanceList])) != 1:
            raise Exception('SopInstances with non-matching PixelSpacing')
        if len(set([x['Rows'] for x in SopInstanceList])) != 1:
            raise Exception('SopInstances with non-matching Rows values')
        if len(set([x['Columns'] for x in SopInstanceList])) != 1:
            raise Exception('SopInstances with non-matching Columns values')
        if len(set([x['SliceThickness'] for x in SopInstanceList])) != 1:
            raise Exception('SopInstances with non-matching SliceThickness')

        # Copy image data and make mask
        image = np.zeros((len(SopInstanceList), SopInstanceList[0]['Rows'], SopInstanceList[0]['Columns']))
        mask = np.zeros((len(SopInstanceList), SopInstanceList[0]['Rows'], SopInstanceList[0]['Columns'])).astype(bool)
        output['numberSmallContoursRemoved'] = 0
        for n, sopInst in enumerate(SopInstanceList):
            image[n,:,:] = sopInst['PixelData']
            for contour in sopInst['ContourList']:
                if contour['ROIName'] == ROIName:
                    if contour['ContourArea']>minContourArea:
                        mask[n,:,:] = np.logical_or(mask[n,:,:], contour['Mask'])
                    else:
                        output['numberSmallContoursRemoved'] += 1

        if checkMaskPresentOnContiguousSlices:
            maskSliceInds = np.unique(np.where(np.sum(mask, axis=(1,2))>0)[0])
            maskSliceIndsDiff = np.unique(np.diff(maskSliceInds))
            if len(maskSliceInds)==1 or (len(maskSliceIndsDiff) == 1 and maskSliceIndsDiff[0] == 1):
                output['maskContiguous'] = True
            else:
                output['maskContiguous'] = False
        else:
            output['maskContiguous'] = None

        # pyradiomics likes the mask and image to be SimpleITK image objects, but converting to these is slow
        #
        # Therefore, output includes all the metadata needed to construct the sitk objects, which we do at the point
        # where pyradiomics is invoked.  This means that manipulating the masks (e.g. dilating, combining masks etc.) is easier also

        output['mask'] = {}
        output['mask']['array'] = mask
        output['mask']['origin'] = tuple(SopInstanceList[0]['ImagePositionPatient'])
        output['mask']['spacing'] = (SopInstanceList[0]['PixelSpacing'][0], SopInstanceList[0]['PixelSpacing'][1], SliceSpacing[0])
        rowVec = SopInstanceList[0]['ImageOrientationPatient'][0:3]
        colVec = SopInstanceList[0]['ImageOrientationPatient'][3:6]
        output['mask']['direction'] = tuple(np.hstack((rowVec, colVec, np.cross(rowVec, colVec))))

        output['image'] = {}
        output['image']['array'] = image
        output['image']['origin'] = tuple(SopInstanceList[0]['ImagePositionPatient'])
        output['image']['spacing'] = (SopInstanceList[0]['PixelSpacing'][0], SopInstanceList[0]['PixelSpacing'][1], SliceSpacing[0])
        output['image']['direction'] = tuple(np.hstack((rowVec, colVec, np.cross(rowVec, colVec))))
        output['image']['InstanceNumbers'] = InstanceNumbers
        output['image']['SliceLocations'] = SliceLocationList
        output['image']['Files'] = [x['File'] for x in SopInstanceList]

        # template code to convert to sitk image object
        # imageSitk = sitk.GetImageFromArray(output['image']['array'])
        # imageSitk.SetOrigin(output['image']['origin'])
        # imageSitk.SetSpacing(output['image']['spacing'])
        # imageSitk.SetDirection(output['image']['direction'])

        return output


    def __loadImageSeries(self):

        if self.ReferencedSeriesUID not in self.seriesFolderDict:
            self.seriesData = {}
            return

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
                thisSopInst['File'] = file
                thisSopInst['PixelData'] = self.__getScaledSlice(dcm)
                thisSopInst['InstanceNumber'] = int(dcm.InstanceNumber)
                if hasattr(dcm, 'AcquisitionNumber') and dcm.AcquisitionNumber != '':
                    thisSopInst['AcquisitionNumber'] = int(dcm.AcquisitionNumber)
                else:
                    thisSopInst['AcquisitionNumber'] = 0
                if hasattr(dcm, 'SliceLocation') and dcm.SliceLocation != '':
                    thisSopInst['SliceLocation'] = float(dcm.SliceLocation)
                else:
                    thisSopInst['SliceLocation'] = 0.0
                thisSopInst['ImagePositionPatient'] = np.array([float(x) for x in dcm.data_element('ImagePositionPatient')])
                thisSopInst['ImageOrientationPatient'] = np.array([float(x) for x in dcm.data_element('ImageOrientationPatient')])
                thisSopInst['PixelSpacing'] = np.array([float(x) for x in dcm.data_element('PixelSpacing')])
                thisSopInst['Rows'] = int(dcm.Rows)
                thisSopInst['Columns'] = int(dcm.Columns)
                if hasattr(dcm, 'SliceThickness') and dcm.SliceThickness != '':
                        thisSopInst['SliceThickness'] = float(dcm.SliceThickness)
                else:
                    thisSopInst['SliceThickness'] = 0.0

                # Get SliceLocation directly from ImagePositionPatient and ImageOrientationPatient
                # This is because the SliceLocation dicom tag is sometimes dodgy
                if self.sliceLocationFromImagePositionPatient:
                    origin = thisSopInst['ImagePositionPatient']
                    rowVec = thisSopInst['ImageOrientationPatient'][0:3]
                    colVec = thisSopInst['ImageOrientationPatient'][3:6]
                    thisSopInst['SliceLocation'] = np.dot(origin, np.cross(rowVec, colVec))

                # even though SOPInstance used as key, include in value (dictionary) so we can double check matching SOPInstances later
                thisSopInst['SOPInstanceUID'] = dcm.SOPInstanceUID

                series['SOPInstanceDict'][dcm.SOPInstanceUID] = thisSopInst

        if 'dcmCommon' not in locals():
            raise Exception('SeriesInstanceUID ' + seriesUID + ' not found in folder ' + folder)

        # extract metadata that should be common to all instances in series
        series['SeriesInstanceUID'] = dcmCommon.SeriesInstanceUID
        series['StudyInstanceUID'] = dcmCommon.StudyInstanceUID
        series['PatientID'] = dcmCommon.PatientID
        series['PatientName'] = str(dcmCommon.PatientName)
        series['StudyDate'] = dcmCommon.StudyDate
        series['SeriesTime'] = dcmCommon.SeriesTime

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

        # For each Acquisition remove non-matching sopInstances (limited to groups that are smaller than self.maxNonCompatibleInstances)
        AcquisitionNumbers = list(set([x['AcquisitionNumber'] for x in series['SOPInstanceDict'].values()]))
        for AcquisitionNumber in AcquisitionNumbers:

            thisAcquisition = {key: value for key, value in series['SOPInstanceDict'].items() if value['AcquisitionNumber'] == AcquisitionNumber}
            for sop in self.__getNonMatchingSopInstances(thisAcquisition):
                series['SOPInstanceDict'].pop(sop, None)

            # Double check that the collection of sopInstances is now compatible
            thisAcquisition = {key: value for key, value in series['SOPInstanceDict'].items() if value['AcquisitionNumber'] == AcquisitionNumber}
            sopInstDiscard = self.__getNonMatchingSopInstances(thisAcquisition)
            if len(sopInstDiscard)>0:
                raise Exception('Series has SOPInstance(s) with non-compatible metadata')

        # # Move the tags that do match to the top level of the series dictionary
        # for tag in self.tagsToMatch:
        #     series[tag] = series['SOPInstanceDict'][next(iter(series['SOPInstanceDict']))][tag]

        # # Delete the tags from each sopInstance
        # for k, v in series['SOPInstanceDict'].items():
            # for tag in self.tagsToMatch:
            #     v.pop(tag)
            # add blank items relating to the RTstruct data that will be filled in with non-empty data in a different function

        # add blank items relating to the RTstruct data that will be filled in with non-empty data in a different function
        for k, v in series['SOPInstanceDict'].items():
            v['ContourList'] = []
            v['MaskList'] = []


        self.seriesData = series

    def __loadSEGdata(self):

        # find names and labels of all segmentation objects
        segNameDict = {}
        segNames = []
        for ss in self.assessor.SegmentSequence:
            segNameDict[ss.SegmentNumber] = ss.SegmentLabel
            segNames.append(ss.SegmentLabel)
        self.seriesData['ROINames'] = segNames

        maskFrames = self.assessor.pixel_array
        # make sure single slice masks have rows/cols/slices along correct dimension
        if len(maskFrames.shape) == 2:
            maskFrames = np.reshape(maskFrames, (1, maskFrames.shape[0], maskFrames.shape[1]))  # the dimension order needs testing!!

        for n, funGrpSeq in enumerate(self.assessor.PerFrameFunctionalGroupsSequence):
            thisSopInstUID = funGrpSeq.DerivationImageSequence[0].SourceImageSequence[0].ReferencedSOPInstanceUID
            thisSopInstance = self.seriesData['SOPInstanceDict'][thisSopInstUID]
            segmentNumber = funGrpSeq.SegmentIdentificationSequence[0].ReferencedSegmentNumber

            thisMask = {'FrameNumber':n,
                        'ROIName': segNameDict[segmentNumber],
                        'ROINumber': segmentNumber}

            thisMask['Mask'] = maskFrames[n,:,:]
            thisMask['ReferencedSOPInstanceUID'] = thisSopInstUID

            thisSopInstance['MaskList'].append(thisMask)


    def __loadRTSdata(self):

        # find names and labels of all ROIs
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
                thisContour['ReferencedSOPInstanceUID'] = cs.ContourImageSequence[0].ReferencedSOPInstanceUID

                thisSopInstance['ContourList'].append(thisContour)


    def __processContour(self, contourData, sopInstance):

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
        spacing = sopInstance['PixelSpacing']
        rowVec = np.reshape(sopInstance['ImageOrientationPatient'][0:3], (3,1))
        colVec = np.reshape(sopInstance['ImageOrientationPatient'][3:6], (3,1))
        rowPixCoord = np.dot(polygonPatient - origin, rowVec) / spacing[1]
        colPixCoord = np.dot(polygonPatient - origin, colVec) / spacing[0]

        # get mask from contour
        mask = np.zeros((sopInstance['Rows'], sopInstance['Columns'])).astype(bool)
        fill_row_coords, fill_col_coords = draw.polygon(colPixCoord.ravel() + self.roiShift['col'], rowPixCoord.ravel() + self.roiShift['row'], (sopInstance['Columns'], sopInstance['Rows']))
        mask[fill_row_coords, fill_col_coords] = True

        return colPixCoord, rowPixCoord, contourArea, mask