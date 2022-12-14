import os, glob, pydicom, csv, uuid, cv2
import numpy as np
from skimage import draw
import matplotlib.pyplot as plt
import SimpleITK as sitk
from radiomics import featureextractor, setVerbosity


class radiomicAnalyser2:

    def __init__(self, assessorFile, seriesFolderDict, paramFileName, maxNonCompatibleInstances=0, verbose=False, roiShift={'row':0, 'col':0}, sliceLocationFromImagePositionPatient=True):

        if verbose:
            print('Processing ' + assessorFile)

        self.assessorFile = assessorFile
        self.seriesFolderDict = seriesFolderDict
        self.paramFileName = paramFileName
        self.roiShift = roiShift

        # This is a list of dicom tags that are checked to see if they match for all sopInstances in the series
        self.tagsToMatch = ['Columns', 'Rows', 'ImageOrientationPatient', 'PixelSpacing', 'SliceThickness']

        # This is to allow series to be loaded that have given number of instances with tags that do not match self.tagsToMatch
        # This is because some series have a localizer in the same series, so this will be detected and removed
        self.maxNonCompatibleInstances = maxNonCompatibleInstances

        self.assessorDcm = pydicom.dcmread(assessorFile)
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

        if self.assessorDcm.SOPClassUID == '1.2.840.10008.5.1.4.1.1.481.3':
            self.__loadRTSdata()

        if self.assessorDcm.SOPClassUID == '1.2.840.10008.5.1.4.1.1.66.4':
            self.__loadSEGdata()

        if verbose:
            print('complete.')


    def __getReferencedSeriesUID(self):

        if self.assessorDcm.SOPClassUID == '1.2.840.10008.5.1.4.1.1.481.3':
            # RT-STRUCT
            rfors = self.assessorDcm.ReferencedFrameOfReferenceSequence[0]
            rtrss = rfors.RTReferencedStudySequence[0]
            return rtrss.RTReferencedSeriesSequence[0].SeriesInstanceUID

        if self.assessorDcm.SOPClassUID == '1.2.840.10008.5.1.4.1.1.66.4':
            # SEG
            return self.assessorDcm.ReferencedSeriesSequence[0].SeriesInstanceUID

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



    def __getNamedROI(self, ROIName, minContourArea=0, checkMaskPresentOnContiguousSlices=True, sliceSpacingUniformityThreshold=1e-4, contiguousInstanceNumberCheck=True):

        if ROIName not in self.seriesData['ROINames']:
            raise Exception(ROIName + ' not found!')

        # Find which AcquisitionNumber the named ROI is in - look in ContourList and MaskList (although will only ever be all Contours or all Masks)
        AcquisitionList = []
        for k, v in self.seriesData['SOPInstanceDict'].items():
            if len(v['ContourList']) > 0:
                for contour in v['ContourList']:
                    if contour['ROIName'] == ROIName:
                        AcquisitionList.append(v['AcquisitionNumber'])
            if len(v['MaskList']) > 0:
                for msk in v['MaskList']:
                    if msk['ROIName'] == ROIName:
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

        thisROI = {'ROIName': ROIName}

        # Check SliceLocations are uniformly spaced (to a tolerance)
        SliceLocList =  []
        for sopInst in SopInstanceList:
            rowVec = np.array(sopInst['ImageOrientationPatient'][0:3])
            colVec = np.array(sopInst['ImageOrientationPatient'][3:6])
            SliceLocList.append(np.dot(np.array(sopInst['ImagePositionPatient']), np.cross(rowVec, colVec)))
        sliceGaps = np.diff(SliceLocList)
        sliceRatioCheck = np.std(sliceGaps) / np.abs(np.mean(sliceGaps))
        if sliceRatioCheck > sliceSpacingUniformityThreshold:
            raise Exception('Non uniform slice spacing: slice ratio = ' + str(sliceRatioCheck))
        thisROI['sliceSpacing'] = np.abs(sliceGaps[0])

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
        thisROI['numberSmallContoursRemoved'] = 0
        for n, sopInst in enumerate(SopInstanceList):
            image[n,:,:] = sopInst['PixelData']
            for contour in sopInst['ContourList']:
                if contour['ROIName'] == ROIName:
                    if contour['ContourArea']>minContourArea:
                        mask[n,:,:] = np.logical_or(mask[n,:,:], contour['Mask'])
                    else:
                        thisROI['numberSmallContoursRemoved'] += 1
            for msk in sopInst['MaskList']:
                if msk['ROIName'] == ROIName:
                    mask[n,:,:] = np.logical_or(mask[n,:,:], msk['Mask'])

        if checkMaskPresentOnContiguousSlices:
            maskSliceInds = np.unique(np.where(np.sum(mask, axis=(1,2))>0)[0])
            maskSliceIndsDiff = np.unique(np.diff(maskSliceInds))
            if len(maskSliceInds)==1 or (len(maskSliceIndsDiff) == 1 and maskSliceIndsDiff[0] == 1):
                thisROI['maskContiguous'] = True
            else:
                thisROI['maskContiguous'] = False
        else:
            thisROI['maskContiguous'] = None

        # pyradiomics likes the mask and image to be SimpleITK image objects, but converting to these is slow
        #
        # Therefore, thisROI includes all the metadata needed to construct the sitk objects, which we do at the point
        # where pyradiomics is invoked.  This means that manipulating the masks (e.g. dilating, combining masks etc.) is easier also

        thisROI['mask'] = {}
        thisROI['mask']['array'] = mask
        thisROI['mask']['origin'] = tuple(SopInstanceList[0]['ImagePositionPatient'])
        thisROI['mask']['spacing'] = (SopInstanceList[0]['PixelSpacing'][0], SopInstanceList[0]['PixelSpacing'][1], thisROI['sliceSpacing'])
        rowVec = SopInstanceList[0]['ImageOrientationPatient'][0:3]
        colVec = SopInstanceList[0]['ImageOrientationPatient'][3:6]
        thisROI['mask']['direction'] = tuple(np.hstack((rowVec, colVec, np.cross(rowVec, colVec))))

        thisROI['image'] = {}
        thisROI['image']['array'] = image
        thisROI['image']['origin'] = tuple(SopInstanceList[0]['ImagePositionPatient'])
        thisROI['image']['spacing'] = (SopInstanceList[0]['PixelSpacing'][0], SopInstanceList[0]['PixelSpacing'][1], thisROI['sliceSpacing'])
        thisROI['image']['direction'] = tuple(np.hstack((rowVec, colVec, np.cross(rowVec, colVec))))
        thisROI['image']['WindowCenter'] = SopInstanceList[0]['WindowCenter']
        thisROI['image']['WindowWidth'] = SopInstanceList[0]['WindowWidth']
        thisROI['image']['InstanceNumbers'] = InstanceNumbers
        thisROI['image']['SliceLocations'] = [x['SliceLocation'] for x in SopInstanceList]
        thisROI['image']['Files'] = [x['File'] for x in SopInstanceList]

        self.selectedROI = thisROI.copy()


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
                thisSopInst['WindowCenter'] = dcm.WindowCenter
                thisSopInst['WindowWidth'] = dcm.WindowWidth

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
        if 'Manufacturer' in dcmCommon:
            series['Manufacturer'] = dcmCommon.Manufacturer
        else:
            series['Manufacturer'] = 'unknown'
        if 'ManufacturerModelName' in dcmCommon:
            series['ManufacturerModelName'] = dcmCommon.ManufacturerModelName
        else:
            series['ManufacturerModelName'] = 'unknown'

        # get any Modality-specific parameters that might be useful
        sopClassUid_CTImageStorage = '1.2.840.10008.5.1.4.1.1.2'
        sopClassUid_MRImageStorage = '1.2.840.10008.5.1.4.1.1.4'
        series['ModalitySpecificParameters'] = {}
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
                    series['ModalitySpecificParameters'][parameter] = getattr(dcmCommon, parameter)
                else:
                    series['ModalitySpecificParameters'][parameter] = None

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

        # add blank items relating to the RTstruct data that will be filled in with non-empty data in a different function
        for k, v in series['SOPInstanceDict'].items():
            v['ContourList'] = []
            v['MaskList'] = []


        self.seriesData = series

    def __loadSEGdata(self):

        # find names and labels of all segmentation objects
        segNameDict = {}
        segNames = []
        for ss in self.assessorDcm.SegmentSequence:
            segNameDict[ss.SegmentNumber] = ss.SegmentLabel
            segNames.append(ss.SegmentLabel)
        self.seriesData['ROINames'] = segNames

        maskFrames = self.assessorDcm.pixel_array
        # make sure single slice masks have rows/cols/slices along correct dimension
        if len(maskFrames.shape) == 2:
            maskFrames = np.reshape(maskFrames, (1, maskFrames.shape[0], maskFrames.shape[1]))  # the dimension order needs testing!!

        for n, funGrpSeq in enumerate(self.assessorDcm.PerFrameFunctionalGroupsSequence):
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
        for ssr in self.assessorDcm.StructureSetROISequence:
            roiNameDict[ssr.ROINumber] = ssr.ROIName
            roiNames.append(ssr.ROIName)
        self.seriesData['ROINames'] = roiNames

        for rcs in self.assessorDcm.ROIContourSequence:
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

    ##########################
    # featureKeyPrefixStr can be used to add a prefix to the feature keys in order to manually identify features that have
    # been computed in a particular way.  E.g. when permuting the voxels I use 'permuted_' as a prefix
    # __getNamedROI(self, ROIName, minContourArea=0, checkMaskPresentOnContiguousSlices=True,
    #               sliceSpacingUniformityThreshold=1e-4, contiguousInstanceNumberCheck=True):

    def computePyradiomicsFeatures(self,
                                   roiName,
                                   minContourArea=0,
                                   checkMaskPresentOnContiguousSlices=True,
                                   sliceSpacingUniformityThreshold=1e-4,
                                   contiguousInstanceNumberCheck=True,
                                   computePercentiles=False,
                                   imageThresholds=None,
                                   binWidthOverRide=None,
                                   binCountOverRide=None,
                                   binEdgesOverRide=None,
                                   gldm_aOverRide=None,
                                   distancesOverRide=None,
                                   resampledPixelSpacing=None,
                                   featureKeyPrefixStr=''):

        self.__getNamedROI(roiName,
                           minContourArea=minContourArea,
                           checkMaskPresentOnContiguousSlices=checkMaskPresentOnContiguousSlices,
                           sliceSpacingUniformityThreshold=sliceSpacingUniformityThreshold,
                           contiguousInstanceNumberCheck=contiguousInstanceNumberCheck)

        maskSitk = sitk.GetImageFromArray(self.selectedROI['mask']['array'].astype(int))
        maskSitk.SetOrigin(self.selectedROI['mask']['origin'])
        maskSitk.SetSpacing(self.selectedROI['mask']['spacing'])
        maskSitk.SetDirection(self.selectedROI['mask']['direction'])

        if imageThresholds is None:
            imageSitk = sitk.GetImageFromArray(self.selectedROI['image']['array'])
        else:
            imageData = copy.deepcopy(self.selectedROI['image']['array'])
            imageData[imageData > np.max(imageThresholds)] = np.max(imageThresholds)
            imageData[imageData < np.min(imageThresholds)] = np.min(imageThresholds)
            imageSitk = sitk.GetImageFromArray(imageData)
        imageSitk.SetOrigin(maskSitk.GetOrigin())
        imageSitk.SetSpacing(maskSitk.GetSpacing())
        imageSitk.SetDirection(maskSitk.GetDirection())

        extractor = featureextractor.RadiomicsFeatureExtractor(self.paramFileName)
        setVerbosity(40)

        if binEdgesOverRide is not None:
            extractor.settings["binEdges"] = binEdgesOverRide
            extractor.settings["binWidth"] = None
            extractor.settings["binCount"] = None

        if binWidthOverRide is not None:
            extractor.settings["binEdges"] = None
            extractor.settings["binWidth"] = binWidthOverRide
            extractor.settings["binCount"] = None

        if binCountOverRide is not None:
            extractor.settings["binEdges"] = None
            extractor.settings["binWidth"] = None
            extractor.settings["binCount"] = binCountOverRide

        if gldm_aOverRide is not None:
            extractor.settings['gldm_a'] = gldm_aOverRide

        if distancesOverRide is not None:
            extractor.settings['distances'] = distancesOverRide

        if resampledPixelSpacing is not None:
            extractor.settings["resampledPixelSpacing"] = resampledPixelSpacing
            extractor.settings["interpolator"] = "sitkLinear"

        # have added functionality to RadiomicsFeatureExtractor that exposes the probability matrices and filteredImages
        # so we can evaluate whether they make sense or not.  In particular, the binWidths
        segmentNumber = int(1)
        featureVector, probabilityMatrices, filteredImages, quantizedImages = extractor.execute(imageSitk, maskSitk,
                                                                                                segmentNumber)

        if computePercentiles and 'firstorder' in extractor.enabledFeatures:
            pixels = self.selectedROI['image']['array'][self.selectedROI['mask']['array']]
            # pyradiomics already computes the 10th, 50th and 90th centiles, so skip these here
            qs = np.hstack((0.05, np.linspace(0.15, 0.45, 7), np.linspace(0.55, 0.85, 7), 0.95))
            for q in qs:
                featureVector['original_histogram_' + str((q * 100).round().astype(int)) + 'Percentile'] = np.quantile(
                    pixels, q)

        for key in featureVector.keys():
            featureVector[featureKeyPrefixStr + key] = featureVector[key]

        self.selectedROI['featureVector'] = featureVector

        print(self.selectedROI['ROIName'] + ' : Radiomic features computed')




    ##########################
    def saveRadiomicsFeatures(self, outputFileName, ProjectName='', StudyPatientName='', includeHeader=True, fileSubscript=''):

        headers = []
        row = []

        # add XNAT info so we can convert to DICOM SR later
        headers.append("source_XNAT_project")
        row.append(ProjectName)

        headers.append("StudyPatientName")
        row.append(str(StudyPatientName))

        fileName = os.path.split(self.assessorFile)[1]
        fileParts = fileName.split("__II__")

        # cheat for cases that haven't been downloaded from XNAT, and therefore have different filename structure
        if len(fileParts) != 4:
            fileNameNoExt = fileName.split('.')[0]
            fileParts = [fileNameNoExt, fileNameNoExt, fileNameNoExt, fileName]

        headers.append("source_XNAT_session")
        row.append(fileParts[1])

        headers.append("source_XNAT_scan")
        row.append(fileParts[2])

        headers.append("source_XNAT_assessor")
        row.append(fileParts[3].split('.')[0])

        headers.append("source_DCM_PatientName")
        row.append(str(self.assessorDcm.PatientName))

        headers.append("source_DCM_SeriesInstanceUID")
        row.append(self.ReferencedSeriesUID)

        headers.append("source_DCM_StudyDate")
        row.append(self.assessorDcm.StudyDate)

        headers.append("source_DCM_StudyTime")
        row.append(self.assessorDcm.StudyTime)

        headers.append("source_DCM_sliceGap_dz")
        row.append(self.selectedROI['sliceSpacing'])

        # mark some columns with string "QueryConfounder" then we can use this later in an automatic
        # check to filter these columns and check if there are any unwanted correlations/clusterings
        # with these parameters
        headers.append("source_DCM_Manufacturer_QueryConfounder")
        row.append(self.seriesData['Manufacturer'])

        headers.append("source_DCM_ManufacturerModelName_QueryConfounder")
        row.append(self.seriesData['ManufacturerModelName'])

        for key, value in self.seriesData['ModalitySpecificParameters'].items():
            headers.append("source_DCM_"+key+"_QueryConfounder")
            row.append(value)

        headers.append("source_annotationUID")
        row.append(self.assessorDcm.SOPInstanceUID)

        headers.append("source_ImageAnnotationCollectionDescription")
        if self.assessorDcm.SOPClassUID=='1.2.840.10008.5.1.4.1.1.481.3':
            # RTS
            row.append(self.assessorDcm.StructureSetLabel)
        if self.assessorDcm.SOPClassUID=='1.2.840.10008.5.1.4.1.1.66.4':
            # SEG
            row.append(self.assessorDcm.SeriesDescription)

        headers.append("source_roiLabel")
        row.append(self.selectedROI['ROIName'])

        headers.extend(list(self.selectedROI['featureVector'].keys()))
        for h in list(self.selectedROI['featureVector'].keys()):
            # special case that needs modification
            # diagnostics_Configuration_Settings element may contain a list of binEdges, and this will usually be too long to fit into the csv output
            # modify the binEdges variable so that it becomes a 3 element list with [start, stop, step]
            if 'diagnostics_Configuration_Settings' in h:
                thisFeature = self.selectedROI['featureVector'].get(h)
                if 'binEdges' in thisFeature.keys() and thisFeature['binEdges'] is not None:
                    thisFeature['binEdges'] = [thisFeature['binEdges'][0], thisFeature['binEdges'][-1], thisFeature['binEdges'][1]-thisFeature['binEdges'][0]]
                    self.selectedROI['featureVector'][h] = thisFeature
            row.append(self.selectedROI['featureVector'].get(h, "N/A"))

        # sort out file name
        outputPath = os.path.split(outputFileName)
        if not os.path.exists(outputPath[0]):
            os.makedirs(outputPath[0])

        # add subscript to filename
        fileName = outputPath[1].split('.')[0] + fileSubscript + '.csv'

        # replace dicom patient name with study patient name.  These should be the same, but this is a useful step if they are not
        fileName = fileName.replace(str(self.assessorDcm.PatientName), StudyPatientName)

        outputName = os.path.join(outputPath[0], fileName)

        if os.path.exists(outputName):
            outputName = outputName.replace('.csv', '_'+str(uuid.uuid1())+'.csv')
            print('\033[1;31;48m' + '_' * 50)
            print('File name clash!! Added UID for uniqueness')
            print('_' * 50 + '\033[0;30;48m')

        with open(outputName, 'w') as fo:
            writer = csv.writer(fo, lineterminator='\n')
            if includeHeader:
                writer.writerow(headers)
            writer.writerow(row)

        print(self.selectedROI['ROIName'] + " : Results file saved")

        return outputName


def saveThumbnail(roiList, outputFileName, titleStr='', cropToMask=True, showInstanceNumbers=True,
                  showSliceLocations=False, imageGrayLevelLimits=None, volumePad=[2, 20, 20], format='pdf'):

    # check all rois are linked to same image volume data
    match = True
    for roi in roiList:
        if roi['image']['array'].shape != roiList[0]['image']['array'].shape:
            match = False
            break
        match = match and np.all(roi['image']['array'] == roiList[0]['image']['array'])
    if not match:
        print('Non-matching image')
    else:
        imageVolume = roiList[0]['image']

    # set default image gray level limits if none input
    if imageGrayLevelLimits is None:
        imageGrayLevelLimits = [imageVolume['WindowCenter'] - 0.5*imageVolume['WindowWidth'],
                                imageVolume['WindowCenter'] + 0.5*imageVolume['WindowWidth']]

    # get bounding box for all masks
    mask = roiList[0]['mask']['array']
    for roi in roiList:
        mask = np.logical_or(mask, roi['mask']['array'])
    axis0 = np.where(np.sum(mask.astype(int), axis=(1, 2)) > 0)[0]
    axis1 = np.where(np.sum(mask.astype(int), axis=(0, 2)) > 0)[0]
    axis2 = np.where(np.sum(mask.astype(int), axis=(0, 1)) > 0)[0]

    if cropToMask:
        idx0 = range(max((0, axis0[0] - volumePad[0])), min((mask.shape[0], axis0[-1] + volumePad[0] + 1)))
        idx1 = range(max((0, axis1[0] - volumePad[1])), min((mask.shape[1], axis1[-1] + volumePad[1] + 1)))
        idx2 = range(max((0, axis2[0] - volumePad[2])), min((mask.shape[2], axis2[-1] + volumePad[2] + 1)))
    else:
        idx0 = range(mask.shape[0])
        idx1 = range(mask.shape[1])
        idx2 = range(mask.shape[2])

    # crop image and all masks
    imageVolumeCrop = imageVolume['array'][idx0, :, :][:, idx1, :][:, :,
                      idx2]  # this discards lots of the metadata and converts imageVolume from a dict to numpy array
    for roi in roiList:
        roi['mask'] = roi['mask']['array'][idx0, :, :][:, idx1, :][:, :,
                      idx2]  # this discards some metadata that we don't need

    # + 2 is so the legend goes on an empty subplot
    nPlt = imageVolumeCrop.shape[0] + 2
    pltRows = int(np.round(np.sqrt(2 * nPlt / 3))) + 1
    pltCols = int(np.ceil(nPlt / pltRows))

    # main figure and axes
    fPlt, axarr = plt.subplots(pltRows, pltCols, gridspec_kw={'wspace': 0.02, 'hspace': 0.02})

    # dummy figure so we can use contour() to get outline of the mask
    figContourDummy, axContourDummy = plt.subplots(1, 1)

    instanceNumbers = np.array(imageVolume['InstanceNumbers'])[idx0]
    sliceLocations = np.array(imageVolume['SliceLocations'])[idx0]

    # make each image and overlay all contours on this image
    colors = ['#d62728', '#28f2ff', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
              '#17becf']
    for slice, ax in enumerate(fPlt.axes):
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if slice < imageVolumeCrop.shape[0]:

            # plot image
            ax.imshow(imageVolumeCrop[slice, :, :], vmin=imageGrayLevelLimits[0], vmax=imageGrayLevelLimits[1],
                      cmap='gray', interpolation='nearest')

            # box with InstanceNumber
            if showInstanceNumbers:
                ax.text(0, 1, str(instanceNumbers[slice]), color='k',
                        bbox=dict(boxstyle='square,pad=0.05', fc='white', ec='none'), fontsize=4, weight='bold',
                        transform=ax.transAxes, ha='left', va='top')

            # box with SliceLocations
            if showSliceLocations:
                ax.text(1, 1, str(np.round(sliceLocations[slice], 2)), color='k',
                        bbox=dict(boxstyle='square,pad=0.05', fc='white', ec='none'), fontsize=4, weight='bold',
                        transform=ax.transAxes, ha='right', va='top')

            # plot all mask boundaries for this slice
            for k, roi in enumerate(roiList):
                maskHere = roi['mask'][slice, :, :].astype(int)
                if np.any(maskHere > 0):
                    # tricks to get the boundary of the outside of the mask pixels using contour()
                    ff = 5
                    res = cv2.resize(maskHere, dsize=(maskHere.shape[1] * ff, maskHere.shape[0] * ff),
                                     interpolation=cv2.INTER_NEAREST)
                    cc = axContourDummy.contour(res, levels=[0.5])
                    for pp in cc.allsegs[0]:
                        pp = (pp - (ff - 1) / 2) / ff
                        pp = np.round(pp - 0.5) + 0.5
                        # linewidth scales to the number of plots and the number of pixels in each plot
                        ax.plot(pp[:, 0], pp[:, 1], colors[k],
                                linewidth=150 / np.sqrt(nPlt) / imageVolumeCrop.shape[1])

    # legend goes on last axes that shouldn't have any images in it
    for k, roi in enumerate(roiList):
        ax.plot(0, 0, colors[k], label=roi['ROIName'])
    ax.legend(fontsize=6)

    fPlt.suptitle(titleStr, fontsize=6, x=0.05, horizontalalignment='left')

    if not os.path.exists(os.path.split(outputFileName)[0]):
        os.makedirs(os.path.split(outputFileName)[0])

    fPlt.savefig(outputFileName, orientation='landscape', format=format, dpi=2400)
    print('Written : ' + outputFileName)

    plt.close('all')

    # return filenames of the image files that were displayed
    return [imageVolume['Files'][idx] for idx in idx0]
