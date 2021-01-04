import pydicom
from xml.dom import minidom
import numpy as np
from itertools import compress
from scipy.linalg import circulant
from scipy.ndimage import label
import matplotlib.pyplot as plt
import os
import re
import SimpleITK as sitk
import csv
import yaml
import cv2
import nibabel as nib
from skimage import draw
import warnings
import copy


# add folder to path for radiomicsFeatureExtractorEnhanced
import sys
sys.path.append('/Users/morton/Documents/GitHub/icrpythonradiomics')

from radiomics import featureextractor, setVerbosity
#from radiomicsFeatureExtractorEnhanced import radiomicsFeatureExtractorEnhanced

class radiomicAnalyser:

    def __init__(self, project, assessorFileName, sopInstDict=None, assessorSubtractFileName=None):

        self.projectStr = project["projectStr"]
        self.assessorFileName = assessorFileName
        self.assessorStyle = project["assessorStyle"]
        self.sopInstDict = sopInstDict
        self.outputPath = project["outputPath"]
        self.roiObjectLabelFilter = project["roiObjectLabelFilter"]
        self.paramFileName = project["paramFileName"]
        self.assessorSubtractFileName = assessorSubtractFileName
        self.ImageAnnotationCollection_Description = ' '

        # these are populated by self.loadImageData() because they are taken from the image dicom files
        # PatientName and dcmPatientName should usually be the same, but sometimes we need to change the patientName that
        # gets written into the spreadsheet when there is a typo/mismatch between the dicom info and other data sources.
        # Need to record the dcmPatientName so we can cross-reference back to the source data if necessary
        self.StudyPatientName = ''
        self.dcmPatientName = ''
        self.StudyDate = ''
        self.StudyTime = ''
        self.Manufacturer = ''
        self.ManufacturerModelName = ''
        self.ModalitySpecificParameters = {}

        self.annotationUID = self.__getAnnotationUID()

        print(' ')
        print('Processing : ' + self.assessorFileName)


    ##########################
    # featureKeyPrefixStr can be used to add a prefix to the feature keys in order to manually identify features that have
    # been computed in a particular way.  E.g. when permuting the voxels I use 'permuted_' as a prefix
    def computeRadiomicFeatures(self, binWidthOverRide=None, computeEntropyOfCounts=False, featureKeyPrefixStr=''):

        # get slice gap
        zLoc = sorted([x[2] for x in self.imageData["imagePositionPatient"]])
        if len(zLoc)>1:
            dz = zLoc[1] - zLoc[0]
            if not np.all(abs(np.subtract(zLoc[1:], zLoc[0:-1])) - abs(dz)) < 1e-6:
                raise Exception("Slice spacing is not uniform")
        else:
            # only one slice, so we can't get the slice spacing
            # set to a default value - this won't have any impact as this parameter
            # is only used in the sitk objects to make sure the mask and images are compatible
            dz = float(1.0)

        maskSitk = sitk.GetImageFromArray(self.mask)
        maskSitk.SetOrigin(tuple(self.imageData["imagePositionPatient"][0]))
        maskSitk.SetSpacing((self.imageData["pixelSpacing"][0], self.imageData["pixelSpacing"][1], abs(dz)))
        maskSitk.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, float(np.sign(dz))))

        imageSitk = sitk.GetImageFromArray(self.imageData["imageVolume"])
        imageSitk.SetOrigin(maskSitk.GetOrigin())
        imageSitk.SetSpacing(maskSitk.GetSpacing())
        imageSitk.SetDirection(maskSitk.GetDirection())

        extractor = featureextractor.RadiomicsFeatureExtractor(self.paramFileName)
        #extractor = radiomicsFeatureExtractorEnhanced(self.paramFileName)
        setVerbosity(40)

        if binWidthOverRide is not None:
            extractor.settings["binWidth"] = binWidthOverRide

        segmentNumber = int(1)
        featureVector = extractor.execute(imageSitk, maskSitk, segmentNumber)
        getProbMat = getattr(extractor, "getProbabilityMatrices", None)
        if callable(getProbMat):
            self.probabilityMatrices = extractor.getProbabilityMatrices(imageSitk, maskSitk, segmentNumber)

        if computeEntropyOfCounts:
            for key in self.probabilityMatrices:
                if not "diagnostic" in key:
                    featureVector["entropyOfCounts_joint_" + key] = 0
                    featureVector["entropyOfCounts_dim1_" + key] = 0
                    featureVector["entropyOfCounts_dim2_" + key] = 0
                    prm = self.probabilityMatrices[key]
                    if len(prm.shape)==3:
                        prm = prm[:,:,:,np.newaxis]
                    for n in range(prm.shape[3]):
                        discretizedProbability = np.unique(prm[0, :, :, n], return_inverse=True)[1]
                        probabilityOfCounts = np.histogram(discretizedProbability, bins=range(discretizedProbability.max()+1))[0]
                        probabilityOfCounts = probabilityOfCounts/probabilityOfCounts.sum()
                        featureVector["entropyOfCounts_joint_" + key] += np.sum(-np.log(np.power(probabilityOfCounts,probabilityOfCounts)))
                        #
                        discretizedProbability_1 = np.unique(np.sum(prm[0, :, :, n], axis=0), return_inverse=True)[1]
                        probabilityOfCounts_1 = np.histogram(discretizedProbability_1, bins=range(discretizedProbability_1.max()+1))[0]
                        probabilityOfCounts_1 = probabilityOfCounts_1/probabilityOfCounts_1.sum()
                        featureVector["entropyOfCounts_dim1_" + key] += np.sum(-np.log(np.power(probabilityOfCounts_1,probabilityOfCounts_1)))
                        #
                        discretizedProbability_2 = np.unique(np.sum(prm[0, :, :, n], axis=1), return_inverse=True)[1]
                        probabilityOfCounts_2 = np.histogram(discretizedProbability_2, bins=range(discretizedProbability_2.max()+1))[0]
                        probabilityOfCounts_2 = probabilityOfCounts_2/probabilityOfCounts_2.sum()
                        featureVector["entropyOfCounts_dim2_" + key] += np.sum(-np.log(np.power(probabilityOfCounts_2,probabilityOfCounts_2)))

        # insert or append featureVector just computed
        if hasattr(self, 'featureVector'):
            for key in featureVector.keys():
                # previous extraction will have completed the diagnostics parts of the output, so don't do them again
                if 'diagnostics' not in key:
                    self.featureVector[featureKeyPrefixStr+key] = featureVector[key]
        else:
            self.featureVector = {}
            for key in featureVector.keys():
                # only add featureKeyPrefixStr to actual features, i.e. not to keys called "diagnostics_..." which are a
                # record of the extraction parameters
                if 'diagnostics' in key:
                    self.featureVector[key] = featureVector[key]
                else:
                    self.featureVector[featureKeyPrefixStr+key] = featureVector[key]

        print('Radiomic features computed')


    ##########################
    def setAssessorFileName(self, assessorFileName):
        self.assessorFileName = assessorFileName

    ##########################
    def setParamFileName(self, paramFileName):
        self.paramFileName = paramFileName

    ##########################
    def setOutputPath(self, outputPath):
        self.outputPath = outputPath

    ##########################
    # This function is necessary because the XNAT metadata on subject/session/scan/assessor are only available via the filename
    # These details are included in the .csv file that is written with self.saveResult(), but sometimes they need adjusting
    # to account for anomalies in the data.  For example, in the TracerX study some patients had two tumours, and this
    # is indicated in the clinical data spreadsheet with the subject name, e.g. K114_L and K114_R.  We need a way to reflect
    # this in the .csv output, so by changing the assessor filename manually we can change these fields in the output file
    #
    # FUNCTION DEPRACATED USE editPatientName instead
    def editAssessorFileName(self, newAssessorFileName):
        warnings.warn("Using this method not advised, use editPatientName instead!")
        self.assessorFileName = newAssessorFileName

    ##########################
    # Sometimes typos or other errors creep in and the patientName metadata (in the dicom/annotation/XNAT metadata) does not agree
    # with other data sources (e.g. clinical data spreadsheets).
    # We write out the dcmPatientName (which should always correspond to the actual source data) so we can cross-reference the radiomicAnalyser() outputs to the source data.
    # We also write out the StudyPatientName, which we can edit manually if we need to
    def editStudyPatientName(self, newPatientName):
        self.StudyPatientName = newPatientName


    ##########################
    def createMask(self):
        if self.assessorStyle['type'].lower() == 'aim' and self.assessorStyle['format'].lower() == 'xml':
            self.__createMaskAimXml()
        if self.assessorStyle['type'].lower() == 'seg' and self.assessorStyle['format'].lower() == 'dcm':
            self.__createMaskDcmSeg()
        if self.assessorStyle['type'].lower() == 'seg' and self.assessorStyle['format'].lower() == 'nii':
            self.mask = np.asarray(nib.load(self.assessorFileName).get_data())
        # ... others to come
        #
        # keep a copy of the original mask
        self.maskOriginal = copy.deepcopy(self.mask)
        # run this to make sure self.roiObjectLabelFound is updated
        self.__getReferencedUIDs()

    ##########################
    def removeOutliersFromMask(self, outlierWidth=3):
        if outlierWidth > 0:
            pixels = np.asarray(self.imageData["imageVolume"][self.mask == 1]).reshape(-1, 1)
            mu = np.mean(pixels)
            sg = np.std(pixels)
            imageVolumeOutliers = np.logical_or(self.imageData["imageVolume"]<(mu-outlierWidth*sg),
                                           self.imageData["imageVolume"]>(mu+outlierWidth*sg))
            self.mask[np.logical_and(self.mask==1.0, imageVolumeOutliers)] = 0.0

    ##########################
    # method to enable mask to be altered outside the object, typically to enable experimentation
    def setMask(self, mask):
        if hasattr(self,'imageData'):
            if self.imageData["imageVolume"].shape == mask.shape:
                self.mask = mask
            else:
                raise Exception("Size of externally set mask doesn't match image volume!")
        else:
            raise Exception("Load image data before externally setting mask!")

    ##########################
    # method to enable image to be altered outside the object, typically to enable experimentation
    def setImage(self, image):
        if hasattr(self,'imageData'):
            if self.imageData["imageVolume"].shape == image.shape:
                self.imageData["imageVolume"] = image
            else:
                raise Exception("Size of externally set image data doesn't match image volume!")
        else:
            raise Exception("Load image data before externally setting image!")


    ##########################
    def __getBinParameters(self):
        with open(self.paramFileName) as file:
            params = yaml.full_load(file)
        if 'binWidth' in params['setting']:
            return {'binWidth': params['setting']['binWidth']}
        elif 'binCount' in params['setting']:
            return {'binCount': params['setting']['binCount']}
        else:
            return None


    ##########################
    def __createMaskDcmSeg(self):
        dcmSeg = pydicom.dcmread(self.assessorFileName)

        # read pixel array (bits) using pydicom convenience method that accounts for weird
        # bit unpacking that is required for python
        maskHere = dcmSeg.pixel_array
        # make sure single slice masks have rows/cols/slices along correct dimension
        if len(maskHere.shape) == 2:
            maskHere = np.reshape(maskHere, (1, maskHere.shape[1], maskHere.shape[0]))  # the dimension order needs testing!!

        self.ImageAnnotationCollection_Description = dcmSeg.ContentLabel

        self.mask = np.zeros(self.imageData["imageVolume"].shape)
        maskCount = 0
        for n, funGrpSeq in enumerate(dcmSeg.PerFrameFunctionalGroupsSequence):
            if len(funGrpSeq.DerivationImageSequence) != 1:
                raise Exception("Dicom Seg file has more than one element in DerivationImageSequence!")
            if len(funGrpSeq.DerivationImageSequence[0].SourceImageSequence) != 1:
                raise Exception("Dicom Seg file has more than one element in SourceImageSequence!")
            if len(funGrpSeq.SegmentIdentificationSequence) != 1:
                raise Exception("Dicom Seg file has more than one element in SegmentIdentificationSequence!")
            referencedSegmentNumber = funGrpSeq.SegmentIdentificationSequence[0].ReferencedSegmentNumber
            referencedSegmentLabel = dcmSeg.SegmentSequence._list[referencedSegmentNumber - 1].SegmentLabel
            if (self.roiObjectLabelFilter is not None) and re.match(self.roiObjectLabelFilter, referencedSegmentLabel) is None:
                    continue
            thisSopInstUID = funGrpSeq.DerivationImageSequence[0].SourceImageSequence[0].ReferencedSOPInstanceUID
            sliceIdx = np.where([x == thisSopInstUID for x in self.imageData["sopInstUID"]])[0][0]
            self.mask[sliceIdx, :, :] = np.logical_or(self.mask[sliceIdx, :, :], maskHere[n,:,:])
            maskCount += 1

        if maskCount==0:
            print('\033[1;31;48m    createMask(): No ROI objects matching label "' + self.roiObjectLabelFilter + '" found in assessor!\033[0;30;48m')




##########################
    def __createMaskAimXml(self):
        xDOM = minidom.parse(self.assessorFileName)
        self.ImageAnnotationCollection_Description = xDOM.getElementsByTagName('description').item(0).getAttribute('value')
        self.mask, self.contours = self.__createMaskAimXmlArrayFromContours(xDOM)


##########################
    def removeFromMask(self, objRemove, dilateDiameter=0):
        if isinstance(objRemove, str):
            # if objRemove is a string, then assume is a filename of an AIM xml file
            xDOM = minidom.parse(objRemove)
            self.maskDelete, self.contoursDelete = self.__createMaskAimXmlArrayFromContours(xDOM)
            if dilateDiameter>0:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilateDiameter, dilateDiameter))
                for n in range(self.maskDelete.shape[0]):
                    self.maskDelete[n, :, :] = cv2.dilate(self.maskDelete[n, :, :], kernel)
        elif type(objRemove) is np.ndarray:
            # if objRemove is a numpy array then just use it
            if hasattr(self, 'maskDelete'):
                self.maskDelete = np.logical_or(self.maskDelete.astype(bool), objRemove.astype(bool))
            else:
                self.maskDelete = objRemove.astype(bool)
        self.mask = np.logical_and(self.mask.astype(bool), np.logical_not(self.maskDelete.astype(bool))).astype(float)


##############################
    def __createMaskAimXmlArrayFromContours(self, xDOM, checkRoiLabel=True):
        if xDOM.getElementsByTagName('ImageAnnotation').length != 1:
            raise Exception("AIM file containing more than one ImageAnnotation not supported yet!")
        xImageAnnotation = xDOM.getElementsByTagName('ImageAnnotation').item(0)
        roiLabel = xImageAnnotation.getElementsByTagName('name').item(0).getAttribute('value')
        if (checkRoiLabel and self.roiObjectLabelFilter is not None) and (re.match(self.roiObjectLabelFilter, roiLabel) is None):
            print('\033[1;31;48m    createMask(): No ROI objects matching label "' + self.roiObjectLabelFilter + '" found in assessor!\033[0;30;48m')
            return
        mask = np.zeros(self.imageData["imageVolume"].shape)
        contours = [[] for _ in range(self.imageData["imageVolume"].shape[0])]
        for xMe in xImageAnnotation.getElementsByTagName('MarkupEntity'):
            # find corresponding slice and include this mask - logical_or needed as there may be more than one contour on a slice
            thisSopInstUID = xMe.getElementsByTagName("imageReferenceUid").item(0).getAttribute("root")
            sliceIdx = np.where([x == thisSopInstUID for x in self.imageData["sopInstUID"]])[0][0]

            xScc = xMe.getElementsByTagName('twoDimensionSpatialCoordinateCollection')
            if len(xScc) != 1:
                raise Exception("AIM file has MarkupEntity with more than one twoDimensionSpatialCoordinateCollection not supported yet")
            index = []
            x = []
            y = []
            # Note that scikit-image and matplotlib take the *center* of the top-left pixel to be at (0,0).
            # On the other hand, AIM takes the top-left corner of the top-left pixel to be at (0,0), and gives the
            # polygon co-ordinates as pixel counts (not patient co-ordinates).  Hence the - 0.5 
            for sc in xScc.item(0).getElementsByTagName('TwoDimensionSpatialCoordinate'):
                index.append(int(sc.getElementsByTagName('coordinateIndex').item(0).getAttribute('value')))
                thisX = float(sc.getElementsByTagName('x').item(0).getAttribute('value'))
                thisY = float(sc.getElementsByTagName('y').item(0).getAttribute('value'))
                x.append(thisX - 0.5)
                y.append(thisY - 0.5)

            # according to https://scikit-image.org/docs/stable/api/skimage.draw.html?highlight=skimage%20draw#module-skimage.draw
            # there is a function polygon2mask, but this doesn't seem to be actually present in the library I have.
            # Since draw.polygon2mask is just a wrapper for draw.polygon I'm using the simpler function directly here.
            fill_row_coords, fill_col_coords = draw.polygon(y, x, (mask.shape[1], mask.shape[2]))
            mask[sliceIdx, fill_row_coords, fill_col_coords] = 1.0

            # keep contours so we can display on thumbnail if we need to
            contours[sliceIdx].append({"x":x, "y":y})
        return mask, contours


    ##########################
    def cleanMask(self, minArea=4):
        # For each slice, clean mask by removing edge voxels with only one neighbour with the same value.
        # Also, for each slice, remove regions that are below some area threshold

        def cleanOnce(mask, value):
            # clean voxels that match value

            # initialise and make sure we enter the while loop
            isolated = np.empty_like(mask, dtype='bool')
            isolated[np.unravel_index(0, isolated.shape)] = True

            while np.any(isolated):
                mask0 = mask[:, 1:-1, 1:-1]
                mask1 = (mask[:, 0:-2, 1:-1] == mask0).astype(int)
                mask2 = (mask[:, 1:-1, 0:-2] == mask0).astype(int)
                mask3 = (mask[:, 2:, 1:-1] == mask0).astype(int)
                mask4 = (mask[:, 1:-1, 2:] == mask0).astype(int)
                isolated = np.logical_and(np.equal(mask0, value), (mask1 + mask2 + mask3 + mask4) <= 1)
                mask0[isolated] = not value
                mask[:, 1:-1, 1:-1] = mask0

            return mask

        # clean isolated false voxels, then isolated true voxels
        self.mask = cleanOnce(self.mask, False)
        #self.mask = cleanOnce(self.mask, True)

        # remove regions that are below area threshold
        for n in range(self.mask.shape[0]):
            labelled_mask, num_labels = label(self.mask[n, :, :] == 0)
            # remove small regions
            refined_mask = 1 - self.mask[n, :, :]
            for thisLabel in range(num_labels):
                labelArea = np.sum(refined_mask[labelled_mask == (thisLabel + 1)])
                if labelArea <= minArea:
                    refined_mask[labelled_mask == (thisLabel + 1)] = 0
            self.mask[n, :, :] = 1 - refined_mask


    # function to average over NxN blocks of pixels
    # legacy from old way of computing mask, but have left in all the same
    ##########################
    def __pixelBlockAverage(self, x):
        vr = np.zeros(x.shape[0])
        vr[0:self.fineGrid] = 1
        Vr = circulant(vr)[self.fineGrid - 1::self.fineGrid, :]
        #
        vc = np.zeros(x.shape[1])
        vc[0:self.fineGrid] = 1
        Vc = circulant(vc)[self.fineGrid - 1::self.fineGrid, :]
        return np.dot(Vr, np.dot(x, np.transpose(Vc))) / (self.fineGrid * self.fineGrid)

    ##########################
    def loadImageData(self, fileType=None, fileName = None):

        # direct loading if specified
        if fileType is 'nii':
            imageData = {}
            imageData["imageVolume"] = np.asarray(nib.load(fileName).get_data())
            # for nifti metadata just put in default values for now
            imageData["imagePositionPatient"] = []
            imageData["sopInstUID"] = []
            for n in range(imageData["imageVolume"].shape[2]):
                imageData["imagePositionPatient"].append([0, 0, 2*n]) # this is hard-coded for IBSI digital phantom for now
                imageData["sopInstUID"].append(str(n))
            imageData["imageOrientationPatient"] = [0, 0, 1, 0, 1, 0] # default
            imageData["pixelSpacing"] = [2, 2] # this is hard-coded for IBSI digital phantom for now
            imageData["windowCenter"] = 0
            imageData["windowWidth"] = 100
            self.imageData = imageData
            # some other metadata from the assessor that needs to be present
            self.ReferencedSeriesUID = ''
            self.ImageAnnotationCollection_Description = ''
            self.roiObjectLabelFound = ''
            return

        refUID = self.__getReferencedUIDs()

        if len(refUID)==0:
            print('\033[1;31;48m    loadImageData(): No ROI objects matching label "' + self.roiObjectLabelFilter + '" found in assessor!\033[0;30;48m')
            return

        sopInstUID = []
        imSlice = []
        imagePositionPatient = []

        # get list of unique referencedSOPInstanceUIDs
        refSopInstUIDs = list(set([x['ReferencedSOPInstanceUID'] for x in refUID]))

        # get study date and time so they can go into the csv output
        dcm = pydicom.dcmread(self.sopInstDict[refUID[0]['ReferencedSOPInstanceUID']])
        self.StudyPatientName = dcm.PatientName
        self.dcmPatientName = dcm.PatientName # assume dcmPatientName and StudyPatientName are the same at this point.  We may manually edit StudyPatientName using editStudyPatientName if we need to
        self.StudyDate = dcm.StudyDate
        self.StudyTime = dcm.StudyTime
        self.Manufacturer = dcm.Manufacturer
        self.ManufacturerModelName = dcm.ManufacturerModelName

        # get any Modality-specific parameters that might be useful for checking for confounders
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

        elif dcm.SOPClassUID == sopClassUid_MRImageStorage:
            # not implemented yet
            parameterList = []
        else:
            parameterList = []
        for parameter in parameterList:
            if hasattr(dcm, parameter):
                self.ModalitySpecificParameters[parameter] = getattr(dcm, parameter)
            else:
                self.ModalitySpecificParameters[parameter] = ''

        for refSopInstUID in refSopInstUIDs:
            dcm = pydicom.dcmread(self.sopInstDict[refSopInstUID])

            # check references match as expected
            if dcm.SeriesInstanceUID != self.ReferencedSeriesUID:
                raise Exception("SopInstance dictionary error: SeriesInstanceUID found in dicom file does not match reference in annotation file!")
            if dcm.SOPInstanceUID != refSopInstUID:
                raise Exception("SopInstance dictionary error: SOPInstanceUID found in dicom file does not match dictionary!")

            # check image is axial
            axialArr = [1, 0, 0, 0, 1, 0]
            axialTol = 1e-6
            axialErr = [np.abs(np.abs(float(x)) - y) > axialTol for x, y in zip(dcm.ImageOrientationPatient, axialArr)]
            if any(axialErr):
                raise Exception("Non-axial image referenced by annotation file - not supported yet!")

            # grab important parts of dicom
            sopInstUID.append(dcm.SOPInstanceUID)
            imSlice.append(dcm.RescaleSlope * dcm.pixel_array + dcm.RescaleIntercept)
            imagePositionPatient.append([float(x) for x in dcm.ImagePositionPatient])

        imageData = {}
        # assuming these are the same for all referenced SOPInstances
        imageData["imageOrientationPatient"] = [float(x) for x in dcm.ImageOrientationPatient]
        imageData["pixelSpacing"] = [float(x) for x in dcm.PixelSpacing]
        if type(dcm.WindowCenter) is pydicom.multival.MultiValue:
            imageData["windowCenter"] = dcm.WindowCenter[0]
        else:
            imageData["windowCenter"] = dcm.WindowCenter
        if type(dcm.WindowWidth) is pydicom.multival.MultiValue:
            imageData["windowWidth"] = dcm.WindowWidth[0]
        else:
            imageData["windowWidth"] = dcm.WindowWidth

        # sort on slice location and store items in self
        sliceLocation = [x[2] for x in imagePositionPatient]
        imageData["sopInstUID"] = [x for _, x in sorted(zip(sliceLocation, sopInstUID))]
        imSlice = [x for _, x in sorted(zip(sliceLocation, imSlice))]
        imageData["imageVolume"] = np.asarray(imSlice)
        imageData["imagePositionPatient"] = [x for _, x in sorted(zip(sliceLocation, imagePositionPatient))]
        self.imageData = imageData
        # we might do some permutations on the voxel locations, so keep a copy of the original image data, in case
        # we need to reset back
        self.imageDataOriginal = copy.deepcopy(self.imageData)


    ##########################
    def permuteVoxels(self, fixedSeed=True):
        if fixedSeed:
            np.random.seed(seed=42)
        # get pixel values inside mask
        voxels = self.imageData["imageVolume"][np.where(self.mask == 1)]
        idxShuffle = np.random.permutation(len(voxels))
        self.imageData["imageVolume"][np.where(self.mask == 1)] = voxels[idxShuffle]


    ##########################
    def __getReferencedUIDs(self):
        if self.assessorStyle['format'].lower() == 'dcm':
            references = self.__getReferencedUIDsDicom()
        elif self.assessorStyle['format'].lower() == 'xml':
            references = self.__getReferencedUIDsAimXml()
        # select segments matching segmentLabel input
        if self.roiObjectLabelFilter is not None:
            indToKeep = [re.match(self.roiObjectLabelFilter, x["label"]) is not None for x in references]
            if not any(indToKeep):
                references = []
            else:
                references = list(compress(references, indToKeep))
        self.roiObjectLabelFound = list(set([x["label"] for x in references]))
        print('         ' + str(self.roiObjectLabelFound))
        return references


    ##########################
    def __getReferencedUIDsDicom(self):
        dcm = pydicom.dcmread(self.assessorFileName)
        if dcm.SOPClassUID == '1.2.840.10008.5.1.4.1.1.481.3':
            # Dicom RT
            # check only one item in each level of hierarchy going down to ReferencedSeriesUID
            if len(dcm.ReferencedFrameOfReferenceSequence) != 1:
                raise Exception("DICOM RT file referencing more than one frame of reference not supported!")
            rfors = dcm.ReferencedFrameOfReferenceSequence[0]

            if len(rfors.RTReferencedStudySequence) != 1:
                raise Exception("DICOM RT file referencing more than one study not supported!")
            rtrss = rfors.RTReferencedStudySequence[0]

            self.ReferencedSeriesUID = rtrss.RTReferencedSeriesSequence[0].SeriesInstanceUID

            if len(rtrss.RTReferencedSeriesSequence) != 1:
                raise Exception("DICOM RT file referencing more than one series not supported!")

            roiNameDict = {}
            for ssr in dcm.StructureSetROISequence:
                roiNameDict[ssr.ROINumber] = ssr.ROIName

            annotationObjectList = []
            for rcs in dcm.ROIContourSequence:
                label = roiNameDict[rcs.ReferencedROINumber]
                for cs in rcs.ContourSequence:
                    if len(cs.ContourImageSequence) != 1:
                        raise Exception(
                            "DICOM RT file containing (individual) contour that references more than one image not supported!")
                    annotationObjectList.append(
                        {"ReferencedSOPInstanceUID": cs.ContourImageSequence[0].ReferencedSOPInstanceUID,
                         "label": label})
            return annotationObjectList

        elif dcm.SOPClassUID == '1.2.840.10008.5.1.4.1.1.66.4':
            # Segmentation Storage
            # check only one referenced series
            if len(dcm.ReferencedSeriesSequence) != 1:
                raise Exception("DICOM SEG file referencing more than one series not supported!")
            # get list of all ReferencedSopInstanceUIDs
            annotationObjectList = []
            for n, funGrpSeq in enumerate(dcm.PerFrameFunctionalGroupsSequence):
                if len(funGrpSeq.DerivationImageSequence) != 1:
                    raise Exception("Dicom Seg file has more than one element in DerivationImageSequence!")
                if len(funGrpSeq.DerivationImageSequence[0].SourceImageSequence) != 1:
                    raise Exception("Dicom Seg file has more than one element in SourceImageSequence!")
                if len(funGrpSeq.SegmentIdentificationSequence) != 1:
                    raise Exception("Dicom Seg file has more than one element in SegmentIdentificationSequence!")
                referencedSegmentNumber = funGrpSeq.SegmentIdentificationSequence[0].ReferencedSegmentNumber
                label = dcm.SegmentSequence._list[referencedSegmentNumber - 1].SegmentLabel
                annotationObjectList.append(
                    {"ReferencedSOPInstanceUID": funGrpSeq.DerivationImageSequence[0].SourceImageSequence[
                        0].ReferencedSOPInstanceUID,
                     "label": label})
            self.ReferencedSeriesUID = dcm.ReferencedSeriesSequence[0].SeriesInstanceUID
            return annotationObjectList


    ##########################
    def __getReferencedUIDsAimXml(self):
        xDOM = minidom.parse(self.assessorFileName)
        xImSer = xDOM.getElementsByTagName('imageSeries')
        seriesUIDs = np.unique([x.getElementsByTagName('instanceUid').item(0).getAttribute('root') for x in xImSer])

        if len(seriesUIDs) != 1:
            raise Exception('AIM file referencing more than one series not supported!')

        self.ReferencedSeriesUID = str(seriesUIDs[0])

        annotationObjectList = []
        for xImAnn in xDOM.getElementsByTagName('ImageAnnotation'):
            label = xImAnn.getElementsByTagName('name').item(0).getAttribute('value')
            for me in xImAnn.getElementsByTagName('MarkupEntity'):
                annotationObjectList.append({"ReferencedSOPInstanceUID": me.getElementsByTagName(
                    'imageReferenceUid').item(0).getAttribute('root'),
                                             "label": label})
        return annotationObjectList


    ##########################
    def __getAnnotationUID(self):
        if self.assessorStyle['format'].lower() == 'dcm':
            dcm = pydicom.dcmread(self.assessorFileName)
            annotationUID = dcm.SOPInstanceUID
        elif self.assessorStyle['format'].lower() == 'xml':
            xDOM = minidom.parse(self.assessorFileName)
            annotationUID = xDOM.getElementsByTagName('ImageAnnotation').item(0).getElementsByTagName('uniqueIdentifier').item(0).getAttribute('root')
        elif self.assessorStyle['format'].lower() == 'nii':
            annotationUID = ''
        return annotationUID


    ##########################
    def saveThumbnail(self, fileStr = '', vmin=None, vmax=None, showContours=False, showMaskBoundary=True, titleStrExtra='', showMaskHolesWithNewColour=False, axisLimits=None):

        def findMaskEdges(mask):

            if mask.ndim == 2:
                mask.shape = [1, mask.shape[0], mask.shape[1]]
                squash = True
            else:
                squash = False

            mask0 = mask[:, 1:-1, 1:-1]
            mask1 = (mask[:, 0:-2, 1:-1] == mask0).astype(int)
            mask2 = (mask[:, 1:-1, 0:-2] == mask0).astype(int)
            mask3 = (mask[:, 2:, 1:-1] == mask0).astype(int)
            mask4 = (mask[:, 1:-1, 2:] == mask0).astype(int)

            edgeMask = mask == 1
            edgeMask[:, 1:-1, 1:-1] = (mask1 + mask2 + mask3 + mask4) < 4
            edgeMask = np.logical_and(mask == 1, edgeMask)

            if squash:
                edgeMask.shape = [edgeMask.shape[1], edgeMask.shape[2]]
                mask.shape = [mask.shape[1], mask.shape[2]]

            return edgeMask

        # crop images to within 20 pixels of the max extent of the mask in all slices
        pad = 20
        maskRows = np.sum(np.sum(self.mask, axis=0) > 0, axis=0) > 0
        maskRows[pad:] = np.logical_or(maskRows[pad:], maskRows[0:-pad])
        maskRows[0:-pad] = np.logical_or(maskRows[0:-pad], maskRows[pad:])
        maskCols = np.sum(np.sum(self.mask, axis=0) > 0, axis=1) > 0
        maskCols[pad:] = np.logical_or(maskCols[pad:], maskCols[0:-pad])
        maskCols[0:-pad] = np.logical_or(maskCols[0:-pad], maskCols[pad:])

        # put slices next to each other in a single row
        maskMontage = self.mask[0, :, :][maskCols, :][:, maskRows]
        imageMontage = self.imageData["imageVolume"][0, :, :][maskCols, :][:, maskRows]
        barWidth = 5
        maskBar = np.zeros((np.sum(maskCols), barWidth))
        imageBar = 500*np.ones((np.sum(maskCols), barWidth))
        for n in range(self.mask.shape[0] - 1):
            maskMontage = np.concatenate((maskMontage, maskBar, self.mask[n + 1, :, :][maskCols, :][:, maskRows]), axis=1)
            imageMontage = np.concatenate((imageMontage, imageBar, self.imageData["imageVolume"][n + 1, :, :][maskCols, :][:, maskRows]), axis=1)

        # get grayscale limits
        if vmin is None:
            vmin = self.imageData["windowCenter"] - self.imageData["windowWidth"]/2
        if vmax is None:
            vmax = self.imageData["windowCenter"] + self.imageData["windowWidth"]/2

        nPlt = 2 + self.mask.shape[0] # extra for a histogram
        pltRows = int(np.round(np.sqrt(2*nPlt/3)))
        pltCols = int(np.ceil(nPlt/pltRows))
        plt.clf()
        fPlt, axarr = plt.subplots(pltRows, pltCols, gridspec_kw={'wspace':0, 'hspace':0})

        linewidth = 0.2
        if np.sum(self.mask)==0:
            minX = 0
            maxX = self.mask.shape[2]
            minY = 0
            maxY = self.mask.shape[1]
        else:
            dim1 = np.where(np.sum(self.mask, axis=(0, 2)) > 0)
            dim2 = np.where(np.sum(self.mask, axis=(0, 1)) > 0)
            minX = min(dim2)
            maxX = max(dim2)
            minY = min(dim1)
            maxY = max(dim1)
        # for contours in self.contours:
        #     for contour in contours:
        #         minX = np.min([minX, np.min(contour["x"])])
        #         maxX = np.max([maxX, np.max(contour["x"])])
        #         minY = np.min([minY, np.min(contour["y"])])
        #         maxY = np.max([maxY, np.max(contour["y"])])
        # make spans at least minSize pixels, then add padSize
        minSize = 40
        padSize = 10
        midX = 0.5*(minX + maxX)
        minX = np.min([midX - minSize/2, minX])-padSize
        maxX = np.max([midX + minSize/2, maxX])+padSize
        midY = 0.5*(minY + maxY)
        minY = np.min([midY - minSize/2, minY])-padSize
        maxY = np.max([midY + minSize/2, maxY])+padSize

        if axisLimits is not None:
            minX = axisLimits["minX"]
            maxX = axisLimits["maxX"]
            minY = axisLimits["minY"]
            maxY = axisLimits["maxY"]

        # get current axis limits and include in the variable we will output from the function
        out = {"axisLimits": {"minX": minX, "maxX": maxX, "minY": minY, "maxY": maxY}}

        for n, ax in enumerate(fPlt.axes):
            if n<(nPlt-2):
                ax.imshow(self.imageData["imageVolume"][n,:,:], cmap='gray', vmin=vmin, vmax=vmax, interpolation='nearest')
                if showContours:
                    contours = self.contours[n]
                    for contour in contours:
                        ax.plot([x for x in contour["x"]], [y for y in contour["y"]], 'c', linewidth=linewidth)
                    contoursDelete = self.contoursDelete[n]
                    for contourDelete in contoursDelete:
                        ax.plot([x for x in contourDelete["x"]], [y for y in contourDelete["y"]], 'r', linewidth=linewidth)
                maskHere = self.mask[n,:,:]
                if showMaskBoundary:
                    idx = np.where(np.logical_and(maskHere[:, 0:-1]==0.0, maskHere[:, 1:]==1.0))
                    ax.plot(np.asarray((idx[1]+0.5, idx[1]+0.5)), np.asarray((idx[0]-0.5,idx[0]+0.5)), 'r', linewidth=linewidth)
                    idx = np.where(np.logical_and(maskHere[:, 0:-1]==1.0, maskHere[:, 1:]==0.0))
                    ax.plot(np.asarray((idx[1]+0.5, idx[1]+0.5)), np.asarray((idx[0]-0.5,idx[0]+0.5)), 'r', linewidth=linewidth)
                    idx = np.where(np.logical_and(maskHere[0:-1,:]==0.0, maskHere[1:,:]==1.0))
                    ax.plot(np.asarray((idx[1]-0.5, idx[1]+0.5)), np.asarray((idx[0]+0.5,idx[0]+0.5)), 'r', linewidth=linewidth)
                    idx = np.where(np.logical_and(maskHere[0:-1,:]==1.0, maskHere[1:,:]==0.0))
                    ax.plot(np.asarray((idx[1]-0.5, idx[1]+0.5)), np.asarray((idx[0]+0.5,idx[0]+0.5)), 'r', linewidth=linewidth)
                    # overplot holes if there are present
                    if (showMaskHolesWithNewColour and hasattr(self, 'maskDelete')):
                        maskHere = np.logical_and(self.maskOriginal[n, :, :].astype(bool), self.maskDelete[n, :, :].astype(bool)).astype(float)
                        idx = np.where(np.logical_and(maskHere[:, 0:-1] == 0.0, maskHere[:, 1:] == 1.0))
                        ax.plot(np.asarray((idx[1] + 0.5, idx[1] + 0.5)), np.asarray((idx[0] - 0.5, idx[0] + 0.5)), 'b',
                                linewidth=linewidth)
                        idx = np.where(np.logical_and(maskHere[:, 0:-1] == 1.0, maskHere[:, 1:] == 0.0))
                        ax.plot(np.asarray((idx[1] + 0.5, idx[1] + 0.5)), np.asarray((idx[0] - 0.5, idx[0] + 0.5)), 'b',
                                linewidth=linewidth)
                        idx = np.where(np.logical_and(maskHere[0:-1, :] == 0.0, maskHere[1:, :] == 1.0))
                        ax.plot(np.asarray((idx[1] - 0.5, idx[1] + 0.5)), np.asarray((idx[0] + 0.5, idx[0] + 0.5)), 'b',
                                linewidth=linewidth)
                        idx = np.where(np.logical_and(maskHere[0:-1, :] == 1.0, maskHere[1:, :] == 0.0))
                        ax.plot(np.asarray((idx[1] - 0.5, idx[1] + 0.5)), np.asarray((idx[0] + 0.5, idx[0] + 0.5)), 'b',
                                linewidth=linewidth)
                ax.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)
                ax.set_xlim(minX, maxX)
                ax.set_ylim(maxY, minY) # to flip y-axis
            elif n==nPlt-1:
                if np.sum(self.mask)>0:
                    yRef = np.asarray(self.imageData["imageVolume"][self.mask == 1]).reshape(-1, 1)
                    binParams = self.__getBinParameters()
                    if 'binWidth' in binParams:
                        bins = np.arange(vmin, vmax, binParams['binWidth'])
                    elif 'binCount' in binParams:
                        bins = np.linspace(min(yRef), max(yRef), num=binParams['binCount']).squeeze()
                    ax.hist(yRef, bins, density=True, histtype='stepfilled')
            else:
                ax.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['right'].set_visible(False)

        if len(self.roiObjectLabelFound)==1:
            roiObjectLabel = self.roiObjectLabelFound[0]
        else:
            roiObjectLabel = str(self.roiObjectLabelFound)
        titleStr = os.path.split(self.assessorFileName)[1].replace('__II__', '  ').split('.')[0] + '  ' + roiObjectLabel + '  '  + self.ImageAnnotationCollection_Description
        plt.gcf().suptitle(titleStr + ' ' + titleStrExtra, fontsize=7)

        fullPath = os.path.join(self.outputPath, 'roiThumbnails', 'subjects')
        if not os.path.exists(fullPath):
            os.makedirs(fullPath)
        fileStr = 'roiThumbnail__' + os.path.split(self.assessorFileName)[1].split('.')[0] + fileStr + '.pdf'
        out["fileName"] = os.path.join(fullPath, fileStr)
        plt.gcf().savefig(out["fileName"],  papertype='a4', orientation='landscape', format='pdf', dpi=1200)
        print('Thumbnail saved '+out["fileName"])
        plt.close()
        return out

    ##########################
    def saveResult(self, writeMode='w', includeHeader=True):

        headers = []
        row = []

        # add XNAT info so we can convert to DICOM SR later
        headers.append("source_XNAT_project")
        row.append(self.projectStr)

        headers.append("StudyPatientName")
        row.append(self.StudyPatientName)

        fileParts = os.path.split(self.assessorFileName)[1].split("__II__")

        headers.append("source_XNAT_session")
        row.append(fileParts[1])

        headers.append("source_XNAT_scan")
        row.append(fileParts[2])

        headers.append("source_XNAT_assessor")
        row.append(fileParts[3].split('.')[0])

        headers.append("source_DCM_PatientName")
        row.append(self.dcmPatientName)

        headers.append("source_DCM_SeriesInstanceUID")
        row.append(self.ReferencedSeriesUID)

        headers.append("source_DCM_StudyDate")
        row.append(self.StudyDate)

        headers.append("source_DCM_StudyTime")
        row.append(self.StudyTime)

        # mark some columns with string "QueryConfounder" then we can use this later in an automatic
        # check to filter these columns and check if there are any unwanted correlations/clusterings
        # with these parameters
        headers.append("source_DCM_Manufacturer_QueryConfounder")
        row.append(self.Manufacturer)

        headers.append("source_DCM_ManufacturerModelName_QueryConfounder")
        row.append(self.ManufacturerModelName)

        for key, value in self.ModalitySpecificParameters.items():
            headers.append("source_DCM_"+key+"_QueryConfounder")
            row.append(value)

        headers.append("source_annotationUID")
        row.append(self.annotationUID)

        headers.append("source_ImageAnnotationCollectionDescription")
        row.append(self.ImageAnnotationCollection_Description)

        headers.append("source_roiLabel")
        if len(self.roiObjectLabelFound)==1:
            row.append(self.roiObjectLabelFound[0])
        else:
            row.append(self.roiObjectLabelFound)

        headers.extend(list(self.featureVector.keys()))
        for h in list(self.featureVector.keys()):
            row.append(self.featureVector.get(h, "N/A"))

        fullPath = os.path.join(self.outputPath, 'radiomicFeatures', 'subjects')
        if not os.path.exists(fullPath):
            os.makedirs(fullPath)
        fileStr = 'radiomicFeatures__' + os.path.split(self.assessorFileName)[1].split('.')[0] + '.csv'
        outputName = os.path.join(fullPath, fileStr)


        with open(outputName, writeMode) as fo:
            writer = csv.writer(fo, lineterminator='\n')
            if includeHeader:
                writer.writerow(headers)
            writer.writerow(row)

        print("Results file saved")
        return outputName


    ##########################
    def saveProbabilityMatrices(self, imageType='original'):

        fig = plt.figure()
        columns = 7
        rows = 5
        # show GLCM
        for n in range(self.probabilityMatrices[imageType + "_glcm"].shape[3]):
            fig.add_subplot(rows, columns, n+1)
            if n==0:
                if len(self.roiObjectLabelFound) == 1:
                    roiObjectLabel = self.roiObjectLabelFound[0]
                else:
                    roiObjectLabel = str(self.roiObjectLabelFound)
                plt.title(os.path.split(self.assessorFileName)[1].replace('__II__', '  ').split('.')[0] + '  ' + roiObjectLabel, fontsize=8, fontdict = {'horizontalalignment': 'left'})

            if np.mod(n,7)==0:
                plt.ylabel('GLCM')
            plt.imshow(self.probabilityMatrices[imageType + "_glcm"][0,:,:,n])
            plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)

        # show GLRLM
        for n in range(self.probabilityMatrices[imageType + "_glrlm"].shape[3]):
            fig.add_subplot(rows, columns, n+15)
            if np.mod(n,7)==0:
                plt.ylabel('GLRLM')
            plt.imshow(self.probabilityMatrices[imageType + "_glrlm"][0, :, :, n], aspect='auto')
            plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)

        fig.add_subplot(rows, 3, 13)
        plt.imshow(self.probabilityMatrices[imageType + "_glszm"][0, :, :], aspect='auto')
        plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
        plt.ylabel('GLSZM')

        fig.add_subplot(rows, 3, 14)
        plt.imshow(self.probabilityMatrices[imageType + "_gldm"][0, :, :], aspect='auto')
        plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
        plt.ylabel('GLDM')

        fig.add_subplot(rows, 3, 15)
        plt.imshow(self.probabilityMatrices[imageType + "_ngtdm"][0, :, :], aspect='auto')
        plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
        plt.ylabel('NGTDM')

        fullPath = os.path.join(self.outputPath, 'probabilityMatrices', 'subjects')
        if not os.path.exists(fullPath):
            os.makedirs(fullPath)
        fileStr = 'probabilityMatrices_' + imageType + '__' + os.path.split(self.assessorFileName)[1].split('.')[0] + '.pdf'
        outputName = os.path.join(fullPath, fileStr)
        plt.gcf().savefig(outputName, papertype='a4', orientation='landscape', format='pdf', dpi=1200)
        print('probabilityMatrices saved ' + outputName)
        plt.close()
        return outputName