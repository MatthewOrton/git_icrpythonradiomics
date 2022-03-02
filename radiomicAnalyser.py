import pydicom
from xml.dom import minidom
import numpy as np
from itertools import compress
from scipy.linalg import circulant
from scipy.ndimage import label
from scipy.io import savemat
import matplotlib.pyplot as plt
import os
import re
import SimpleITK as sitk
import csv
import yaml
import cv2
import nibabel as nib
from skimage import draw
from skimage.segmentation import flood_fill
import warnings
import copy
from scipy.stats import norm
import inspect
import nrrd
import uuid

# add folder to path for radiomicsFeatureExtractorEnhanced and the mixture model module
import sys
sys.path.append('/Users/morton/Documents/GitHub/icrpythonradiomics')

from radiomics import featureextractor, setVerbosity
from mixture_cdf import GaussianMixtureCdf
from mixture_cdf import BayesianGaussianMixtureCdf

class radiomicAnalyser:

    def __init__(self, project, assessorFileName, sopInstDict=None, extraDictionaries=None, assessorSubtractFileName=None, axialTol=1e-6, roiShift=[0,0]):

        self.projectStr = project["projectStr"]
        self.assessorFileName = assessorFileName
        self.assessorStyle = project["assessorStyle"]
        self.sopInstDict = sopInstDict
        self.extraDictionaries = extraDictionaries
        self.outputPath = project["outputPath"]
        self.roiObjectLabelFilter = project["roiObjectLabelFilter"]
        self.paramFileName = project["paramFileName"]
        self.assessorSubtractFileName = assessorSubtractFileName
        self.ImageAnnotationCollection_Description = ' '
        self.axialTol = axialTol
        self.roiShift = roiShift

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

    def getModuleFileLocation(self):
        return __file__

    ##########################
    # featureKeyPrefixStr can be used to add a prefix to the feature keys in order to manually identify features that have
    # been computed in a particular way.  E.g. when permuting the voxels I use 'permuted_' as a prefix
    def computeRadiomicFeatures(self, binWidthOverRide=None, binCountOverRide=None, binEdgesOverRide=None, gldm_aOverRide=None, distancesOverRide=None, resampledPixelSpacing=None, computeEntropyOfCounts=False, featureKeyPrefixStr=''):

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
        setVerbosity(40)

        if binEdgesOverRide is not None:
            extractor.settings["binEdges"] = binEdgesOverRide
            extractor.settings["binWidth"] = None
            extractor.settings["binCount"] = None
            self.binEdgesOverRide = binEdgesOverRide

        if binWidthOverRide is not None:
            extractor.settings["binEdges"] = None
            extractor.settings["binWidth"] = binWidthOverRide
            extractor.settings["binCount"] = None
            self.binWidthOverRide = binWidthOverRide

        if binCountOverRide is not None:
            extractor.settings["binEdges"] = None
            extractor.settings["binWidth"] = None
            extractor.settings["binCount"] = binCountOverRide
            self.binCountOverRide = binCountOverRide

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
        featureVector, self.probabilityMatrices, self.filteredImages, self.quantizedImages = extractor.execute(imageSitk, maskSitk, segmentNumber)

        if 'firstorder' in extractor.enabledFeatures:
            pixels = self.imageData['imageVolume'][self.mask == 1]
            # pyradiomics already computes the 10th, 50th and 90th centiles, so skip these here
            qs = np.hstack((0.05, np.linspace(0.15, 0.45, 7), np.linspace(0.55, 0.85, 7), 0.95))
            for q in qs:
                featureVector['original_histogram_' + str((q * 100).round().astype(int)) + 'Percentile'] = np.quantile(pixels, q)

        # if extraction with separate directions has been used, then
        # post-process the corresponding features so each value is returned as a new feature with a feature name that includes
        # the direction (angle) of the GLCM, and normalised using the P_glcm_0 and P_glcm_inf values
        #
        # get array of directions (angles)
        featNameAngles = [s for s in list(featureVector.keys()) if "SeparateDirections_Angles" in s]
        if featNameAngles:
            # just use the first as the others should be the same
            angles = featureVector[featNameAngles[0]]

            # convert angles array to list of strings that we can use as postfixes for the feature names
            anglesStr = [('__' + '{:+d}'.format(s[0]) + '{:+d}'.format(s[1]) + '{:+d}'.format(s[2])) for s in angles]
            anglesStr = [s.replace('-', 'n') for s in anglesStr]
            anglesStr = [s.replace('+', 'p') for s in anglesStr]
            # special for the P_glcm_inf one that is returned with angles = [-100, -100, -100]
            anglesStr = [s.replace('__n100n100n100','__inf') for s in anglesStr]

            # get names for features with separate directions, and remove the Angles features
            featNamesSepDir = [s for s in list(featureVector.keys()) if "SeparateDirections" in s]
            featNamesSepDir = [s for s in featNamesSepDir if "SeparateDirections_Angles" not in s]

            # get indices of the actual feature values, i.e. not the _000 or _inf values that are used for
            # normalisation
            idxValues = list(range(len(angles)))
            idxValues.remove(anglesStr.index('__p0p0p0'))
            idxValues.remove(anglesStr.index('__inf'))

            # normalise the main values for each feature using the P_glcm_0 and P_glcm_inf values, and replace the
            # original item in featureVector (which is an array) with a collection of items whose names indicate which
            # direction they were calculated with
            for feat in featNamesSepDir:
                thisFeature = featureVector[feat]
                f0 = thisFeature[anglesStr.index('__p0p0p0')]
                fInf = thisFeature[anglesStr.index('__inf')]
                if True: # apply normalization
                    thisFeature[idxValues] = (thisFeature[idxValues] - fInf)/(f0 - fInf)
                # remove original item from featureVector (this item is an array, which we can't output to a .csv very easily)
                featureVector.pop(feat)
                # put back thisFeature using the original name and the anglesStr postFix
                for n, value in enumerate(thisFeature):
                    thisName = feat+anglesStr[n]
                    featureVector[thisName] = value
            # remove the angles features as they are no longer needed
            for fa in featNameAngles:
                featureVector.pop(fa)

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
        if not hasattr(self, 'featureVector'):
            self.featureVector = {}

        for key in featureVector.keys():
            self.featureVector[featureKeyPrefixStr+key] = featureVector[key]

        print('Radiomic features computed')

        return np.asarray(self.imageData["imageVolume"][self.mask == 1]).reshape(-1, 1)

    def histogramEqualise(self, flavour='Bayesian', nComponents=6, debugPlot=False, targetDistribution='normal'):

        if targetDistribution is None:
            return

        # if mask is input then get the reference data from y using the mask
        dataMask = np.asarray(self.imageData["imageVolume"][self.mask == 1]).reshape(-1, 1)

        # skip some data so we never fit more than 10000 elements
        skip = np.ceil(dataMask.shape[0]/10000).astype(int)
        dataMaskTrain = dataMask[0::skip,:]

        # Fit Gaussian mixture model to reference data dataMask.
        if flavour is 'Bayesian':
            # With Bayesian model the prior acts to regularise the mixture model, i.e. only
            # a "sensible" number of components will have non-negligible
            # weights, so the mixture model will be parsimonious.  The
            # mixture components themselves also tend to cluster together
            # to further encourage parsimony.
            gmm = BayesianGaussianMixtureCdf(n_components=nComponents, random_state=10, max_iter=10000).fit(dataMaskTrain)
        else:
            # this is standard EM Gaussian mixture model, and is included for evaluation purposes, but
            # not recommended for main analysis
            gmm = GaussianMixtureCdf(n_components=nComponents, random_state=10, max_iter=10000).fit(dataMaskTrain)

        if debugPlot:
            x = np.linspace(np.min(dataMask), np.max(dataMask), 1000)
            bins = np.array(range(int(x[0]), int(x[-1] + 2))) - 0.5

            logprob_standard = gmm.score_samples(x.reshape(-1, 1))
            fig, ax = plt.subplots(2)
            ax[0].plot(x, np.exp(logprob_standard))
            ax[0].hist(dataMask, bins, density=True, histtype='stepfilled')
            ax[0].plot(x, np.exp(logprob_standard))
            ax[0].hist(dataMask, bins, density=True, histtype='stepfilled')

        if targetDistribution == 'normal':
            # Do histogram equalisation of fitted mixture model onto a truncated standard normal distribution.  The truncation is included
            # in order to ensure the bin edges used by the radiomics package are consistent across subjects (when the BinCount is specified).
            BW = 4  # BW = 4 implies a 4-sigma truncation
            loc = 0
            scale = 1
            out = {}
            cc = norm.cdf(-BW, loc=loc, scale=scale)
            dataAll = self.imageData["imageVolume"].reshape(-1, 1)

            # equalise data from mask and clip to +/- BW
            dataMask = norm.ppf(gmm.cdf(dataMask) * (1 - 2 * cc) + cc, loc=loc, scale=scale)
            dataMask[dataMask < -BW] = -BW
            dataMask[dataMask > BW] = BW
            # at this stage there may not be any voxels equal to -BW or BW, but we are using this only to find the min and max values inside the mask

            # equalise all data and clip to +/- BW
            dataAll = norm.ppf(gmm.cdf(dataAll) * (1 - 2 * cc) + cc, loc=loc, scale=scale)
            dataAll[dataAll < -BW] = -BW
            dataAll[dataAll > BW] = BW
            dataAll = dataAll.reshape(self.imageData["imageVolume"].shape)

            # Find voxels inside the mask that are equal to the max and set one of them to BW.  Do similar for the min.
            idxMax = np.where(np.bitwise_and(dataAll == np.max(dataMask), self.mask == 1))
            idxMin = np.where(np.bitwise_and(dataAll == np.min(dataMask), self.mask == 1))
            dataAll[idxMax[0][0], idxMax[1][0], idxMax[2][0]] = BW
            dataAll[idxMin[0][0], idxMin[1][0], idxMin[2][0]] = -BW

            if debugPlot:
                ax[1].hist(dataMask, np.linspace(-BW,BW,16), density=True, histtype='stepfilled')
                plt.show()


        if targetDistribution == 'uniform':
            dataAll = self.imageData["imageVolume"].reshape(-1, 1)
            # equalise all data and clip to +/- BW
            dataAll = gmm.cdf(dataAll)
            dataAll = dataAll.reshape(self.imageData["imageVolume"].shape)

            if debugPlot:
                dataMask = gmm.cdf(dataMask)
                ax[1].hist(dataMask, np.linspace(0,1,16), density=True, histtype='stepfilled')
                plt.show()


        self.imageData["imageVolume"] = dataAll

        return gmm

    def saveImageAndMaskToMatlab(self):
        imData = self.imageData["imageVolume"]
        mask = self.mask
        matlabFolderName = os.path.join(self.outputPath,'matlabData')
        if not os.path.exists(matlabFolderName):
            os.mkdir(matlabFolderName)
        matlabFileName = os.path.join(matlabFolderName,os.path.split(self.assessorFileName)[1].split('.')[0]+'.mat')
        savemat(matlabFileName, {'imData':imData, 'mask':mask})
        print('Written matlab file to : '+matlabFileName)


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
        if self.assessorStyle['type'].lower() == 'aim' and self.assessorStyle['format'].lower() == 'dcm':
            self.__createMaskDcmRts()
        if self.assessorStyle['type'].lower() == 'seg' and self.assessorStyle['format'].lower() == 'nii':
            self.mask = np.asarray(nib.load(self.assessorFileName).get_data())
        if self.assessorStyle['type'].lower() == 'seg' and self.assessorStyle['format'].lower() == 'nrrd':
            self.mask, _ = nrrd.read(self.assessorFileName)
            self.mask = np.moveaxis(self.mask, -1, 0)

        # ... others to come
        #
        # keep a copy of the original mask
        self.maskOriginal = copy.deepcopy(self.mask)
        # run this to make sure self.roiObjectLabelFound is updated
        self.__getReferencedUIDs()
        print('ROI label  : ' + str(self.roiObjectLabelFound))

    ##########################
    def removeEmptySlices(self):
        sliceUse = np.sum(self.mask, axis=(1,2))>0
        self.mask = self.mask[sliceUse,:,:]
        self.maskOriginal = self.maskOriginal[sliceUse,:,:]
        #
        self.imageData["imageVolume"] = self.imageData["imageVolume"][sliceUse, :, :]
        self.imageData["imageInstanceNumber"] = [i for i,j in zip(self.imageData["imageInstanceNumber"],sliceUse) if j]
        self.imageData["sopInstUID"] = [i for i,j in zip(self.imageData["sopInstUID"],sliceUse) if j]
        self.imageData["imagePositionPatient"] = [i for i,j in zip(self.imageData["imagePositionPatient"],sliceUse) if j]
        #
        self.imageDataOriginal["imageVolume"] = self.imageDataOriginal["imageVolume"][sliceUse, :, :]
        self.imageDataOriginal["imageInstanceNumber"] = [i for i,j in zip(self.imageDataOriginal["imageInstanceNumber"],sliceUse) if j]
        self.imageDataOriginal["sopInstUID"] = [i for i,j in zip(self.imageDataOriginal["sopInstUID"],sliceUse) if j]
        self.imageDataOriginal["imagePositionPatient"] = [i for i,j in zip(self.imageDataOriginal["imagePositionPatient"],sliceUse) if j]


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
    def clipImage(self, **kwargs): # stdFactor=float, quantiles=[float, float], clip=[float, float]

        if len(kwargs)>1:
            raise Exception("Only one keyword allowed")
        keyword = list(kwargs.keys())[0]
        value = list(kwargs.values())[0]

        pixels = np.asarray(self.imageData["imageVolume"][self.mask == 1]).reshape(-1, 1)

        if keyword == 'stdFactor':
            mu = np.mean(pixels)
            sg = np.std(pixels)
            low  = mu - value * sg
            high = mu + value * sg

        if keyword == 'quantiles':
            low  = np.quantile(pixels, np.min(value))
            high = np.quantile(pixels, np.max(value))

        if keyword == 'clip':
            low  = np.min(value)
            high = np.max(value)

        highVoxels = np.logical_and(self.imageData["imageVolume"] > high, self.mask == 1.0)
        self.imageData["imageVolume"][highVoxels] = high

        lowVoxels = np.logical_and(self.imageData["imageVolume"] < low, self.mask == 1.0)
        self.imageData["imageVolume"][lowVoxels] = low


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
            if hasattr(self,'binWidthOverRide'):
                return {'binWidth': self.binWidthOverRide}
            else:
                return {'binWidth': params['setting']['binWidth']}
        elif 'binCount' in params['setting']:
            if hasattr(self,'binCountOverRide'):
                return {'binCount': self.binCountOverRide}
            else:
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
            maskHere = np.reshape(maskHere, (1, maskHere.shape[0], maskHere.shape[1]))  # the dimension order needs testing!!

        if 'SeriesDescription' in dcmSeg:
            self.ImageAnnotationCollection_Description = dcmSeg.SeriesDescription
        else:
            self.ImageAnnotationCollection_Description = dcmSeg.ContentLabel

        # find ReferencedSegmentNumber for the ROI we have already found
        referencedSegmentNumber = [x.SegmentNumber for x in dcmSeg.SegmentSequence if x.SegmentLabel == self.roiObjectLabelFound]
        if len(referencedSegmentNumber) != 1:
            raise Exception("More than one segment with same name found in Dicom Seg file!")
        referencedSegmentNumber = referencedSegmentNumber[0]

        self.mask = np.zeros(self.imageData["imageVolume"].shape)
        maskCount = 0
        if 'DerivationImageSequence' in dcmSeg.PerFrameFunctionalGroupsSequence[0] and 'SegmentIdentificationSequence' in dcmSeg.PerFrameFunctionalGroupsSequence[0]:
            for n, funGrpSeq in enumerate(dcmSeg.PerFrameFunctionalGroupsSequence):
                if funGrpSeq.SegmentIdentificationSequence[0].ReferencedSegmentNumber == referencedSegmentNumber:
                    thisSopInstUID = funGrpSeq.DerivationImageSequence[0].SourceImageSequence[0].ReferencedSOPInstanceUID
                    sliceIdx = np.where([x == thisSopInstUID for x in self.imageData["sopInstUID"]])[0][0]
                    self.mask[sliceIdx, :, :] = np.logical_or(self.mask[sliceIdx, :, :], maskHere[n,:,:])
                    maskCount += 1

        # the TCIA nsclc-radiogenomics data have messed up the labels and stored the ReferencedSOPInstanceUIDs in a strange place
        # ignore label and put mask slices in as necessary
        elif 'ReferencedInstanceSequence' in dcmSeg.ReferencedSeriesSequence[0]:
            for n, refSerItem in enumerate(dcmSeg.ReferencedSeriesSequence[0].ReferencedInstanceSequence):
                thisSopInstUID = refSerItem.ReferencedSOPInstanceUID
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
    def __createMaskDcmRts(self):
        rts = pydicom.dcmread(self.assessorFileName)
        self.ImageAnnotationCollection_Description =  rts.StructureSetLabel # name of attribute in self is just to fit with AIM file naming.
        self.mask, self.contours = self.__createMaskDcmRtsArrayFromContours(rts)

    ##########################
    def __createMaskDcmRtsArrayFromContours(self, rts, checkRoiLabel=True):

        if len(rts.ROIContourSequence)>1 and not checkRoiLabel:
            raise Exception("DICOM RT file containing more than one ROI not currently supported!")

        roiNumber = None
        for ssr in rts.StructureSetROISequence:
            if ssr.ROIName == self.roiObjectLabelFound:
                roiNumber = ssr.ROINumber
                break
        if roiNumber is None:
            raise Exception("Cannot find roiObjectLabel in DICOM RT file!")

        for rcs in rts.ROIContourSequence:
            if int(rcs.ReferencedROINumber) == int(roiNumber):
                thisContourSequence = rcs.ContourSequence

        mask = np.zeros(self.imageData["imageVolume"].shape)
        contours = [[] for _ in range(self.imageData["imageVolume"].shape[0])]
        for cs in thisContourSequence:
            if len(cs.ContourImageSequence) != 1:
                raise Exception("DICOM RT file containing (individual) contour that references more than one image not supported!")

            referencedSOPInstanceUID = cs.ContourImageSequence[0].ReferencedSOPInstanceUID
            coords = np.array([float(x) for x in cs.ContourData])
            polygonPatient = coords.reshape((int(len(coords) / 3), 3))

            sliceIdx = self.imageData["sopInstUID"].index(referencedSOPInstanceUID)
            origin = self.imageData["imagePositionPatient"][sliceIdx]
            spacing = self.imageData["pixelSpacing"]
            xNorm = self.imageData["imageOrientationPatient"][0:3]
            yNorm = self.imageData["imageOrientationPatient"][3:6]
            x = np.dot(polygonPatient - origin, xNorm) / spacing[0]
            y = np.dot(polygonPatient - origin, yNorm) / spacing[1]

            # according to https://scikit-image.org/docs/stable/api/skimage.draw.html?highlight=skimage%20draw#module-skimage.draw
            # there is a function polygon2mask, but this doesn't seem to be actually present in the library I have.
            # Since draw.polygon2mask is just a wrapper for draw.polygon I'm using the simpler function directly here.
            fill_row_coords, fill_col_coords = draw.polygon(y+self.roiShift[1], x+self.roiShift[0], (mask.shape[1], mask.shape[2]))
            mask[sliceIdx, fill_row_coords, fill_col_coords] = 1.0

            # keep contours so we can display on thumbnail if we need to
            contours[sliceIdx].append({"x":x, "y":y})

        return mask, contours


    ##########################
    def removeFromMask(self, objRemove, dilateDiameter=0):
        if isinstance(objRemove, str):
            # if objRemove is a string, then assume is a filename of an AIM xml file
            xDOM = minidom.parse(objRemove)
            self.maskDelete, self.contoursDelete = self.__createMaskAimXmlArrayFromContours(xDOM)
            if dilateDiameter > 0:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilateDiameter, dilateDiameter))
                for n in range(self.maskDelete.shape[0]):
                    self.maskDelete[n, :, :] = cv2.dilate(self.maskDelete[n, :, :], kernel)
        elif type(objRemove) is np.ndarray:
            # if objRemove is a numpy array then just use it
            if hasattr(self, 'maskDelete'):
                self.maskDelete = np.logical_or(self.maskDelete.astype(bool), objRemove.astype(bool))
            else:
                self.maskDelete = objRemove.astype(bool)
        self.mask = np.logical_and(self.mask.astype(bool), np.logical_not(self.maskDelete.astype(bool))).astype(
            float)

    ##########################
    def cleanMaskEdge(self):
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
        self.mask = cleanOnce(self.mask, True)

    def removeSmallMaskRegions(self, maxArea=10, maskIn=None):
        # organise inputs so that we can use this operation either on the mask variable in the object, or operate on an external variable
        if maskIn is None:
            mask = self.mask
        else:
            mask = maskIn

        # Remove any isolated regions that are below a certain area, except if there is only one region on that slice

        for n in range(mask.shape[0]):
            labelled_mask, num_labels = label(mask[n, :, :] == 1)
            if num_labels==1:
                continue
            # remove small regions
            refined_mask = mask[n, :, :]
            for thisLabel in range(num_labels):
                labelArea = np.sum(refined_mask[labelled_mask == (thisLabel + 1)])
                if labelArea <= maxArea:
                    refined_mask[labelled_mask == (thisLabel + 1)] = 0
            mask[n, :, :] = refined_mask

        if maskIn is None:
            self.mask = mask
        else:
            return mask


    def fillMaskHoles(self, maxArea=float('inf'), maskIn=None):
        # organise inputs so that we can use this operation either on the mask variable in the object, or operate on an external variable
        if maskIn is None:
            mask = self.mask
        else:
            mask = maskIn

        # Remove holes that are below area threshold.  Default maxArea set so that
        # *all* holes will be filled, unless specified otherwise.

        for n in range(mask.shape[0]):
            maskHoles = 1-flood_fill(mask[n,:,:],(0,0),1)
            labelled_mask, num_labels = label(maskHoles == 1)
            # remove small holes
            for thisLabel in range(num_labels):
                maskHere = labelled_mask == (thisLabel+1)
                if np.sum(maskHere) > maxArea:
                    maskHoles[maskHere] = 0
            mask[n, :, :] = np.logical_xor(mask[n, :, :]==1, maskHoles==1)

        if maskIn is None:
            self.mask = mask
        else:
            return mask

    def selectSlicesMaxAreaMovingAverage(self, width=3):
        if width is not None and self.mask.shape[0]>width:
            idx = np.argmax(np.convolve(np.sum(self.mask, axis=(1, 2)), np.ones(width), mode='valid'))
            self.mask = self.mask[idx:idx + width, :, :]
            self.imageData["imageVolume"] = self.imageData["imageVolume"][idx:idx + width, :, :]

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
    def loadImageData(self, fileType=None, fileName=None, includeExtraTopAndBottomSlices=False):

        # direct loading if specified
        if fileType == 'nii':
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

            # we might do some permutations on the voxel locations, so keep a copy of the original image data, in case
            # we need to reset back
            self.imageDataOriginal = copy.deepcopy(self.imageData)
            return

        if fileType == 'nrrd':
            imageData = {}
            imageData["imageVolume"], fileheader = nrrd.read(fileName)
            # z axis needs to be first
            imageData['imageVolume'] = np.moveaxis(imageData['imageVolume'], -1, 0)
            spaceDirections = np.asarray(fileheader['space directions'])
            # assuming axial data, so slice thickness is in z co-ordinate
            sliceThickness = spaceDirections[2,2]
            imageData["imagePositionPatient"] = []
            imageData["sopInstUID"] = []
            imageData["imageInstanceNumber"] = []
            for n in range(imageData["imageVolume"].shape[2]):
                imageData["imagePositionPatient"].append([0, 0, n*sliceThickness])
                imageData["sopInstUID"].append(str(n))
                imageData["imageInstanceNumber"].append(n)
            imageData["imageOrientationPatient"] = [0, 0, 1, 0, 1, 0] # default
            imageData["pixelSpacing"] = [spaceDirections[0,0], spaceDirections[1,1]] # this is hard-coded for IBSI digital phantom for now
            imageData["windowCenter"] = 300 # default for now
            imageData["windowWidth"] = 600
            self.imageData = imageData
            # some other metadata from the assessor that needs to be present
            self.ReferencedSeriesUID = ''
            self.ImageAnnotationCollection_Description = ''
            self.roiObjectLabelFound = ''

            # we might do some permutations on the voxel locations, so keep a copy of the original image data, in case
            # we need to reset back
            self.imageDataOriginal = copy.deepcopy(self.imageData)
            return

        refUID = self.__getReferencedUIDs()

        if len(refUID)==0:
            print('\033[1;31;48m    loadImageData(): No ROI objects matching label "' + self.roiObjectLabelFilter + '" found in assessor!\033[0;30;48m')
            return

        sopInstUID = []
        imSlice = []
        imagePositionPatient = []
        imagePositionPatientNormal = [] # this is imagePositionPatient projected onto the image normal vector and is used to sort the slices
        imageInstanceNumber = []

        # get list of unique referencedSOPInstanceUIDs
        refSopInstUIDs = list(set([x['ReferencedSOPInstanceUID'] for x in refUID]))

        # get study date and time so they can go into the csv output
        dcm = pydicom.dcmread(self.sopInstDict[refUID[0]['ReferencedSOPInstanceUID']])
        self.StudyPatientName = str(dcm.PatientName)
        self.dcmPatientName = str(dcm.PatientName) # assume dcmPatientName and StudyPatientName are the same at this point.  We may manually edit StudyPatientName using editStudyPatientName if we need to
        self.StudyDate = dcm.StudyDate
        self.StudyTime = dcm.StudyTime
        if 'Manufacturer' in dcm:
            self.Manufacturer = dcm.Manufacturer
        else:
            self.Manufacturer = 'unknown'
        if 'ManufacturerModelName' in dcm:
            self.ManufacturerModelName = dcm.ManufacturerModelName
        else:
            self.ManufacturerModelName = 'unknown'


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

        if includeExtraTopAndBottomSlices:
            if not hasattr(self, 'extraDictionaries'):
                raise Exception('To include extra top and bottom slices you need to input an instanceNumberDict')
            else:
                instanceDict = self.extraDictionaries['instanceNumDict']
                sopInst2instanceNumberDict = self.extraDictionaries['sopInst2instanceNumberDict']
            # find extra slices from the InstanceNumbers in the referenced SopInstances
            instanceNumbers = list()
            for refSopInstUID in refSopInstUIDs:
                instanceNumbers.append(int(sopInst2instanceNumberDict[refSopInstUID]))
            if (min(instanceNumbers)-1) in instanceDict.keys():
                extraSopInstance = instanceDict[min(instanceNumbers)-1]['SOPInstanceUID']
                refSopInstUIDs.append(extraSopInstance)
            if (min(instanceNumbers)-2) in instanceDict.keys():
                extraSopInstance = instanceDict[min(instanceNumbers)-2]['SOPInstanceUID']
                refSopInstUIDs.append(extraSopInstance)
            if (max(instanceNumbers)+1) in instanceDict.keys():
                extraSopInstance = instanceDict[max(instanceNumbers)+1]['SOPInstanceUID']
                refSopInstUIDs.append(extraSopInstance)
            if (max(instanceNumbers)+2) in instanceDict.keys():
                extraSopInstance = instanceDict[max(instanceNumbers)+2]['SOPInstanceUID']
                refSopInstUIDs.append(extraSopInstance)


        for refSopInstUID in refSopInstUIDs:
            dcm = pydicom.dcmread(self.sopInstDict[refSopInstUID])

            # check references match as expected
            if dcm.SeriesInstanceUID != self.ReferencedSeriesUID:
                raise Exception("SopInstance dictionary error: SeriesInstanceUID found in dicom file does not match reference in annotation file!")
            if dcm.SOPInstanceUID != refSopInstUID:
                raise Exception("SopInstance dictionary error: SOPInstanceUID found in dicom file does not match dictionary!")

            # check image is axial
            # axialArr = [1, 0, 0, 0, 1, 0]
            # axialTol = self.axialTol
            # axialErrorValue = [np.abs(np.abs(float(x)) - y) for x, y in zip(dcm.ImageOrientationPatient, axialArr)]
            # axialErr = [np.abs(x) > axialTol for x in axialErrorValue]
            # if any(axialErr):
            #     print(axialErrorValue)
            #     raise Exception("Non-axial image referenced by annotation file - not supported yet!")

            imPos = np.array([float(x) for x in dcm.ImagePositionPatient])
            imagePositionPatient.append(imPos)

            imOri = np.array([float(x) for x in dcm.ImageOrientationPatient])
            imagePositionPatientNormal.append(np.dot(imPos, np.cross(imOri[0:3], imOri[3:])))
            imageInstanceNumber.append(dcm.InstanceNumber)

            # grab important parts of dicom
            sopInstUID.append(dcm.SOPInstanceUID)
            if 'RescaleSlope' in dcm:
                RescaleSlope = dcm.RescaleSlope
            else:
                RescaleSlope = 1.0
            if 'RescaleIntercept' in dcm:
                RescaleIntercept = dcm.RescaleIntercept
            else:
                RescaleIntercept = 0.0
            imSlice.append(RescaleSlope * dcm.pixel_array + RescaleIntercept)

        imageData = {}
        # assuming these are the same for all referenced SOPInstances
        imageData["imageOrientationPatient"] = [float(x) for x in dcm.ImageOrientationPatient]
        imageData["pixelSpacing"] = [float(x) for x in dcm.PixelSpacing]
        if hasattr(dcm,'WindowCenter'):
            if type(dcm.WindowCenter) is pydicom.multival.MultiValue:
                imageData["windowCenter"] = dcm.WindowCenter[0]
            else:
                imageData["windowCenter"] = dcm.WindowCenter
        else:
            imageData["windowCenter"] = 0
        if hasattr(dcm, 'WindowWidth'):
            if type(dcm.WindowWidth) is pydicom.multival.MultiValue:
                imageData["windowWidth"] = dcm.WindowWidth[0]
            else:
                imageData["windowWidth"] = dcm.WindowWidth
        else:
            imageData["windowWidth"] = 100

        # sort on slice location and store items in self
        reverseFlag = True
        imageData["sopInstUID"] = [x for _, x in sorted(zip(imagePositionPatientNormal, sopInstUID), reverse=reverseFlag)]
        imSlice = [x for _, x in sorted(zip(imagePositionPatientNormal, imSlice), reverse=reverseFlag)]
        imageData["imageVolume"] = np.asarray(imSlice)
        imageData["imagePositionPatient"] = [x for _, x in sorted(zip(imagePositionPatientNormal, imagePositionPatient), reverse=reverseFlag)]
        imageData["imageInstanceNumber"] = [x for _, x in sorted(zip(imagePositionPatientNormal, imageInstanceNumber), reverse=reverseFlag)]
        imagePositionPatientNormal.sort()
        imageData["imagePositionPatientNormal"] = imagePositionPatientNormal

        print('Slice thickness = ' + str(dcm.SliceThickness))
        print('Slice spacing = ' + str(imagePositionPatientNormal[1]-imagePositionPatientNormal[0]))

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
        self.imageData["imageVolume"][self.mask == 1] = voxels[idxShuffle]


    ##########################
    def __getReferencedUIDs(self):
        if self.assessorStyle['format'].lower() == 'dcm':
            references = self.__getReferencedUIDsDicom()
        elif self.assessorStyle['format'].lower() == 'xml':
            references = self.__getReferencedUIDsAimXml()
        else:
            references = []
        # select segments matching segmentLabel input
        if self.roiObjectLabelFilter is not None:
            indToKeep = [re.match(self.roiObjectLabelFilter, x["label"]) is not None for x in references]
            if not any(indToKeep):
                references = []
            else:
                references = list(compress(references, indToKeep))
        roiObjectLabelFound = list(set([x["label"] for x in references]))
        if len(roiObjectLabelFound)>1:
            raise Exception("More than one roiObject selected from DICOM RT file - only one currently supported. Use more specific roiObjectLabelFilter string.")
        if len(roiObjectLabelFound)==0:
            self.roiObjectLabelFound = ''
        else:
            self.roiObjectLabelFound = roiObjectLabelFound[0]
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
            if 'DerivationImageSequence' in dcm.PerFrameFunctionalGroupsSequence[0] and 'SegmentIdentificationSequence' in dcm.PerFrameFunctionalGroupsSequence[0]:
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
                        {"ReferencedSOPInstanceUID": funGrpSeq.DerivationImageSequence[0].SourceImageSequence[0].ReferencedSOPInstanceUID,
                         "label": label})

            # the TCIA nsclc-radiogenomics data have messed up the labels and stored the ReferencedSOPInstanceUIDs in a strange place
            # just put unknown for segment label
            elif 'ReferencedInstanceSequence' in dcm.ReferencedSeriesSequence[0]:
                for refSerItem in dcm.ReferencedSeriesSequence[0].ReferencedInstanceSequence:
                    annotationObjectList.append(
                        {"ReferencedSOPInstanceUID": refSerItem.ReferencedSOPInstanceUID,
                         "label": 'unknown'})

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
        elif self.assessorStyle['format'].lower() == 'nrrd':
            annotationUID = ''
        return annotationUID


    ##########################
    def saveThumbnail(self, quantizedImageType=None, fileStr = '', scaling=None, vmin=None, vmax=None, showContours=False, padSize=10, minSize=40, showMaskBoundary=True, titleStrExtra='', showMaskHolesWithNewColour=False, axisLimits=None, bins=None, pathStr='roiThumbnails', showHistogram=True, titleFontSize=7, linewidth=0.2):

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

        # get image array
        if quantizedImageType is None:
            imArr = self.imageData["imageVolume"]
            maskArr = self.mask
        else:
            imArr = self.quantizedImages[quantizedImageType] # uncomment to use this one and comment out the next line
            #imArr = self.filteredImages["original"]["image"]
            maskArr = self.filteredImages["original"]["mask"]

        # crop images to within pad pixels of the max extent of the mask in all slices
        pad = 20
        maskRows = np.sum(np.sum(maskArr, axis=0) > 0, axis=0) > 0
        maskRows[pad:] = np.logical_or(maskRows[pad:], maskRows[0:-pad])
        maskRows[0:-pad] = np.logical_or(maskRows[0:-pad], maskRows[pad:])
        maskCols = np.sum(np.sum(maskArr, axis=0) > 0, axis=1) > 0
        maskCols[pad:] = np.logical_or(maskCols[pad:], maskCols[0:-pad])
        maskCols[0:-pad] = np.logical_or(maskCols[0:-pad], maskCols[pad:])

        # put slices next to each other in a single row
        maskMontage = maskArr[0, :, :][maskCols, :][:, maskRows]
        imageMontage = imArr[0, :, :][maskCols, :][:, maskRows]
        barWidth = 5
        maskBar = np.zeros((np.sum(maskCols), barWidth))
        imageBar = 500*np.ones((np.sum(maskCols), barWidth))
        for n in range(maskArr.shape[0] - 1):
            maskMontage = np.concatenate((maskMontage, maskBar, maskArr[n + 1, :, :][maskCols, :][:, maskRows]), axis=1)
            imageMontage = np.concatenate((imageMontage, imageBar, imArr[n + 1, :, :][maskCols, :][:, maskRows]), axis=1)

        # get grayscale limits
        yRef = np.asarray(imArr[maskArr == 1]).reshape(-1, 1)
        if vmin is None and vmax is None and scaling is None:
            vmin = self.imageData["windowCenter"] - self.imageData["windowWidth"]/2
            vmax = self.imageData["windowCenter"] + self.imageData["windowWidth"]/2
        elif scaling=='ROI':
            vmin = np.quantile(yRef, 0.01)
            vmax = np.quantile(yRef,0.99)

        if bins is None and quantizedImageType is not None:
            bins =  np.unique(yRef)

        nPlt = 2 + maskArr.shape[0] # extra for a histogram
        pltRows = int(np.round(np.sqrt(2*nPlt/3)))
        pltCols = int(np.ceil(nPlt/pltRows))
        plt.clf()
        fPlt, axarr = plt.subplots(pltRows, pltCols, gridspec_kw={'wspace':0, 'hspace':0})

        if np.sum(maskArr)==0:
            minX = 0
            maxX = maskArr.shape[2]
            minY = 0
            maxY = maskArr.shape[1]
        else:
            dim1 = np.where(np.sum(maskArr, axis=(0, 2)) > 0)
            dim2 = np.where(np.sum(maskArr, axis=(0, 1)) > 0)
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
        import progressbar

        with progressbar.ProgressBar(max_value=len(fPlt.axes)) as bar:
            for n, ax in enumerate(fPlt.axes):
                bar.update(n)
                if n<(nPlt-2):
                    imDisp = imArr[n,:,:]
                    #nxx = np.round(minX+0.15*(maxX-minX)).astype(int)
                    #nyy = np.round(minY+0.2*(maxY-minY)).astype(int)
                    #imDisp[0:nxx, 0:nyy] = vmin
                    ax.imshow(imDisp, cmap='gray', vmin=vmin, vmax=vmax, interpolation='nearest')
                    # ax.text(minX+0.02*(maxX-minX), minY+0.02*(maxY-minY), str(self.imageData["imageInstanceNumber"][n]), color='w', fontsize=3, ha='left', va='top', backgroundcolor='k', transform=ax.transAxes) #clip_on=True)
                    ax.text(0, 1, str(self.imageData["imageInstanceNumber"][n]), color='k', bbox=dict(boxstyle='square,pad=0.05', fc='white', ec='none'), fontsize=4, weight='bold', transform=ax.transAxes, ha='left', va='top')
                    if showContours:
                        contours = self.contours[n]
                        dx = self.roiShift[0]   # xnatCollaborations viewer has a bug that results in a 1 pixel shift, so roiShift = [-1, -1] will fix thiscan input
                        dy = self.roiShift[1]
                        for contour in contours:
                            # make sure will be closed
                            xPlot = [x+dx for x in contour["x"]]+[contour['x'][0]+dx]
                            yPlot = [y+dy for y in contour["y"]]+[contour['y'][0]+dy]
                            ax.plot(xPlot, yPlot, 'b', linewidth=linewidth)
                        if hasattr(self,'contoursDelete'):
                            contoursDelete = self.contoursDelete[n]
                            for contourDelete in contoursDelete:
                                xPlot = [x+dx for x in contourDelete["x"]] + [contourDelete['x'][0]+dx]
                                yPlot = [y+dy for y in contourDelete["y"]] + [contourDelete['y'][0]+dy]
                                ax.plot(xPlot, yPlot, 'r', linewidth=linewidth)
                    maskHere = maskArr[n,:,:]
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
                            holeColor = 'c'
                            maskHere = np.logical_and(self.maskOriginal[n, :, :].astype(bool), self.maskDelete[n, :, :].astype(bool)).astype(float)
                            idx = np.where(np.logical_and(maskHere[:, 0:-1] == 0.0, maskHere[:, 1:] == 1.0))
                            ax.plot(np.asarray((idx[1] + 0.5, idx[1] + 0.5)), np.asarray((idx[0] - 0.5, idx[0] + 0.5)), holeColor,
                                    linewidth=linewidth)
                            idx = np.where(np.logical_and(maskHere[:, 0:-1] == 1.0, maskHere[:, 1:] == 0.0))
                            ax.plot(np.asarray((idx[1] + 0.5, idx[1] + 0.5)), np.asarray((idx[0] - 0.5, idx[0] + 0.5)), holeColor,
                                    linewidth=linewidth)
                            idx = np.where(np.logical_and(maskHere[0:-1, :] == 0.0, maskHere[1:, :] == 1.0))
                            ax.plot(np.asarray((idx[1] - 0.5, idx[1] + 0.5)), np.asarray((idx[0] + 0.5, idx[0] + 0.5)), holeColor,
                                    linewidth=linewidth)
                            idx = np.where(np.logical_and(maskHere[0:-1, :] == 1.0, maskHere[1:, :] == 0.0))
                            ax.plot(np.asarray((idx[1] - 0.5, idx[1] + 0.5)), np.asarray((idx[0] + 0.5, idx[0] + 0.5)), holeColor,
                                    linewidth=linewidth)
                    ax.xaxis.set_visible(False)
                    ax.yaxis.set_visible(False)
                    ax.set_xlim(minX, maxX)
                    ax.set_ylim(maxY, minY) # to flip y-axis
                elif n==(pltRows*pltCols-1):
                    if showHistogram and np.sum(maskArr)>0:
                        if bins is None:
                            binParams = self.__getBinParameters()
                            if 'binWidth' in binParams:
                                bins = np.arange(vmin, vmax, binParams['binWidth'])
                            elif 'binCount' in binParams:
                                bins = np.linspace(min(yRef), max(yRef), num=binParams['binCount']).squeeze()
                        ax.hist(yRef, bins, density=True, histtype='stepfilled')
                        ax.set_xlim([bins[0], bins[-1]])
                    else:
                        ax.xaxis.set_visible(False)
                        ax.yaxis.set_visible(False)
                        ax.spines['top'].set_visible(False)
                        ax.spines['bottom'].set_visible(False)
                        ax.spines['left'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                else:
                    ax.xaxis.set_visible(False)
                    ax.yaxis.set_visible(False)
                    ax.spines['top'].set_visible(False)
                    ax.spines['bottom'].set_visible(False)
                    ax.spines['left'].set_visible(False)
                    ax.spines['right'].set_visible(False)

        titleStr = os.path.split(self.assessorFileName)[1].split('.')[0]

        # titleStr = os.path.split(self.assessorFileName)[1].replace('__II__', '  ').split('.')[0] + '  ' + self.roiObjectLabelFound # + '  '  + self.ImageAnnotationCollection_Description
        # titleStr = '\033[1m\033[4m' + titleStr.replace('__II__', '\033[0m\033[0m  ', 1) + '\033[1m\033[4m  ' + self.roiObjectLabelFound + '\033[0m\033[0m'

        titleStr = os.path.split(self.assessorFileName)[1].split('.')[0]
        titleStr = titleStr.replace(self.dcmPatientName, self.StudyPatientName)
        titleStr = titleStr.split('__II__')
        titleStr = r'$\bf{' + titleStr[0].replace('_', '\_') + '}$   ' + r'$\bf{' + self.roiObjectLabelFound.replace('_', '\_') + '}$  ' + '  '.join(titleStr[1:])
        #    os.path.split(self.assessorFileName)[1].replace('__II__', '}$  ').split('.')[0] + '  ' + self.roiObjectLabelFound # + '  '  + self.ImageAnnotationCollection_Description

        #titleStr = r'$\bf{' + titleStr.replace('__II__', '  ', 1) + '  ' + r'$\bf{' + self.roiObjectLabelFound.replace('_','\_') + '}$'

        plt.gcf().suptitle(titleStr + ' ' + titleStrExtra, fontsize=titleFontSize, x = 0.05, horizontalalignment='left')

        fullPath = os.path.join(self.outputPath, pathStr, 'subjects')
        if not os.path.exists(fullPath):
            os.makedirs(fullPath)
        fileStr = 'roiThumbnail__' + os.path.split(self.assessorFileName)[1].split('.')[0] + '_' + self.roiObjectLabelFound + fileStr + '.pdf'
        fileStr = fileStr.replace(self.dcmPatientName, self.StudyPatientName)

        out["fileName"] = os.path.join(fullPath, fileStr)
        plt.gcf().savefig(out["fileName"],  orientation='landscape', format='pdf', dpi=1200) #papertype='a4',
        print('Thumbnail saved '+out["fileName"])
        plt.close()

        out["vmin"] = vmin
        out["vmax"] = vmax

        return out

    ##########################
    def saveResult(self, writeMode='w', includeHeader=True, fileSubscript=''):

        headers = []
        row = []

        # add XNAT info so we can convert to DICOM SR later
        headers.append("source_XNAT_project")
        row.append(self.projectStr)

        headers.append("StudyPatientName")
        row.append(self.StudyPatientName)

        fileName = os.path.split(self.assessorFileName)[1]
        fileParts = fileName.split("__II__")

        # cheat for cases that haven't been downloaded from XNAT, and therefore have particular filename structure
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
            # special case that needs modification
            # diagnostics_Configuration_Settings element may contain a list of binEdges, and this will usually be too long to fit into the csv output
            # modify the binEdges variable so that it becomes a 3 element list with [start, stop, step]
            if 'diagnostics_Configuration_Settings' in h:
                thisFeature = self.featureVector.get(h)
                if 'binEdges' in thisFeature.keys() and thisFeature['binEdges'] is not None:
                    thisFeature['binEdges'] = [thisFeature['binEdges'][0], thisFeature['binEdges'][-1], thisFeature['binEdges'][1]-thisFeature['binEdges'][0]]
                    self.featureVector[h] = thisFeature
            row.append(self.featureVector.get(h, "N/A"))

        fullPath = os.path.join(self.outputPath, 'radiomicFeatures', 'subjects')
        if not os.path.exists(fullPath):
            os.makedirs(fullPath)
        fileStr = 'radiomicFeatures__' + os.path.split(self.assessorFileName)[1].split('.')[0] + fileSubscript + '.csv'

        fileStr = fileStr.replace(self.dcmPatientName, self.StudyPatientName)

        outputName = os.path.join(fullPath, fileStr)

        if os.path.exists(outputName):
            outputName = outputName.replace('.csv', '_'+str(uuid.uuid1())+'.csv')
            print('\033[1;31;48m' + '_' * 50)
            print('File name clash!! Added UID for uniqueness')
            print('_' * 50 + '\033[0;30;48m')

        with open(outputName, writeMode) as fo:
            writer = csv.writer(fo, lineterminator='\n')
            if includeHeader:
                writer.writerow(headers)
            writer.writerow(row)

        print("Results file saved")
        return outputName


    ##########################
    def saveProbabilityMatrices(self, imageType='original', mainTitle=True, mainTitleStr = '', showHistogram=True, fileStr='', supressGLCMdiagonal=False):

        fig = plt.figure()
        columns = 7
        rows = 5
        fontsize=7
        # show GLCM
        for n in range(self.probabilityMatrices[imageType + "_glcm"].shape[3]):
            fig.add_subplot(rows, columns, n+1)
            if n==0 and mainTitle:
                titleStr = os.path.split(self.assessorFileName)[1]
                titleStr = titleStr.replace('__II__', '  ').split('.')[0]
                titleStr = titleStr.replace(self.dcmPatientName, self.StudyPatientName)
                plt.title(titleStr + '  ' + self.roiObjectLabelFound + '  ' + imageType + '  ' + mainTitleStr, fontsize=7, fontdict = {'horizontalalignment': 'left'})

            if np.mod(n,7)==0:
                plt.ylabel('GLCM', fontsize=fontsize)
            pMat = self.probabilityMatrices[imageType + "_glcm"][0,:,:,n]
            if supressGLCMdiagonal:
                pMat *= (1 - np.eye(pMat.shape[0]))
            plt.imshow(pMat)
            plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)

        # show GLRLM
        for n in range(self.probabilityMatrices[imageType + "_glrlm"].shape[3]):
            fig.add_subplot(rows, columns, n+15)
            if np.mod(n,7)==0:
                plt.ylabel('GLRLM', fontsize=fontsize)
            plt.imshow(self.probabilityMatrices[imageType + "_glrlm"][0, :, :, n], aspect='auto')
            plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)

        fig.add_subplot(rows, 5, 21)
        plt.imshow(self.probabilityMatrices[imageType + "_glszm"][0, :, :], aspect='auto')
        plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
        plt.ylabel('GLSZM', fontsize=fontsize)

        fig.add_subplot(rows, 5, 22)
        plt.imshow(self.probabilityMatrices[imageType + "_gldm"][0, :, :], aspect='auto')
        plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
        plt.ylabel('GLDM', fontsize=fontsize)

        fig.add_subplot(rows, 5, 23)
        plt.imshow(self.probabilityMatrices[imageType + "_ngtdm"][0, :, :], aspect='auto')
        plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
        plt.ylabel('NGTDM', fontsize=fontsize)

        # rough approximation of 1D histogram
        pr = np.squeeze(np.sum(self.probabilityMatrices[imageType + "_glcm"][0,:,:,0],axis=1))
        prFWHM = np.nonzero(pr>(0.05*np.max(pr)))[0]

        if showHistogram:
            fig.add_subplot(rows, 5, 24)
            plt.bar(np.arange(len(pr)), pr/np.max(pr), width=1)
            # plt.plot(pr>(0.05*np.max(pr)))
            # plt.title(np.max(prFWHM) - np.min(prFWHM))

            # fig.add_subplot(rows, 5, 25)
            # plt.plot(pr/np.max(pr))
            # plt.yscale('log')

        fullPath = os.path.join(self.outputPath, 'probabilityMatrices', 'subjects')
        if not os.path.exists(fullPath):
            os.makedirs(fullPath)
        fileStr = 'probabilityMatrices_' + '__' + os.path.split(self.assessorFileName)[1].split('.')[0] + '_' + imageType + '_' + fileStr + '.pdf'
        fileStr = fileStr.replace(self.dcmPatientName, self.StudyPatientName)

        outputName = os.path.join(fullPath, fileStr)
        plt.gcf().savefig(outputName, orientation='landscape', format='pdf', dpi=1200) #papertype='a4'
        print('probabilityMatrices saved ' + outputName)
        plt.close()
        return outputName