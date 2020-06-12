import os
import numpy as np
import zipfile
import pydicom
import SimpleITK as sitk
import glob
import operator
from PIL import Image, ImageDraw
from autoDeleteTempFolder import autoDeleteTempFolder
from xml.dom import minidom
from itertools import compress
import yaml
from distutils.dir_util import copy_tree

from radiomics import featureextractor
import featureextractorenhanced

from scipy.ndimage import label
from scipy.linalg import circulant

import matplotlib.pyplot as plt
import csv
from mixture_cdf import GaussianMixtureCdf
from mixture_cdf import BayesianGaussianMixtureCdf
from scipy.stats import norm
from collections import OrderedDict
from shutil import copyfile

def _histEq(y, mask=None, flavour = 'Bayesian', nComponents = 20):

    if mask is not None:
        # if mask is input then get the reference data from y using the mask
        yRef = np.asarray(y[mask == 1]).reshape(-1, 1)
    else:
        # if no mask then assume y is the reference data
        yRef = y


    # Fit Gaussian mixture model to reference data yRef.
    if flavour is 'Bayesian':
        # With Bayesian model the prior acts to regularise the mixture model, i.e. only
        # a "sensible" number of components will have non-negligible
        # weights, so the mixture model will be parsimonious.  The
        # mixture components themselves also tend to cluster together
        # to further encourage parsimony.
        gmm = BayesianGaussianMixtureCdf(nComponents, random_state=10, max_iter=10000).fit(yRef)
    else:
        # this is standard EM Gaussian mixture model, and is included for evaluation purposes, but
        # not recommended for main analysis
        gmm = GaussianMixtureCdf(nComponents, random_state=10, max_iter=10000).fit(yRef)

    # Do histogram equalisation of fitted mixture model onto a truncated standard normal distribution.  The truncation is included
    # in order to ensure the bin edges used by the radiomics package are consistent across subjects (when the BinCount is specified).
    BW = 4 # BW = 4 implies a 4-sigma truncation
    loc = 0
    scale = 1
    out = {}
    cc = norm.cdf(-BW, loc=loc, scale=scale)
    yRef = norm.ppf(gmm.cdf(yRef)*(1-2*cc)+cc, loc=loc, scale=scale)
    # Sometimes numerical precision on norm.ppf means min/max are outside +/-BW, so force to be at limits.
    yRef[yRef < -BW] = -BW
    yRef[yRef > BW] = BW
    if mask is None:
        # Find values that are equal to the max and set one datum to BW.  Do similar for the min.
        idxMax = np.where(yRef==np.max(yRef))
        yRef[idxMax[0][0]] = BW
        idxMin = np.where(yRef==np.min(yRef))
        yRef[idxMin[0][0]] = -BW
        out["data"] = yRef
    else:
        y = norm.ppf(gmm.cdf(y)*(1-2*cc)+cc, loc=loc, scale=scale)
        # force the max/min values to be +/-W exactly, same as for yRef.
        y[y < -BW] = -BW
        y[y > BW] = BW
        # Find voxels inside the mask that are equal to the max and set one of them to BW.  Do similar for the min.
        idxMax = np.where(np.bitwise_and(y == np.max(yRef), mask==1))
        idxMin = np.where(np.bitwise_and(y == np.min(yRef), mask==1))
        y[idxMax[0][0], idxMax[1][0], idxMax[2][0]] = BW
        y[idxMin[0][0], idxMin[1][0], idxMin[2][0]] = -BW
        out["data"] = y
    out["gmm"] = gmm
    return out

# function to average over NxN blocks of pixels
def pixelBlockAverage(x,N):
    vr = np.zeros(x.shape[0])
    vr[0:N] = 1
    Vr = circulant(vr)[N-1::N,:]
    #
    vc = np.zeros(x.shape[1])
    vc[0:N] = 1
    Vc = circulant(vc)[N-1::N,:]
    return np.dot(Vr,np.dot(x,np.transpose(Vc)))/(N*N)

class radiomicAnalyser:

    def __init__(self, xnat_experiment, xnat_assessor, outputPath, assessorType):

        if not xnat_assessor.label in xnat_experiment.assessors.keys():
            raise Exception("Assessor "+xnat_assessor.label+" not found in experiment "+xnat_experiment.label)
        self.xnat_assessor = xnat_experiment.assessors[xnat_assessor.label]
        self.assessorType = assessorType
        self.xnat_experiment = xnat_experiment
        self.outputPath = outputPath
        self.bayesianGMMcomponents = 20 # default number of components in mixture model for histogram equalization
        self.folderForLocalCopy = ''

        print('Processing subject: ' + xnat_experiment.dcm_patient_name)

    ############
    def setFolderForLocalCopy(self, filename):
        self.folderForLocalCopy = filename


    ##########################
    def loadROIData(self, segmentLabelsToExtractStr=None, copyToLocalFolder=False):

        if self.assessorType['type'] == "SEG":
            with autoDeleteTempFolder('xnat') as temp:
                tempSeg = os.path.join(temp.folder, 'seg.zip')
                self.xnat_assessor.download(tempSeg, verbose=False)
                with zipfile.ZipFile(tempSeg, 'r') as zip_ref:
                    zip_ref.extractall(temp.folder)
                dcmFile = glob.glob(os.path.join(temp.folder,'**','*.dcm'), recursive=True)
                if len(dcmFile) > 1:
                     raise Exception("downloaded assessor contains more than one DICOM file - only one expected")
                dcmSeg = pydicom.dcmread(dcmFile[0])

                if copyToLocalFolder and os.path.exists(self.folderForLocalCopy):
                    thisFolder = os.path.join(self.folderForLocalCopy, str(dcmSeg.PatientName))
                    if not os.path.exists(thisFolder):
                        os.mkdir(thisFolder)
                    copyfile(dcmFile[0], os.path.join(thisFolder, os.path.split(dcmFile[0])[1]))

            self.patientName = str(dcmSeg.PatientName)

            if len(dcmSeg.ReferencedSeriesSequence) != 1:
                raise Exception("DICOM SEG file referencing more than one series not supported!")
            self.refSeriesUID = dcmSeg.ReferencedSeriesSequence[0].SeriesInstanceUID

            # read pixel array (bits) using pydicom convenience method that accounts for weird
            # bit unpacking that is required for python
            maskHere = dcmSeg.pixel_array
            # make sure single slice masks have rows/cols/slices along correct dimension
            if len(maskHere.shape)==2:
                maskHere = np.reshape(maskHere, (1, maskHere.shape[1], maskHere.shape[0]))  # the dimension order needs testing!!

            # get refSopInstance for each frame
            self.refSopInstUID = []
            for n, funGrpSeq in enumerate(dcmSeg.PerFrameFunctionalGroupsSequence):
                if len(funGrpSeq.DerivationImageSequence) != 1:
                    raise Exception("Dicom Seg file has more than one element in DerivationImageSequence!")
                if len(funGrpSeq.DerivationImageSequence[0].SourceImageSequence) != 1:
                    raise Exception("Dicom Seg file has more than one element in SourceImageSequence!")
                if len(funGrpSeq.SegmentIdentificationSequence) != 1:
                    raise Exception("Dicom Seg file has more than one element in SegmentIdentificationSequence!")
                referencedSegmentNumber = funGrpSeq.SegmentIdentificationSequence[0].ReferencedSegmentNumber
                referencedSegmentLabel = dcmSeg.SegmentSequence._list[referencedSegmentNumber - 1].SegmentLabel
                # list items are dictionary containing "uid" and "data".  "data" is there to store the contour co-ordinates
                # when loading dicomRT files, but included here so other functions work properly
                self.refSopInstUID.append(
                    {"uid": funGrpSeq.DerivationImageSequence[0].SourceImageSequence[0].ReferencedSOPInstanceUID,
                     "data": None,
                     "referencedSegmentLabel": referencedSegmentLabel.lower(),
                     "mask": maskHere[n,:,:]})

            # slice mask and refSopInstUID list to only include segments matching input string
            if segmentLabelsToExtractStr is not None:
                indToKeep = [x["referencedSegmentLabel"] == segmentLabelsToExtractStr for x in self.refSopInstUID]
                if not any(indToKeep):
                    print(' ')
                    print(self.patientName + ' ' + self.xnat_experiment.data["label"] + ':' + self.xnat_assessor.data[
                        "label"])
                    [print(x["referencedSegmentLabel"]) for x in self.refSopInstUID]
                    print(' ')
                self.refSopInstUID = list(compress(self.refSopInstUID, indToKeep))

            # get pixelSpacing
            if len(dcmSeg.SharedFunctionalGroupsSequence) != 1:
                raise Exception("Dicom Seg file has more than one element in SharedFunctionGroupsSequence")
            if len(dcmSeg.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence) != 1:
                raise Exception("Dicom Seg file has more than one element in PixelMeasuresSequence")
            self.maskPixelSpacing = [float(x) for x in dcmSeg.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0].PixelSpacing]
            print('Segmentation file loaded')

        if self.assessorType['type'] == "AIM":
            with autoDeleteTempFolder('xnat') as temp:
                tempAim = os.path.join(temp.folder, 'aim.zip')
                self.xnat_assessor.download(tempAim)
                with zipfile.ZipFile(tempAim, 'r') as zip_ref:
                    zip_ref.extractall(temp.folder)

                if self.assessorType['fileType']=="xml":
                    xmlFile = glob.glob(os.path.join(temp.folder, '**', '*.xml'), recursive=True)
                    if len(xmlFile) > 1:
                        raise Exception("downloaded assessor contains more than one xml file - only one expected")
                    xDOM = minidom.parse(xmlFile[0])

                    xPerson = xDOM.getElementsByTagName('person').item(0)
                    self.patientName = xPerson.getElementsByTagName('name').item(0).getAttribute('value')

                    if copyToLocalFolder and os.path.exists(self.folderForLocalCopy):
                        thisFolder = os.path.join(self.folderForLocalCopy, self.patientName)
                        if not os.path.exists(thisFolder):
                            os.mkdir(thisFolder)
                        copyfile(xmlFile[0], os.path.join(thisFolder, os.path.split(xmlFile[0])[1]))

                    # assume only one imageSeries item linked to one instanceUid item
                    self.refSeriesUID = xDOM.getElementsByTagName('imageSeries').item(0).getElementsByTagName('instanceUid').item(0).getAttribute('root')

                    if xDOM.getElementsByTagName('ImageAnnotation').length != 1:
                        raise Exception("AIM file containing more than one ImageAnnotation not supported!")

                    self.refSopInstUID = []
                    for me in xDOM.getElementsByTagName('MarkupEntity'):
                        scc = me.getElementsByTagName('twoDimensionSpatialCoordinateCollection')

                        if len(scc) != 1:
                            raise Exception("AIM file has MarkupEntity with more than one twoDimensionSpatialCoordinateCollection")
                        index = []
                        x = []
                        y = []
                        z = [] # this will not be the z co-ordinate, but needs to be included for compatibility with RT-struct.
                               # use the shapeIdentifier attribute instead as this is sequential for each contour
                        for sc in scc.item(0).getElementsByTagName('TwoDimensionSpatialCoordinate'):
                            index.append(int(sc.getElementsByTagName('coordinateIndex').item(0).getAttribute('value')))
                            x.append(float(sc.getElementsByTagName('x').item(0).getAttribute('value')))
                            y.append(float(sc.getElementsByTagName('y').item(0).getAttribute('value')))
                            z.append(int(me.getElementsByTagName('shapeIdentifier').item(0).getAttribute('value')))
                        # check index list starts and ends with 0, and is sorted
                        if not (index[0]==0 and index[-1]==0 and len(set(map(operator.sub, index[1:-1], index[0:-2])))==1):
                            raise Exception("AIM file with non-closed or non-sorted contour!")
                        # interleave co-ordinate lists to match RT-struct format
                        contourData = [val for tup in zip(*[x, y, z]) for val in tup]
                        self.refSopInstUID.append({"uid":me.getElementsByTagName('imageReferenceUid').item(0).getAttribute('root'),
                                                   "data":contourData})

                elif self.assessorType['fileType']=="dcm":
                    dcmFile = glob.glob(os.path.join(temp.folder, '**', '*.dcm'), recursive=True)
                    if len(dcmFile) > 1:
                        raise Exception("downloaded assessor contains more than one DICOM file - only one expected")
                    dcmRT = pydicom.dcmread(dcmFile[0])

                    if copyToLocalFolder and os.path.exists(self.folderForLocalCopy):
                        thisFolder = os.path.join(self.folderForLocalCopy, str(dcmRT.PatientName))
                        if not os.path.exists(thisFolder):
                            os.mkdir(thisFolder)
                        copyfile(dcmFile[0], os.path.join(thisFolder, os.path.split(dcmFile[0])[1]))

                    self.patientName = str(dcmRT.PatientName)

                    # check only one item in each level of hierarchy going down to ReferencedSeriesUID
                    if len(dcmRT.ReferencedFrameOfReferenceSequence) != 1:
                        raise Exception("DICOM RT file referencing more than one frame of reference not supported!")
                    rfors = dcmRT.ReferencedFrameOfReferenceSequence[0]
                    if len(rfors.RTReferencedStudySequence) != 1:
                        raise Exception("DICOM RT file referencing more than one study not supported!")
                    rtrss = rfors.RTReferencedStudySequence[0]
                    if len(rtrss.RTReferencedSeriesSequence) != 1:
                        raise Exception("DICOM RT file referencing more than one series not supported!")
                    self.refSeriesUID = rtrss.RTReferencedSeriesSequence[0].SeriesInstanceUID

                    if len(dcmRT.ROIContourSequence) != 1:
                        raise Exception("DICOM RT file containing more than one ROIContourSequence not supported!")

                    self.refSopInstUID = []
                    for cs in dcmRT.ROIContourSequence[0].ContourSequence:
                        if len(cs.ContourImageSequence) != 1:
                            raise Exception("DICOM RT file containing (individual) contour that references more than one image not supported!")
                        self.refSopInstUID.append({"uid":cs.ContourImageSequence[0].ReferencedSOPInstanceUID,
                                                   "data":cs.ContourData})



    ##########################
    def loadImageData(self, copyToLocalFolder=False, seriesInstDict=None):

        # locate scan referenced by assessor
        scanFromSeriesUID = {}
        scans = self.xnat_experiment.scans.values()
        for scan in scans:
            scanFromSeriesUID[scan.uid] = scan

        with autoDeleteTempFolder('xnat') as temp:
            if (seriesInstDict is not None) and (self.refSeriesUID in seriesInstDict):
                # copy into temp
                copy_tree(seriesInstDict[self.refSeriesUID], temp.folder)
            else:
                scanFromSeriesUID[self.refSeriesUID].download_dir(temp.folder, verbose=False)
            dcmFiles = glob.glob(os.path.join(temp.folder, '**', '*.dcm'), recursive=True)
            dcmFolder = np.unique([os.path.split(x)[0] for x in dcmFiles])
            if not dcmFolder.size==1:
                raise Exception("Downloaded images located in more than one folder!")

            # Use sitk library as a rapid method for getting at all the header data.
            # Unfortunately, sitk library doesn't work for image loading if the slice
            # locations are not evenly spaced, which is violated for some CT data.
            seriesReader = sitk.ImageSeriesReader()
            seriesReader.MetaDataDictionaryArrayUpdateOn()
            dicom_names = seriesReader.GetGDCMSeriesFileNames(dcmFolder[0])
            seriesReader.SetFileNames(dicom_names)
            readFileNames = seriesReader.GetFileNames()
            _ = seriesReader.Execute()

            # check if axial as all further code assumes axial (to some extent)
            imageOrientationPatient = seriesReader.GetMetaData(0, "0020|0037")
            axialArr = [1, 0, 0, 0, 1, 0]
            axialTol = 1e-6
            if any([np.abs(np.abs(float(x))-y)>axialTol for x,y in zip(imageOrientationPatient.split('\\'), axialArr)]):
                raise Exception("Non-axial image referenced by annotation file - not supported yet!")

            # make dictionary to find file from SOPInstanceUID
            fileFromUID = {}
            for n, thisFileName in enumerate(readFileNames):
                fileFromUID[seriesReader.GetMetaData(n, "0008|0018")] = thisFileName

            # convenience parameters
            self.numRows = int(seriesReader.GetMetaData(0, "0028|0010"))
            self.numCols = int(seriesReader.GetMetaData(0, "0028|0011"))
            self.imagePixelSpacing = [float(x) for x in seriesReader.GetMetaData(0, "0028|0030").split('\\')]

            # load sopInstances that are referenced by the annotation file into a dictionary
            # using a dictionary should automatically account for any duplicated referenced sopInstances
            dcmFromSopInstDict = {}
            # also get slice locations as these will be used to package slices into a coherent volume
            sliceLocationList = []
            for n, rs in enumerate(self.refSopInstUID):
                thisFileName = fileFromUID[rs["uid"]]
                dcm = pydicom.dcmread(thisFileName)
                if not (rs["uid"] == dcm.SOPInstanceUID):
                    raise Exception("SOPInstanceUID does not match corresponding list item")
                slLoc = dcm.ImagePositionPatient[2]
                sliceLocationList.append(slLoc)
                dcmFromSopInstDict[dcm.SOPInstanceUID] = dcm
                if copyToLocalFolder and os.path.exists(self.folderForLocalCopy):
                    thisFolder = os.path.join(self.folderForLocalCopy, self.patientName)
                    if not os.path.exists(thisFolder):
                        os.mkdir(thisFolder)
                    thisFolder = os.path.join(thisFolder, 'images')
                    if not os.path.exists(thisFolder):
                        os.mkdir(thisFolder)
                    copyfile(fileFromUID[rs["uid"]], os.path.join(thisFolder, os.path.split(thisFileName)[1]))

        # this step removes duplicates and sorts on slice location (note that dcmFromSliceLocationDict will also have no duplicates)
        self.sliceLocationListSorted = np.unique(sliceLocationList)

        self.imVolume = np.zeros((len(self.sliceLocationListSorted), self.numRows, self.numCols))
        self.imageSlicePos = {}
        self.mask = np.zeros((len(self.sliceLocationListSorted), self.numRows, self.numCols))

        # loop over referenced SopInstances, filling in imVolume and mask
        # for imVolume we might overwrite a slice more than once if there are two mask regions on the same slice
        # and we will amalgamate mask regions that are on the same slice
        for n, rs in enumerate(self.refSopInstUID):

            # get image data for slice
            dcm = dcmFromSopInstDict[rs["uid"]]
            sliceIndex = list(self.sliceLocationListSorted == dcm.ImagePositionPatient[2]).index(True)
            self.imVolume[sliceIndex,:,:] = dcm.RescaleSlope * dcm.pixel_array + dcm.RescaleIntercept
            # use sliceIndex to index into a dictionary as we will retrieve the elements this way later
            # hmm, timageSopInstanceUID and imageSlicePos may not now be necessary...
            #self.imageSopInstanceUID[sliceIndex] = dcm.SOPInstanceUID
            self.imageSlicePos[sliceIndex] = dcm.ImagePositionPatient
            self.mask[sliceIndex, :, :] = np.logical_or(self.mask[sliceIndex, :, :], rs["mask"])

            # if annotation is a AIM then make mask from contours
            if self.assessorType['type'] == "AIM":
                polygonPatient = np.array(rs["data"]).reshape((int(len(rs["data"])/3),3))
                origin = np.array([float(t) for t in dcm.ImagePositionPatient])
                spacing = np.array([float(t) for t in dcm.PixelSpacing])
                xNorm = np.array([float(t) for t in dcm.ImageOrientationPatient[0:3]])
                yNorm = np.array([float(t) for t in dcm.ImageOrientationPatient[3:6]])
                if self.assessorType['fileType']=='dcm':
                    # dicom-RT: center of the top-left voxel is at "origin" and polygonPatient is in patient co-ordinates
                    if self.assessorType['withError']==False:
                        xImage = np.dot(polygonPatient - origin, xNorm)/spacing[0]
                        yImage = np.dot(polygonPatient - origin, yNorm)/spacing[1]
                    else:
                        # ... the extra -1 here accounts for an error in an old version of the AIM -> dicomRT conversion code
                        xImage = np.dot(polygonPatient - origin, xNorm)/spacing[0] - 1
                        yImage = np.dot(polygonPatient - origin, yNorm)/spacing[1] - 1
                elif self.assessorType['fileType']=='xml':
                    # AIM xml: top-left of the top-left voxel is at [0,0] and polygonPatient is in image co-ordinates
                    xImage = polygonPatient[:,0] - 0.5
                    yImage = polygonPatient[:,1] - 0.5

                polygonPixel = list(xImage) + list(yImage) # concatenate to make correct size array

                # make mask for matrix size 5x larger than original, then for each 5x5 section final mask will be set if
                # more than half of
                fineGrid = 5
                # add 0.5 because Draw(img).polygon has origin at corner of top-left pixel, not center
                polygonPixel[0::2] = list(fineGrid*(xImage+0.5))
                polygonPixel[1::2] = list(fineGrid*(yImage+0.5))
                img = Image.new('L', (fineGrid*dcm.Columns, fineGrid*dcm.Rows), 0)
                ImageDraw.Draw(img).polygon(polygonPixel, outline=1, fill=2)
                # fill is 2, outline is 1, so threshold for output of pixelBlockAverage indicating half
                # the pixel is inside the polygon is 1
                # do logical OR with existing mask in case there is more than one contour on an image
                self.mask[sliceIndex,:,:] = np.logical_or(self.mask[sliceIndex,:,:], pixelBlockAverage(np.array(img), fineGrid)>1)
                self.maskPixelSpacing = [float(x) for x in dcm.PixelSpacing]
        print("Images loaded")


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

    ##########################
    def getLumenMask(self):

        for n in range(self.mask.shape[0]):
            labelled_mask, num_labels = label(self.mask[n, :, :] == 0)
            # remove small regions
            refined_mask = 1 - self.mask[n, :, :]
            for thisLabel in range(num_labels):
                labelArea = np.sum(refined_mask[labelled_mask == (thisLabel + 1)])
                if labelArea <= 4:
                    refined_mask[labelled_mask == (thisLabel + 1)] = 0
            self.mask[n, :, :] = 1 - refined_mask

    ##########################
    def clipLowCountBins(self, lowCount=2, paramFileName='Params.yaml'):

        if not (hasattr(self,'imVolume') and hasattr(self,'mask')):
            raise Exception("Image data or mask not loaded, so cannot clip low bin counts!")

        if (paramFileName is not None) and paramFileName.endswith(".yaml"):
            with open(paramFileName, "r") as stream:
                dict = yaml.safe_load(stream)
            if 'binWidth' in dict['setting'].keys():
                binWidth = dict['setting']['binWidth']
            else:
                raise Exception("Parameter file does not specify binWidth - not applying clipping")
        else:
            raise Exception("Problem with Parameter file (.yaml)")

        # make bins and compute histogram to match what pyradiomics will do
        roiValues = self.imVolume[np.where(self.mask)]
        minBinEdge = (np.floor(roiValues.min() / binWidth) - 1) * binWidth
        maxBinEdge = (np.ceil(roiValues.max() / binWidth) + 1) * binWidth
        nBins = int((maxBinEdge - minBinEdge) / binWidth)
        binEdges = np.linspace(minBinEdge, maxBinEdge, nBins + 1)

        counts = np.histogram(roiValues, bins=binEdges)[0]

        idxMax = np.argmax(counts)

        # find rightmost bin where the counts are <= lowCount
        idxLow = np.where(counts[0:idxMax+1] <= lowCount)[0].max()
        # find rightmost bin where the counts are <= lowCount
        idxHigh = idxMax + np.where(counts[idxMax:] <= lowCount)[0].min()

        # clip limits
        clipLow = binEdges[idxLow + 1]
        clipHigh = binEdges[idxHigh]

        # shrink clip limits a bit to make sure we are inside the corresponding bin edges
        clipLow = clipLow + np.abs(clipLow)*10*np.finfo(float).eps
        clipHigh = clipHigh - np.abs(clipHigh)*10*np.finfo(float).eps

        # clip image volume
        self.imVolume = np.clip(self.imVolume, clipLow, clipHigh)


    ##########################
    def computeRadiomicFeatures(self, paramFileName='Params.yaml', oldDictStr='', newDictStr=''):

        zLoc = self.sliceLocationListSorted
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
        maskSitk.SetOrigin(self.imageSlicePos[0])
        maskSitk.SetSpacing((float(self.maskPixelSpacing[0]), float(self.maskPixelSpacing[1]), abs(dz)))
        maskSitk.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, float(np.sign(dz))))

        imageSitk = sitk.GetImageFromArray(self.imVolume)
        imageSitk.SetOrigin(self.imageSlicePos[0])
        imageSitk.SetSpacing((float(self.imagePixelSpacing[0]), float(self.imagePixelSpacing[1]), abs(dz)))
        imageSitk.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, float(np.sign(dz))))

        params = os.path.join(os.getcwd(),paramFileName)
        #extractor = featureextractor.RadiomicsFeatureExtractor(params)
        extractor = featureextractorenhanced.RadiomicsFeatureExtractorEnhanced(params)

        segmentNumber = int(1)
        featureVector = extractor.execute(imageSitk, maskSitk, segmentNumber)
        self.probabilityMatrices = extractor.getProbabilityMatrices(imageSitk, maskSitk, segmentNumber)

        # rename keys if indicated
        if ((oldDictStr is not '') and (newDictStr is not '')):
            featureVectorRenamed = OrderedDict()
            for key in featureVector.keys():
                newKey = str(key).replace(oldDictStr, newDictStr)
                featureVectorRenamed[newKey] = featureVector[key]
            featureVector = featureVectorRenamed

        # insert or append featureVector just computed
        if hasattr(self, 'featureVector'):
            for key in featureVector.keys():
                self.featureVector[key] = featureVector[key]
        else:
            self.featureVector = featureVector

        print('Radiomic features computed')

    ##########################
    def saveThumbnail(self, fileStr = '', vmin=None, vmax=None):

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
        imageMontage = self.imVolume[0, :, :][maskCols, :][:, maskRows]
        maskBar = np.zeros((np.sum(maskCols), 5))
        imageBar = 500*np.ones((np.sum(maskCols), 5))
        for n in range(self.mask.shape[0] - 1):
            maskMontage = np.concatenate((maskMontage, maskBar, self.mask[n + 1, :, :][maskCols, :][:, maskRows]), axis=1)
            imageMontage = np.concatenate((imageMontage, imageBar, self.imVolume[n + 1, :, :][maskCols, :][:, maskRows]), axis=1)

        # get grayscale limits
        maskPct = np.percentile(np.asarray(self.imVolume[self.mask == 1]).reshape(-1, 1),[1, 99])
        if vmin is None:
            vmin = maskPct[0]
        if vmax is None:
            vmax = maskPct[1]

        # plot montaged cropped images
        plt.clf()
        plt.imshow(imageMontage, cmap='gray', vmin=vmin, vmax=vmax)
        plt.colorbar(orientation='vertical')

        # clunky but effective way of drawing a line for the boundary of the mask
        maskEdges = findMaskEdges(maskMontage).astype(int)
        edgeIdx = np.where(maskEdges == 1)
        for n in range(len(edgeIdx[0])):
            if maskMontage[edgeIdx[0][n] + 1, edgeIdx[1][n]] == 0:
                x = [edgeIdx[1][n] - 0.5, edgeIdx[1][n] + 0.5]
                y = [edgeIdx[0][n] + 0.5, edgeIdx[0][n] + 0.5]
                plt.plot(x, y, color='cyan', linewidth=0.1)
            if maskMontage[edgeIdx[0][n] - 1, edgeIdx[1][n]] == 0:
                x = [edgeIdx[1][n] - 0.5, edgeIdx[1][n] + 0.5]
                y = [edgeIdx[0][n] - 0.5, edgeIdx[0][n] - 0.5]
                plt.plot(x, y, color='cyan', linewidth=0.1)
            if maskMontage[edgeIdx[0][n], edgeIdx[1][n] + 1] == 0:
                x = [edgeIdx[1][n] + 0.5, edgeIdx[1][n] + 0.5]
                y = [edgeIdx[0][n] - 0.5, edgeIdx[0][n] + 0.5]
                plt.plot(x, y, color='cyan', linewidth=0.1)
            if maskMontage[edgeIdx[0][n], edgeIdx[1][n] - 1] == 0:
                x = [edgeIdx[1][n] - 0.5, edgeIdx[1][n] - 0.5]
                y = [edgeIdx[0][n] - 0.5, edgeIdx[0][n] + 0.5]
                plt.plot(x, y, color='cyan', linewidth=0.1)

        plt.xticks([], [])
        plt.yticks([], [])

        plt.title(str(self.patientName) + '  ' + str(self.xnat_assessor.label) + '  ' + str(self.xnat_assessor.data["name"]))
        plt.axis('off')

        fullPath = os.path.join(self.outputPath, 'roiThumbnails', 'subjects')
        if not os.path.exists(fullPath):
            os.makedirs(fullPath)
        if fileStr is not '':
            fileStr = '__'+fileStr
        fileStr = 'roiThumbnail__' + str(self.patientName) + '__' + str(self.xnat_assessor.label) + '__' + str(self.xnat_assessor.data["name"]) + fileStr + '.pdf'
        outputName = os.path.join(fullPath, fileStr)
        plt.gcf().savefig(outputName,  papertype='a4', orientation='landscape', format='pdf', dpi=1200)
        print('Thumbnail saved '+outputName)
        plt.close()
        return outputName

    ##########################
    def saveResult(self):

        headers = []
        row = []

        # add XNAT info so we can convert to DICOM SR later
        headers.append("XNAT_project")
        row.append(self.xnat_experiment.project)

        headers.append("XNAT_subject_ID")
        row.append(self.xnat_experiment.subject_id)

        headers.append("XNAT_image_session_ID")
        row.append(self.xnat_experiment.id)

        headers.append("XNAT_session_label")
        row.append(self.xnat_experiment.label)

        headers.append("XNAT_assessor_ID")
        row.append(self.xnat_assessor.id)

        headers.append("XNAT_assessor_label")
        row.append(self.xnat_assessor.label)

        headers.append("XNAT_dcm_PatientId")
        row.append(self.xnat_experiment.data["dcmPatientId"])

        headers.append("XNAT_dcm_PatientName")
        row.append(self.xnat_experiment.data["dcmPatientName"])

        headers.append("XNAT_dcm_StudyInstanceUID")
        row.append(self.xnat_experiment.uid)

        headers.append("XNAT_dcm_SeriesInstanceUID")
        row.append(self.refSeriesUID)

        headers.append("XNAT_dcmSeg_SopInstanceUID")
        row.append(self.xnat_assessor.uid)

        headers.extend(list(self.featureVector.keys()))
        for h in list(self.featureVector.keys()):
            row.append(self.featureVector.get(h, "N/A"))

        fullPath = os.path.join(self.outputPath, 'radiomicFeatures', 'subjects')
        if not os.path.exists(fullPath):
            os.makedirs(fullPath)
        fileStr = 'radiomicFeatures__' + str(self.patientName) + '__' + str(self.xnat_assessor.label) + '__' + str(self.xnat_assessor.data["name"]) + '.csv'
        outputName = os.path.join(fullPath, fileStr)

        with open(outputName, 'w') as fo:
            writer = csv.writer(fo, lineterminator='\n')
            writer.writerow(headers)
            writer.writerow(row)

        print("Results file saved")
        return outputName

    ##########################
    def saveProbabilityMatrices(self, fileStr=''):

        fig = plt.figure()
        columns = 7
        rows = 5
        # show GLCM
        for n in range(self.probabilityMatrices["original_glcm"].shape[3]):
            fig.add_subplot(rows, columns, n+1)
            if n==0:
                plt.title(str(self.patientName) + '  ' + str(self.xnat_assessor.label) + '  ' + str(self.xnat_assessor.data["name"]),
                          fontdict = {'horizontalalignment': 'left'})

            if np.mod(n,7)==0:
                plt.ylabel('GLCM')
            plt.imshow(self.probabilityMatrices["original_glcm"][0,:,:,n])
            plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)

        # show GLRLM
        for n in range(self.probabilityMatrices["original_glrlm"].shape[3]):
            fig.add_subplot(rows, columns, n+15)
            if np.mod(n,7)==0:
                plt.ylabel('GLRLM')
            plt.imshow(self.probabilityMatrices["original_glrlm"][0, :, :, n])
            plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)

        fig.add_subplot(rows, columns, 30)
        plt.imshow(self.probabilityMatrices["original_glszm"][0, :, :])
        plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
        plt.ylabel('GLSZM')

        fig.add_subplot(rows, columns, 32)
        plt.imshow(self.probabilityMatrices["original_gldm"][0, :, :])
        plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
        plt.ylabel('GLDM')

        fig.add_subplot(rows, columns, 34)
        plt.imshow(self.probabilityMatrices["original_ngtdm"][0, :, :])
        plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
        plt.ylabel('NGTDM')

        fullPath = os.path.join(self.outputPath, 'probabilityMatrices', 'subjects')
        if not os.path.exists(fullPath):
            os.makedirs(fullPath)
        if fileStr is not '':
            fileStr = '__' + fileStr
        fileStr = 'probabilityMatrices__' + str(self.patientName) + '__' + str(self.xnat_assessor.label) + '__' + str(
            self.xnat_assessor.data["name"]) + fileStr + '.pdf'
        outputName = os.path.join(fullPath, fileStr)
        plt.gcf().savefig(outputName, papertype='a4', orientation='landscape', format='pdf', dpi=1200)
        print('probabilityMatrices saved ' + outputName)
        plt.close()
        return outputName

    ##########################
    def saveImageValues(self, fileStr=''):

        fullPath = os.path.join(self.outputPath, 'imageValues', 'subjects')
        if not os.path.exists(fullPath):
            os.makedirs(fullPath)
        if fileStr is not '':
            fileStr = '__' + fileStr
        fileStr = 'imageValues__' + str(self.patientName) + '__' + str(self.xnat_assessor.label) + '__' + str(
            self.xnat_assessor.data["name"]) + fileStr + '.csv'
        outputName = os.path.join(fullPath, fileStr)
        indices = np.where(self.mask)
        imageValues = self.imVolume[indices]
        saveList = []
        saveList.append(imageValues)
        saveList.append(indices[0])
        saveList.append(indices[1])
        if (len(indices)>1):
            saveList.append(indices[2])
        np.savetxt(outputName, saveList, fmt = '%.16f', delimiter=',', newline='\n', header='image values\nrow index\ncol index\nslice index\n\n')
        print('imageValues saved ' + outputName)
        return outputName


    ##########################
    def histogramEqualiseImageData(self):

        self.imVolume = _histEq(self.imVolume, self.mask, flavour = 'Bayesian', nComponents=self.bayesianGMMcomponents)["data"]

    ##########################
    def histogramEqualiseEvaluate(self):

        # grab pixel data as 1D array
        yRef = np.asarray(self.imVolume[self.mask == 1]).reshape(-1, 1)

        standard = _histEq(yRef, flavour='EM', nComponents=10)
        bayes = _histEq(yRef, flavour='Bayesian', nComponents=self.bayesianGMMcomponents)

        x = np.linspace(np.min(yRef), np.max(yRef), 1000)
        bins = np.linspace(x[0], x[-1], 50)

        logprob_standard = standard["gmm"].score_samples(x.reshape(-1, 1))
        logprob_bayes = bayes["gmm"].score_samples(x.reshape(-1, 1))

        standard_cdf = standard["gmm"].cdf(x)
        yT_standard = standard["data"]

        bayes_cdf = bayes["gmm"].cdf(x)
        yT_bayes = bayes["data"]

        fig = plt.figure(figsize=(6.4, 10))
        fig.suptitle(self.patientName)

        xLim = (-250, 250)

        ax = fig.add_subplot(312)
        ax.set_prop_cycle(None)
        ax.plot(x, norm.ppf(bayes_cdf, loc=0, scale=1), label='DP-GMM')
        ax.plot(x, norm.ppf(standard_cdf, loc=0, scale=1), linewidth=0.8, label='GMM')
        ax.set_xlim(xLim[0], xLim[1])
        ax.legend()

        ax = fig.add_subplot(621)
        ax.plot(x, np.exp(logprob_standard))
        ax.hist(yRef, bins, density=True, histtype='stepfilled')
        ax.title.set_text('GMM')
        ax.set_xlim(xLim[0], xLim[1])

        ax = fig.add_subplot(622)
        ax.set_prop_cycle(None)
        ax.plot(x, np.exp(logprob_bayes))
        ax.hist(yRef, bins, density=True, histtype='stepfilled')
        ax.title.set_text('DP-GMM')
        ax.set_xlim(xLim[0], xLim[1])

        ax = fig.add_subplot(623)
        ax.set_prop_cycle(None)
        ax.plot(x, np.exp(logprob_standard))
        ax.hist(yRef, bins, density=True, histtype='stepfilled')
        ax.set_xlim(xLim[0], xLim[1])
        ax.set_yscale('log')
        yMax = ax.get_ylim()[1]
        ax.set_ylim(1e-4 * yMax, 2 * yMax)

        ax = fig.add_subplot(624)
        ax.set_prop_cycle(None)
        ax.plot(x, np.exp(logprob_bayes))
        ax.hist(yRef, bins, density=True, histtype='stepfilled')
        ax.set_xlim(xLim[0], xLim[1])
        ax.set_yscale('log')
        yMax = ax.get_ylim()[1]
        ax.set_ylim(1e-4 * yMax, 2 * yMax)

        x = np.linspace(np.min(yT_standard), np.max(yT_standard), 100)
        bins = np.linspace(x[0], x[-1], 50)

        ax = fig.add_subplot(629)
        ax.set_prop_cycle(None)
        ax.plot(x, norm.pdf(x, loc=0, scale=1))
        ax.hist(yT_standard, bins, density=True, histtype='stepfilled')
        ax.set_xlim(x[0], x[-1])

        ax = fig.add_subplot(6, 2, 10)
        ax.set_prop_cycle(None)
        ax.plot(x, norm.pdf(x, loc=0, scale=1))
        ax.hist(yT_bayes, bins, density=True, histtype='stepfilled')
        ax.set_xlim(x[0], x[-1])

        ax = fig.add_subplot(6, 2, 11)
        ax.set_prop_cycle(None)
        ax.plot(x, norm.pdf(x, loc=0, scale=1))
        ax.hist(yT_standard, bins, density=True, histtype='stepfilled')
        ax.set_xlim(x[0], x[-1])
        ax.set_yscale('log')
        yMax = ax.get_ylim()[1]
        ax.set_ylim(1e-4 * yMax, 2 * yMax)

        ax = fig.add_subplot(6, 2, 12)
        ax.set_prop_cycle(None)
        ax.plot(x, norm.pdf(x, loc=0, scale=1))
        ax.hist(yT_bayes, bins, density=True, histtype='stepfilled')
        ax.set_xlim(x[0], x[-1])
        ax.set_yscale('log')
        yMax = ax.get_ylim()[1]
        ax.set_ylim(1e-4 * yMax, 2 * yMax)

        #plt.show()

        fullPath = os.path.join(self.outputPath, 'histEq', 'subjects')
        if not os.path.exists(fullPath):
            os.makedirs(fullPath)
        outputName = os.path.join(fullPath, 'histEq_'+self.patientName+'.pdf')
        plt.gcf().savefig(outputName,  papertype='a4', orientation='portrait', format='pdf', dpi=1200)
        print('Histogram equalisation figure saved')
        return outputName
