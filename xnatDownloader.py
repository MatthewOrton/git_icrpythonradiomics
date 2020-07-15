import os
import xnat
import glob
import zipfile
import pydicom
from shutil import rmtree
from xml.dom import minidom
from numpy import unique
from itertools import compress
from re import compile

def myStrJoin(strList):
    return '__II__'.join(strList)

def myStrSplit(str):
    return str.split('__II__')

class xnatDownloader:

    ##########################
    def __init__(self,
                 serverURL='',
                 projectStr='',
                 downloadPath='',
                 assessorStyle='',
                 removeSecondaryAndSnapshots=False,
                 roiCollectionLabelFilter='',
                 roiLabelFilter=None):

        self.serverURL = serverURL
        self.projectStr = projectStr
        self.downloadPath = downloadPath
        self.assessorStyle = assessorStyle
        self.removeSecondaryAndSnapshots = removeSecondaryAndSnapshots
        self.roiCollectionLabelFilter = roiCollectionLabelFilter
        self.roiLabelFilter = roiLabelFilter
        self.xnat_session = None

        self.downloadPathZip = os.path.join(self.downloadPath, 'temp.zip')
        if os.path.exists(self.downloadPathZip):
            os.remove(self.downloadPathZip)

        self.downloadPathUnzip = os.path.join(self.downloadPath, 'tempUnzip')
        if os.path.exists(self.downloadPathUnzip):
            rmtree(self.downloadPathUnzip)

        print(' ')
        print('Folder for downloads = ' + self.downloadPath + '\n')

        if os.path.exists(self.downloadPath):
            blitz = input("Download folder exists.  Delete contents? : ")
            if blitz == 'y':
                rmtree(self.downloadPath)
                os.mkdir(self.downloadPath)
        else:
            os.mkdir(self.downloadPath)


    ##########################
    def getAssessorStyle(self):
        return self.assessorStyle


    ##########################
    def setAssessorStyle(self, assessorStyle):
        self.assessorStyle = assessorStyle


    ##########################
    def downloadAssessors_Project(self):
        # Get list of all subjects in project.
        subjectList = self.getSubjectList_Project()
        self.subjectList_downloadAssessors(subjectList)


    ##########################
    # Download annotations associated with listed experiments.
    # Store as flat files in self.downloadPath and change filename to make it easier to locate
    def experimentList_downloadAssessors(self, experimentList):
        for experiment in experimentList:
            print(experiment + ':')
            try:
                with xnat.connect(server=self.serverURL) as self.xnat_session:
                    if experiment in self.xnat_session.projects[self.projectStr].experiments.keys():
                        xnat_experiment = self.xnat_session.projects[self.projectStr].experiments[experiment]
                        self.__downloadAndMoveAssessors(xnat_experiment)
                    else:
                        print('Experiment not found!')
                    print(' ')
            except Exception as e:
                print(e)


    ##########################
    # Download scans associated with listed experiments.
    # Folder structure is: experiment>scan>image and is the same as for downloadSubjectList()
    def experimentList_downloadExperiments(self, experimentList):
        for experiment in experimentList:
            print(experiment + ':')
            try:
                with xnat.connect(server=self.serverURL) as self.xnat_session:
                    if experiment in self.xnat_session.projects[self.projectStr].experiments.keys():
                        xnat_experiment = self.xnat_session.projects[self.projectStr].experiments[experiment]
                        self.__downloadAndRenameExperimentFolder(xnat_experiment)
                    else:
                        print('Experiment not found!')
                print(' ')
            except Exception as e:
                print(e)


    ##########################
    # Download listed scans.
    # Folder structure is: experiment>scan>image and is the same as for downloadSubjectList()
    def scanList_downloadScans(self, scanList):
        for scan in scanList:
            experimentID = scan[0]
            scanID = scan[1]
            print(experimentID + '/' + scanID + ':')
            try:
                with xnat.connect(server=self.serverURL) as self.xnat_session:
                    if experimentID in self.xnat_session.projects[self.projectStr].experiments.keys():
                        xnat_experiment = self.xnat_session.projects[self.projectStr].experiments[experimentID]
                        if scanID in xnat_experiment.scans.data.keys():
                            self.__downloadAndRenameExperimentFolder(xnat_experiment, scanID)
                        else:
                            print('Scan not found!')
                    else:
                        print('Experiment not found!')
                print(' ')
            except Exception as e:
                print(e)


    ##########################
    def getExperimentList_Project(self):
        experimentList = []
        with xnat.connect(server=self.serverURL) as xnat_session:
            xnat_experiments = xnat_session.projects[self.projectStr].experiments
            for xnat_experiment in xnat_experiments.values():
                experimentList.append(xnat_experiment.label)
        return experimentList


    ##########################
    def getSubjectList_Project(self):

        # Get list of all subjects in project.
        subjectList = []
        with xnat.connect(server=self.serverURL) as xnat_session:
            xnat_subjects = xnat_session.projects[self.projectStr].subjects
            for xnat_subject in xnat_subjects.values():
                subjectList.append(xnat_subject.label)
        subjectList.sort()
        return subjectList


    ##########################
    def subjectList_downloadAssessors(self, subjectList):

        # For all listed subjects, download all scans from associated experiments
        # Folder structure is: experiment>scan>image, but the experiment folder name has the subject label prepended.
        # This is so that the resulting folder structure is the same whether we download subjects or experiments.

        for subject in subjectList:
            print(subject + ':')
            try:
                with xnat.connect(server=self.serverURL) as self.xnat_session:
                    if subject in self.xnat_session.projects[self.projectStr].subjects.keys():
                        for xnat_experiment in self.xnat_session.projects[self.projectStr].subjects[subject].experiments.values():
                            print('   ' + xnat_experiment.label)
                            self.__downloadAndMoveAssessors(xnat_experiment)
                    else:
                        print('Subject not found!')
                    print(' ')
            except Exception as e:
                print(e)


    ##########################
    def subjectList_downloadExperiments(self, subjectList):

        # For all listed subjects, download all scans from associated experiments
        # Folder structure is: experiment>scan>image, but the experiment folder name has the subject label prepended.
        # This is so that the resulting folder structure is the same whether we download subjects or experiments.

        for subject in subjectList:
            print(subject + ':')
            try:
                with xnat.connect(server=self.serverURL) as self.xnat_session:
                    if subject in self.xnat_session.projects[self.projectStr].subjects.keys():
                        for xnat_experiment in self.xnat_session.projects[self.projectStr].subjects[subject].experiments.values():
                            print('   ' + xnat_experiment.label)
                            self.__downloadAndRenameExperimentFolder(xnat_experiment)
                    else:
                        print('Subject not found!')
                print(' ')
            except Exception as e:
                print(e)


    ##########################
    def downloadImagesReferencedByAssessors(self, keepEntireScan=False):
        # This method downloads the whole series referenced by the assessors then deletes any non-referenced images.
        # This is useful mainly to save HD space.
        # Files are named to indicate their contents, and are stored flat in the indicated download folder
        assessorFiles = glob.glob(os.path.join(self.downloadPath, 'assessors', '*.*'))
        assessorFiles.sort()
        if not keepEntireScan:
            refImageDownloadPath = os.path.join(self.downloadPath,'referencedImages')
            if not os.path.exists(refImageDownloadPath):
                os.mkdir(refImageDownloadPath)
        for assessorFile in assessorFiles:
            print('Downloading images referenced by ' + os.path.split(assessorFile)[1])
            xnat_labels = myStrSplit(os.path.split(assessorFile)[1])
            references = self.__getReferencedUIDsAndLabels(assessorFile)
            if len(references["referencedSopInstances"])==0:
                print('No roi objects found matching label "' + self.roiLabelFilter + '"')
                print(' ')
                continue
            try:
                with xnat.connect(server=self.serverURL) as self.xnat_session:
                    if keepEntireScan:
                        xnat_project = self.xnat_session.projects[self.projectStr]
                        xnat_experiments = xnat_project.experiments
                        xnat_experiment = xnat_experiments[xnat_labels[1]]
                        self.__downloadAndRenameExperimentFolder(xnat_experiment, scanID=xnat_labels[2], destinFolder='referencedScans')
                    else:
                        xnat_scan = self.xnat_session.projects[self.projectStr].experiments[xnat_labels[1]].scans[xnat_labels[2]]

                        beforeContents = os.listdir(refImageDownloadPath)
                        # uri of xml file containing scan info
                        # self.xnat_session._original_uri + xnat_scan.uri + '/files?format=xml'
                        xnat_scan.download_dir(refImageDownloadPath)
                        afterContents = os.listdir(refImageDownloadPath)
                        thisFolder = list(set(afterContents) - set(beforeContents))
                        if len(thisFolder) != 1:
                            raise Exception("Scan download folder not found / multiple folders!")
                        # sometimes dicom files have no extension, so just get all storage objects
                        stored = glob.glob(os.path.join(refImageDownloadPath, thisFolder[0], '**', '*'), recursive=True)
                        dcmCount = 0
                        uidList = [x["ReferencedSOPInstanceUID"] for x in references["referencedSopInstances"]]
                        for item in stored:
                            if os.path.isdir(item):
                                continue
                            if not ((os.path.splitext(item)[1] == '') or (os.path.splitext(item)[1] == '.dcm')):
                                continue

                            if pydicom.dcmread(item).SOPInstanceUID in uidList:
                                longName = myStrJoin([xnat_labels[0], xnat_labels[1], xnat_labels[2], os.path.split(item)[1]])
                                os.rename(item, os.path.join(refImageDownloadPath, longName))
                                dcmCount += 1
                        rmtree(os.path.join(refImageDownloadPath, thisFolder[0]))
                        refCount = len(set([x["ReferencedSOPInstanceUID"] for x in references["referencedSopInstances"]]))
                        if dcmCount >= refCount:
                            print('\033[1;32;48m' + str(dcmCount) + '/' + str(refCount) + ' referenced images saved')
                            print('\033[0;30;48m')
                        else:
                            print('\033[1;31;48m' + str(dcmCount) + '/' + str(refCount) + ' referenced images saved')
                            print('\033[0;30;48m ')
            except Exception as e:
                print(e)


    ##########################
    def __downloadAndRenameExperimentFolder(self, xnat_experiment, scanID=None, destinFolder='experiments'):

        # download experiment and find the folder that it comes in (occasionally the folder name will be something other
        # than the experiment label)
        beforeContents = os.listdir(self.downloadPath)
        if scanID is None:
            xnat_experiment.download_dir(self.downloadPath)
        else:
            # if downloading one scan, build the path to where it will be found and check it's not already there
            scanType = compile(r"[^a-zA-Z0-9]").sub("_", xnat_experiment.scans.data[scanID].data["type"])
            scanDestinFolder = os.path.join(self.downloadPath,
                                      destinFolder,
                                      myStrJoin([xnat_experiment.subject.label, xnat_experiment.label]),
                                      'scans',
                                      scanID + "-" + scanType)
            if os.path.exists(scanDestinFolder):
                raise Exception("Scan already downloaded!\n")
            else:
                xnat_experiment.scans.data[scanID].download_dir(self.downloadPath)
        afterContents = os.listdir(self.downloadPath)
        thisFolder = list(set(afterContents) - set(beforeContents))

        if len(thisFolder)>1:
            raise Exception("More than one folder downloaded!")
        if len(thisFolder)==0:
            raise Exception("Downloaded folder not found!")

        # move/rename experiment folder to include the subject label as well
        if not os.path.exists(os.path.join(self.downloadPath, destinFolder)):
            os.mkdir(os.path.join(self.downloadPath, destinFolder))
        currentFolder = os.path.join(self.downloadPath, thisFolder[0])
        newFolder = os.path.join(self.downloadPath, destinFolder, myStrJoin([xnat_experiment.subject.label, xnat_experiment.label]))
        if os.path.exists(os.path.join(newFolder, 'scans')):
            # if newFolder already exists then check that the new scans aren't already there before moving them
            newFolderScans = os.listdir(os.path.join(newFolder, 'scans'))
            currentFolderScans = os.listdir(os.path.join(currentFolder, 'scans'))
            if len(set(currentFolderScans).intersection(set(newFolderScans))) != 0:
                raise Exception("Scan already downloaded!")
            for folder in currentFolderScans:
                os.rename(os.path.join(currentFolder, 'scans', folder), os.path.join(newFolder, 'scans', folder))
            # tidy up
            os.rmdir(os.path.join(currentFolder, 'scans'))
            os.rmdir(currentFolder)
        else:
            os.rename(currentFolder, newFolder)

        if self.removeSecondaryAndSnapshots:
            for badFolder in glob.glob(os.path.join(newFolder, "**", "secondary"), recursive=True):
                rmtree(badFolder)
            for badFolder in glob.glob(os.path.join(newFolder, "**", "SNAPSHOTS"), recursive=True):
                rmtree(badFolder)

        print('Complete')



    ##########################
    def __downloadAndMoveAssessors(self, xnat_experiment, segmentLabel=None):

        if len(xnat_experiment.assessors) > 0:
            scanDict = {}
            for xnat_scan in xnat_experiment.scans.values():
                scanDict[xnat_scan.uid] = xnat_scan
            for xnat_assessor in xnat_experiment.assessors.values():

                # skip if doesn't match requested collectionType
                if self.assessorStyle["type"] != xnat_assessor.data["collectionType"]:
                    continue

                # download and unzip
                xnat_assessor.download(self.downloadPathZip, verbose=False)
                with zipfile.ZipFile(self.downloadPathZip, 'r') as zip_ref:
                    zip_ref.extractall(self.downloadPathUnzip)
                os.remove(self.downloadPathZip)

                # file is buried in lots of folders
                thisFile = glob.glob(os.path.join(self.downloadPathUnzip, '**', '*.'+self.assessorStyle["format"].lower()), recursive=True) + \
                           glob.glob(os.path.join(self.downloadPathUnzip, '**', '*.'+self.assessorStyle["format"].upper()), recursive=True)

                if len(thisFile) > 1:
                    raise Exception("More than one .dcm or .xml file in downloaded assessor!")
                if len(thisFile) == 0:
                    raise Exception("Cannot find .dcm or .xml file in downloaded assessor!")

                references = self.__getReferencedUIDsAndLabels(thisFile[0])

                # tidy up and skip if format doesn't match requested
                thisExt = os.path.splitext(thisFile[0])[1]
                if (thisExt.lower() != '.'+self.assessorStyle["format"].lower()) or (self.roiCollectionLabelFilter.lower() not in references["roiCollectionLabel"].lower()):
                    os.remove(thisFile[0])
                    rmtree(self.downloadPathUnzip)
                    continue

                print('      ' + xnat_assessor.label)
                print('        ' + references["roiCollectionLabel"])
                references = self.__getReferencedUIDsAndLabels(thisFile[0])
                scanID = scanDict[references["referencedSeriesUID"]].id
                # move assessor file and rename it
                if not os.path.exists(os.path.join(self.downloadPath, 'assessors')):
                    os.mkdir(os.path.join(self.downloadPath, 'assessors'))
                assessorFileName = os.path.join(self.downloadPath, 'assessors', myStrJoin([xnat_experiment.subject.label, xnat_experiment.label, scanID, xnat_assessor.label]) + thisExt)
                # check if file already downloaded, and also check the assessor has the same UID
                # if the UID is the same then don't copy into destination folder
                # if UID is different then move new assessor into special folder
                if os.path.exists(assessorFileName):
                    refOld = self. __getReferencedUIDsAndLabels(assessorFileName)
                    refNew = self. __getReferencedUIDsAndLabels(thisFile[0])
                    if refOld["annotationUID"] == refNew["annotationUID"]:
                        print('      Assessor already downloaded')
                        os.remove(thisFile[0])
                    else:
                        print('      Assessor with matching name, but different UID already downloaded - look in ../assessorsCollision')
                        if not os.path.exists(os.path.join(self.downloadPath, 'assessorsCollision')):
                            os.mkdir(os.path.join(self.downloadPath, 'assessorsCollision'))
                        os.rename(thisFile[0], assessorFileName.replace("assessors", "assessorsCollision"))
                else:
                    os.rename(thisFile[0], assessorFileName)
                rmtree(self.downloadPathUnzip)
        else:
            print('      ----- No Assessors found -----')


    ##########################
    def __getReferencedUIDsAndLabels(self, assessorFileName):

        if self.assessorStyle['format'].lower() == 'dcm':
            references = self.__getReferencedUIDsAndLabelsDicom(assessorFileName)
            
        elif self.assessorStyle['format'].lower() == 'xml':
            references = self.__getReferencedUIDsAndLabelsAimXml(assessorFileName)
            
        # select segments matching segmentLabel input
        if self.roiLabelFilter is not None:
            indToKeep = [x["label"] == self.roiLabelFilter for x in references["referencedSopInstances"]]
            if not any(indToKeep):
                references["referencedSopInstances"] = []
            else:
                references["referencedSopInstances"] = list(compress(references["referencedSopInstances"], indToKeep))
        return references


    ##########################
    def __getReferencedUIDsAndLabelsDicom(self, assessorFileName):

        dcm = pydicom.dcmread(assessorFileName)
        if dcm.SOPClassUID == '1.2.840.10008.5.1.4.1.1.481.3':
            # Dicom RT
            # check only one item in each level of hierarchy going down to ReferencedSeriesUID
            if len(dcm.ReferencedFrameOfReferenceSequence) != 1:
                raise Exception("DICOM RT file referencing more than one frame of reference not supported!")
            rfors = dcm.ReferencedFrameOfReferenceSequence[0]

            if len(rfors.RTReferencedStudySequence) != 1:
                raise Exception("DICOM RT file referencing more than one study not supported!")
            rtrss = rfors.RTReferencedStudySequence[0]

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
            return {"referencedSeriesUID":rtrss.RTReferencedSeriesSequence[0].SeriesInstanceUID,
                    "referencedSopInstances":annotationObjectList,
                    "roiCollectionLabel": dcm.StructureSetLabel}

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
            return {"referencedSeriesUID": dcm.ReferencedSeriesSequence[0].SeriesInstanceUID, 
                    "referencedSopInstances": annotationObjectList,
                    "roiCollectionLabel": dcm.SeriesDescription,
                    "annotationUID": dcm.SopInstanceUID}

    ##########################
    def __getReferencedUIDsAndLabelsAimXml(self, assessorFileName):
        
        xDOM = minidom.parse(assessorFileName)

        # assume only one series is referenced
        self.ReferencedSeriesUID = xDOM.getElementsByTagName('imageSeries').item(0).getElementsByTagName('instanceUid').item(0).getAttribute('root')

        # get description node whose parent is an ImageAnnotationCollectionNode
        description = [x for x in xDOM.getElementsByTagName('description') if x.parentNode.nodeName == "ImageAnnotationCollection"][0].getAttribute('value')

        # get uniqueIdentifier node whose parent is an ImageAnnotationCollectionNode
        annotationUID = [x for x in xDOM.getElementsByTagName('uniqueIdentifier') if x.parentNode.nodeName == "ImageAnnotationCollection"][0].getAttribute('root')

        annotationObjectList = []
        for xImAnn in xDOM.getElementsByTagName('ImageAnnotation'):
            label = xImAnn.getElementsByTagName('name').item(0).getAttribute('value')
            for me in xImAnn.getElementsByTagName('MarkupEntity'):
                annotationObjectList.append({"ReferencedSOPInstanceUID": me.getElementsByTagName(
                    'imageReferenceUid').item(0).getAttribute('root'),
                                             "label": label})
        return {"referencedSeriesUID": self.ReferencedSeriesUID,
                "referencedSopInstances": annotationObjectList,
                "roiCollectionLabel": description,
                "annotationUID":annotationUID}
