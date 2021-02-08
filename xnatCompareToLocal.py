import os
import xnat
import glob
import pydicom
from re import compile
import copy
import re

def myStrJoin(strList):
    return '__II__'.join(strList)

def myStrSplit(str):
    return str.split('__II__')

class xnatCompareToLocal:

    ##########################
    def __init__(self,
                 serverURL='',
                 projectName='',
                 localFolder=''):

        self.serverURL = serverURL
        self.projectName = projectName
        self.localFolder = localFolder

        # check that project exists
        with xnat.connect(server=self.serverURL) as xnat_session:
            if self.projectName not in xnat_session.projects.keys():
                print(' ')
                print(self.projectName+" not found on "+self.serverURL)
                exit()







    ##########################
    def compareLocalScansToXnatScans(self, seriesDescriptionRemoveThisStr = '', scanNumberExtra = None):
        errorCount = 0
        subjectExperimentFolders = glob.glob(os.path.join(self.localFolder,'*','')) # last '' will cause to select only folders not files
        subjectExperimentFolders.sort()

        for subjectExperimentFolder in subjectExperimentFolders:
            subjectName, experimentName = myStrSplit(os.path.split(os.path.split(subjectExperimentFolder)[0])[1])
            scanFolders = glob.glob(os.path.join(subjectExperimentFolder, 'scans','*'))

            for scanFolder in scanFolders:
                localFiles = glob.glob(os.path.join(scanFolder, 'resources','DICOM','files','*'))
                # read first dicom to get the seriesNumber
                dcm = pydicom.dcmread(localFiles[0])
                # modified series description to match the way xnatDownloader changes this string

                if hasattr(dcm, 'SeriesDescription') and dcm.SeriesDescription != '':
                    seriesDesc = (dcm.SeriesDescription).replace(seriesDescriptionRemoveThisStr,'')
                    if seriesDesc=='':
                        seriesDescModified = 'unknown'
                    else:
                        seriesDescModified = compile(r"[^a-zA-Z0-9]").sub("_", seriesDesc)
                else:
                    seriesDescModified = 'unknown'
                scanName = scanFolder.split('/')[-1].replace('-' + seriesDescModified, '')
                if scanNumberExtra is not None:
                    # scanNumber must be numeric, so if it isn't then just take the first run of numeric characters
                    scanName = re.split(r"\D", scanName)[0]
                    scanName = str(int(scanName) + scanNumberExtra)

                with xnat.connect(server=self.serverURL) as xnat_session:
                    xnat_subjects = xnat_session.projects[self.projectName].subjects
                    if subjectName in xnat_subjects:
                        xnat_experiments = xnat_subjects[subjectName].experiments
                        if experimentName in xnat_experiments:
                            xnat_scans = xnat_experiments[experimentName].scans
                            if scanName in xnat_scans:
                                xnatFiles = xnat_scans[scanName].resources['DICOM'].files
                                thisStr = subjectName +' // '+ experimentName +' // '+ scanName
                                if len(localFiles) == len(xnatFiles):
                                    resultStr = 'OK                   '
                                else:
                                    resultStr = 'ERROR                '
                                    errorCount += 1
                                print(resultStr + thisStr)
                            else:
                                print('Scan not found       ' + subjectName +' // '+ experimentName +' // '+ scanName)
                                errorCount += 1
                        else:
                            print('Experiment not found ' + subjectName +' // '+ experimentName)
                            errorCount += 1
                    else:
                        print('Subject not found    ' + subjectName)
                        errorCount += 1
        print(' ')
        print('Number of errors = '+str(errorCount))

    ##########################
    def compareXnatAssessorsToLocalAssessors(self, nameFilter):
        # quickly get list of subjects, then open a new connection to xnat for each as it sometimes drops the connection if you only do it once as it takes a long time
        subjectList = []
        with xnat.connect(server=self.serverURL) as xnat_session:
            xnat_subjects = xnat_session.projects[self.projectName].subjects
            for xnat_subject in xnat_subjects.values():
                subjectList.append(xnat_subject.label)
        subjectList.sort()
        for subject in subjectList:
            with xnat.connect(server=self.serverURL) as xnat_session:
                xnat_experiments = xnat_session.projects[self.projectName].subjects[subject].experiments
                for xnat_experiment in xnat_experiments.values():
                    xnat_assessors = xnat_experiment.assessors
                    for xnat_assessor in xnat_assessors.values():
                        if nameFilter in xnat_assessor.name:
                            if len(glob.glob(os.path.join(self.localFolder, '*'+xnat_assessor.label+'*.*')))==0:
                                print('No local copy   :' + subject + ' // ' + xnat_experiment.label + ' // ' + xnat_assessor.label+ ' // ' + xnat_assessor.name)
                            else:
                                print('Local copy found: ' + xnat_assessor.label+ ' // ' + xnat_assessor.name)


   ##########################
    def deleteAssessors(self, nameFilter):
        print(' ')
        print("Are you sure you want to delete all assessors")
        print("from project '" + self.projectName + "' with label")
        proceed = input("containing the string '" + nameFilter +  "' ? ")
        if proceed == 'y':
            # quickly get list of subjects, then open a new connection to xnat for each as it sometimes drops the connection if you only do it once as it takes a long time
            subjectList = []
            with xnat.connect(server=self.serverURL) as xnat_session:
                xnat_subjects = xnat_session.projects[self.projectName].subjects
                for xnat_subject in xnat_subjects.values():
                    subjectList.append(xnat_subject.label)
            subjectList.sort()
            for subject in subjectList:
                with xnat.connect(server=self.serverURL) as xnat_session:
                    xnat_experiments = xnat_session.projects[self.projectName].subjects[subject].experiments
                    for xnat_experiment in xnat_experiments.values():
                        xnat_assessors = xnat_experiment.assessors
                        for xnat_assessor in xnat_assessors.values():
                            if nameFilter in xnat_assessor.name:
                                print('Deleting  : ' + subject + ' // ' + xnat_experiment.label + ' // ' + xnat_assessor.label+ ' // ' + xnat_assessor.name)
                                xnat_assessor.delete(remove_files=True)
                            else:
                                print('Found  : ' + subject + ' // ' + xnat_experiment.label + ' // ' + xnat_assessor.label+ ' // ' + xnat_assessor.name)
