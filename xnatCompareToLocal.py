import os
import xnat
import glob
import pydicom

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
    def compareLocalToXnat(self):
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
                scanName = str(dcm.SeriesNumber)
                with xnat.connect(server=self.serverURL) as xnat_session:
                    xnat_subjects = xnat_session.projects[self.projectName].subjects
                    if subjectName in xnat_subjects:
                        xnat_experiments = xnat_subjects[subjectName].experiments
                        if experimentName in xnat_experiments:
                            xnat_scans = xnat_experiments[experimentName].scans
                            if scanName in xnat_scans:
                                xnatFiles = xnat_scans[scanName].resources['DICOM'].files
                                thisStr = subjectName +'/'+ experimentName +'/'+ scanName
                                if len(localFiles) == len(xnatFiles):
                                    resultStr = 'OK                   '
                                else:
                                    resultStr = 'ERROR                '
                                    errorCount += 1
                                print(resultStr + thisStr)
                            else:
                                print('Scan not found       ' + subjectName +'/'+ experimentName +'/'+ scanName)
                                errorCount += 1
                        else:
                            print('Experiment not found ' + subjectName +'/'+ experimentName)
                            errorCount += 1
                    else:
                        print('Subject not found    ' + subjectName)
                        errorCount += 1
        print(' ')
        print('Number of errors = '+str(errorCount))

    ##########################
    def compareXnatToLocal(self, nameFilter):
        with xnat.connect(server=self.serverURL) as xnat_session:
            xnat_subjects = xnat_session.projects[self.projectName].subjects
            for xnat_subject in xnat_subjects.values():
                xnat_experiments = xnat_subject.experiments
                for xnat_experiment in xnat_experiments.values():
                    xnat_assessors = xnat_experiment.assessors
                    for xnat_assessor in xnat_assessors.values():
                        if nameFilter in xnat_assessor.name:
                            if len(glob.glob(os.path.join(self.localFolder, '*'+xnat_assessor.label+'*.*')))==0:
                                print(xnat_subject.label + ' // ' + xnat_experiment.label + ' // ' + xnat_assessor.label+ ' // ' + xnat_assessor.name)