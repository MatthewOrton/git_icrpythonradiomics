import os, glob
import pydicom
import progressbar
import pickle

# make dictionary to find image from its SOPInstanceUID
# save file and recompute if necessary
# if sopClassUid is input then it will also output a list of sopInstances that match this sopClassUid
def getSopInstDict(path):

    sopInstDict = {}
    sopInst2instanceNumberDict = {}
    instanceNumberDict = {}             # initially this was just a dictionary with InstanceNumber as the keys, but this (obviously!) doesn't work if there are >1 series in path
                                        # I have re-done so that instanceNumberDict is a dictionary of dictionaries - the outer dictionary has SeriesInstanceUID as keys, and the inner dictionary has InstanceNumber as the keys
    sopInstToSopClassUidDict = {}

    quickLoad = path + '_sopInstDict.pickle'
    if os.path.isfile(quickLoad):
        with open(quickLoad, 'rb') as f:
            unpickle = pickle.load(f)
            sopInstDict = unpickle['sopInstDict']
            sopInstToSopClassUidDict = unpickle['sopInstToSopClassUidDict']
            instanceNumberDict = unpickle['instanceNumberDict']
            sopInst2instanceNumberDict = unpickle['sopInst2instanceNumberDict']

    # make list of files that aren't in the pre-computed dictionary
    files = list(set(glob.glob(os.path.join(path, '**'), recursive=True)) - set(list(sopInstDict.values())))

    if len(files)>0:
        print('Updating SOPInstanceUID dictionary:')
        with progressbar.ProgressBar(max_value=len(files)) as bar:
            for n, imageFile in enumerate(files):
                if not os.path.isdir(imageFile) and pydicom.misc.is_dicom(imageFile):
                    dcm = pydicom.dcmread(imageFile)

                    sopInstDict[str(dcm.SOPInstanceUID)] = imageFile
                    sopInstToSopClassUidDict[str(dcm.SOPInstanceUID)] = str(dcm.SOPClassUID)
                    # get fields that can may be useful when search/sort/finding files
                    if hasattr(dcm,'InstanceNumber'):
                        thisInstanceNumber = int(dcm.InstanceNumber)
                        sopInst2instanceNumberDict[str(dcm.SOPInstanceUID)] = thisInstanceNumber
                        thisDict = {'SOPInstanceUID':str(dcm.SOPInstanceUID)}
                        if hasattr(dcm,'AcquisitionNumber') and dcm.AcquisitionNumber:
                            thisDict['AcquisitionNumber'] = int(dcm.AcquisitionNumber)
                        else:
                            thisDict['AcquisitionNumber'] = int(-1)
                        if hasattr(dcm,'SliceLocation'):
                            thisDict['SliceLocation'] = dcm.SliceLocation
                        if dcm.SeriesInstanceUID in instanceNumberDict:
                            if int(dcm.InstanceNumber) in instanceNumberDict[dcm.SeriesInstanceUID].keys():
                                raise Exception('InstanceNumber already found in SeriesInstanceUID dictionary!')
                            else:
                                instanceNumberDict[dcm.SeriesInstanceUID][thisInstanceNumber] = thisDict
                        else:
                            instanceNumberDict[dcm.SeriesInstanceUID] = {thisInstanceNumber:thisDict}
                bar.update(n)
        print('Complete')
        print(' ')

    # remove from dictionary any files that aren't found on file system
    filesNotPresent = list(set(list(sopInstDict.values())) - set(glob.glob(os.path.join(path, '**'), recursive=True)))
    sopInstDict = {key: value for key, value in sopInstDict.items() if value not in filesNotPresent}

    if len(sopInstDict)>0:
        with open(quickLoad, 'wb') as f:
            pickle.dump({'sopInstDict':sopInstDict, 'sopInstToSopClassUidDict':sopInstToSopClassUidDict, 'instanceNumberDict':instanceNumberDict, 'sopInst2instanceNumberDict':sopInst2instanceNumberDict}, f)

    return sopInstDict, sopInstToSopClassUidDict, instanceNumberDict, sopInst2instanceNumberDict