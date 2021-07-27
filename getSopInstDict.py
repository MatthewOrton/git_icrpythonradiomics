import os, glob
import pydicom
import progressbar
import pickle

# make dictionary to find image from its SOPInstanceUID
# save file and recompute if necessary
# if sopClassUid is input then it will also output a list of sopInstances that match this sopClassUid
def getSopInstDict(path):

    sopInstDict = {}
    sopInstToSopClassUidDict = {}

    quickLoad = path + '_sopInstDict.pickle'
    if os.path.isfile(quickLoad):
        with open(quickLoad, 'rb') as f:
            unpickle = pickle.load(f)
            sopInstDict = unpickle['sopInstDict']
            sopInstToSopClassUidDict = unpickle['sopInstToSopClassUidDict']

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
                bar.update(n)
        print('Complete')
        print(' ')

    # remove from dictionary any files that aren't found on file system
    filesNotPresent = list(set(list(sopInstDict.values())) - set(glob.glob(os.path.join(path, '**'), recursive=True)))
    sopInstDict = {key: value for key, value in sopInstDict.items() if value not in filesNotPresent}

    if len(sopInstDict)>0:
        with open(quickLoad, 'wb') as f:
            pickle.dump({'sopInstDict':sopInstDict, 'sopInstToSopClassUidDict':sopInstToSopClassUidDict}, f)

    return sopInstDict, sopInstToSopClassUidDict