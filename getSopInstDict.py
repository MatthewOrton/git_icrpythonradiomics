import os, glob
import pydicom
import progressbar
import pickle

# make dictionary to find image from its SOPInstanceUID
# save file and recompute if necessary
def getSopInstDict(path):

    sopInstDict = {}

    quickLoad = path + '_sopInstDict.pickle'
    if os.path.isfile(quickLoad):
        with open(quickLoad, 'rb') as f:
            sopInstDict = pickle.load(f)

    # make list of files that aren't in the pre-computed dictionary
    imageFiles = list(set(glob.glob(os.path.join(path, '**'), recursive=True)) - set(list(sopInstDict.values())))

    if len(imageFiles)>0:
        print('Updating SOPInstanceUID dictionary:')
        with progressbar.ProgressBar(max_value=len(imageFiles)) as bar:
            for n, imageFile in enumerate(imageFiles):
                if not os.path.isdir(imageFile) and pydicom.misc.is_dicom(imageFile):
                    sopInstDict[pydicom.dcmread(imageFile).SOPInstanceUID] = imageFile
                bar.update(n)
        print('Complete')
        print(' ')

    # remove from dictionary any files that aren't found on file system
    filesNotPresent = list(set(list(sopInstDict.values())) - set(glob.glob(os.path.join(path, '**'), recursive=True)))
    sopInstDict = {key: value for key, value in sopInstDict.items() if value not in filesNotPresent}

    if len(sopInstDict)>0:
        with open(quickLoad, 'wb') as f:
            pickle.dump(sopInstDict, f)

    return sopInstDict