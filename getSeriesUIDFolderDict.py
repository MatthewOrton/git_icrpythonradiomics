import os, glob
import pydicom
import progressbar
import pickle

# make dictionary to find image from its SOPInstanceUID
# save file and recompute if necessary
# if sopClassUid is input then it will also output a list of sopInstances that match this sopClassUid

def getSeriesUIDFolderDict(searchFolder):

    seriesFolderDict = {}

    quickLoad = searchFolder + '_seriesFolderDict.pickle'
    if os.path.isfile(quickLoad):
        with open(quickLoad, 'rb') as f:
            unpickle = pickle.load(f)
            seriesFolderDict = unpickle['seriesFolderDict']
            filesChecked = unpickle['filesChecked']
    else:
        filesChecked = []

    # make list of files that aren't in the pre-computed dictionary
    allFiles = glob.glob(os.path.join(searchFolder, '**'), recursive=True)
    files = list(set(allFiles) - set(filesChecked))

    if len(files)>0:
        print('Updating SOPInstanceUID dictionary:')
        with progressbar.ProgressBar(max_value=len(files)) as bar:
            for n, file in enumerate(files):
                if not os.path.isdir(file) and pydicom.misc.is_dicom(file):
                    dcm = pydicom.dcmread(file)

                    thisFolder = os.path.split(file)[0]

                    # add folder of current file to dictionary, and replace if it is higher up the directory tree (shorter folder path string)
                    if dcm.SeriesInstanceUID not in seriesFolderDict or len(thisFolder) < len(seriesFolderDict[dcm.SeriesInstanceUID]):
                        seriesFolderDict[dcm.SeriesInstanceUID] = thisFolder

                bar.update(n)
        print('Complete')
        print(' ')

    if len(seriesFolderDict)>0:
        with open(quickLoad, 'wb') as f:
            pickle.dump({'seriesFolderDict':seriesFolderDict, 'filesChecked':allFiles}, f)

    return seriesFolderDict