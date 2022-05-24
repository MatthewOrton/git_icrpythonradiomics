import os, glob
import pydicom
import progressbar
import pickle
import warnings

# Make dictionary to find file from SOPInstanceUID and folder from Series UIDS and Study UIDs.
# Assumes image files from any given series are in the same folder.
# Assumes image files from any given study are in the same folder at some level, but does not assume anything about sub-folders for series.
# Checks that the study folders found only contain images for that study and warns if not.

# save file and recompute if necessary
# if sopClassUid is input then it will also output a list of sopInstances that match this sopClassUid
def dicomFinderDict(path):

    # dictionaries to translate dicom UIDs into file or folder locations
    sopInstanceDict = {}
    seriesInstanceDict = {}
    studyInstanceDictInit = {}
    # dictionary from filename so we don't need to read the files twice to get some of the dicom metadata
    fileDict = {}

    quickLoad = path + '_dicomFinderDict.pickle'
    if os.path.isfile(quickLoad):
        with open(quickLoad, 'rb') as f:
            print('Loading existing dictionary for folder '+ path)
            unpickle = pickle.load(f)
            sopInstanceDict = unpickle['sopInstanceDict']
            seriesInstanceDict = unpickle['seriesInstanceDict']
            studyInstanceDictInit = unpickle['studyInstanceDictInit']
            studyInstanceDict = unpickle['studyInstanceDict']
            fileDict = unpickle['fileDict']

    # make list of files in current path that aren't in the pre-computed dictionary
    files = glob.glob(os.path.join(path, '**'), recursive=True)
    files = [x for x in files if not os.path.isdir(x)]
    files = list(set(files) - set(fileDict.keys()))

    # Delete any non-dicom files from file list
    # Only do this here if there is a quickLoad file as we will do this in the main loop below
    if os.path.isfile(quickLoad):
        files = [x for x in files if pydicom.misc.is_dicom(x)]



    if len(files)>0:
        print('Updating SOPInstanceUID dictionary:')
        with progressbar.ProgressBar(max_value=len(files)) as bar:
            for n, file in enumerate(files):
                if pydicom.misc.is_dicom(file):

                    dcm = pydicom.dcmread(file)

                    # dictionary from filename so we don't need to read the files twice to get some of the dicom metadata
                    # may not need the SopInstUID and SeriesUID, but store them anyway in case we develop this function further
                    fileDict[file] = {'SopInstUID':dcm.SOPInstanceUID, 'SeriesUID':dcm.SeriesInstanceUID, 'StudyUID':dcm.StudyInstanceUID}

                    # N.B. we assume files from any series are in the same folder
                    seriesFolder = os.path.split(file)[0]

                    # add SOPInstanceUID to dictionary, and warn if it is already there
                    if dcm.SOPInstanceUID in sopInstanceDict.keys():
                        warnings.warn('SOPInstanceUID already found!\n' + '    Existing    : ' + sopInstanceDict[dcm.SOPInstanceUID]['file'] + '\n' + '    Conflicting : ' + file)
                    else:
                        sopInstanceDict[dcm.SOPInstanceUID] = {'file':file, 'SeriesInstanceUID':dcm.SeriesInstanceUID, 'StudyInstanceUID':dcm.StudyInstanceUID}

                    # add SeriesInstaunceUID to dictionary, and check if this series is already there, but from a different folder location
                    if dcm.SeriesInstanceUID in seriesInstanceDict.keys():
                        if seriesInstanceDict[dcm.SeriesInstanceUID]['folder'] == seriesFolder:
                            seriesInstanceDict[dcm.SeriesInstanceUID]['SOPInstanceList'].append(dcm.SOPInstanceUID)
                        else:
                            warnings.warn('SeriesInstanceUID found in two folders!\n' + '    Folder 1    : ' + seriesInstanceDict[dcm.SeriesInstanceUID]['folder'] + '\n' + '    Folder 2    : ' + seriesFolder)
                    else:
                        seriesInstanceDict[dcm.SeriesInstanceUID] = {'folder':seriesFolder, 'StudyInstanceUID':dcm.StudyInstanceUID, 'SOPInstanceList':[dcm.SOPInstanceUID]}

                    # Different folder structures may exist for studies: (i) each study folder has sub-folders for each series, (ii) each study folder contains all images for that study without being separated into series sub-folders.
                    # Make a list of all the seriesFolders, then work out the studyFolder location by processing these lists when we've made them all
                    if dcm.StudyInstanceUID in studyInstanceDictInit.keys():
                        studyInstanceDictInit[dcm.StudyInstanceUID].append(seriesFolder)
                    else:
                        studyInstanceDictInit[dcm.StudyInstanceUID] = [seriesFolder]

                bar.update(n)

        # process studyInstanceDictInit to get the common parts of the folder names
        studyInstanceDict = {}
        for key, values in studyInstanceDictInit.items():
            studyFolder = values[0]
            for value in values:
                studyFolderMatch = os.path.commonprefix([studyFolder, value])
                # check that the matching prefix strings actually correspond to a folder location, i.e. the next character is a folder separator
                if len(value) > len(studyFolderMatch) and value[len(studyFolderMatch)] != os.sep:
                    studyFolder = os.path.split(value[0:len(studyFolderMatch)])[0]
                else:
                    studyFolder = studyFolderMatch
            studyInstanceDict[key] = studyFolder
            # check that all the files in this folder are from the same study
            studyFiles = glob.glob(os.path.join(studyFolder, '**'), recursive=True)
            for file in studyFiles:
                if not os.path.isdir(file) and fileDict[file]['StudyUID'] != key:
                    warnings.warn('Two folders with matching StudyInstanceUIDs found - please investigate! (have not implemented code to find which folders these might be in)\n    StudyInstanceUID = ' + key)

        if len(sopInstanceDict)>0:
            with open(quickLoad, 'wb') as f:
                pickle.dump({'sopInstanceDict':sopInstanceDict,
                             'seriesInstanceDict':seriesInstanceDict,
                             'studyInstanceDictInit':studyInstanceDictInit,
                             'studyInstanceDict':studyInstanceDict,
                             'fileDict':fileDict}, f)

    return sopInstanceDict, seriesInstanceDict, studyInstanceDict