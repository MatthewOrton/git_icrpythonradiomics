import numpy as np
import matplotlib.pyplot as plt
import cv2

def makeThumbnails(roiList, outputFileName, imageGrayLevelLimits=[-100, 200], volumePad = [2, 20, 20]):

    # check all rois are linked to same image volume data
    match = True
    for roi in roiList:
        if roi['image']['array'].shape != roiList[0]['image']['array'].shape:
            match = False
            break
        match = match and np.all(roi['image']['array'] == roiList[0]['image']['array'])
    if not match:
        print('Non-matching image')
    else:
        imageVolume = roiList[0]['image']
        for roi in roiList:
            roi.pop('image')

    # get bounding box for all masks
    mask = roiList[0]['mask']['array']
    for roi in roiList:
        mask = np.logical_or(mask, roi['mask']['array'])
    axis0 = np.where(np.sum(mask.astype(int), axis=(1, 2))>0)[0]
    axis1 = np.where(np.sum(mask.astype(int), axis=(0, 2))>0)[0]
    axis2 = np.where(np.sum(mask.astype(int), axis=(0, 1))>0)[0]

    idx0 = range(max((0, axis0[0] - volumePad[0])), min((mask.shape[0], axis0[-1] + volumePad[0] + 1)))
    idx1 = range(max((0, axis1[0] - volumePad[1])), min((mask.shape[1], axis1[-1] + volumePad[1] + 1)))
    idx2 = range(max((0, axis2[0] - volumePad[2])), min((mask.shape[2], axis2[-1] + volumePad[2] + 1)))

    # crop image and all masks
    imageVolume = imageVolume['array'][idx0, :, :][:, idx1, :][:, :, idx2]  # this discards lots of the metadata and converts imageVolume from a dict to numpy array
    for roi in roiList:
        roi['mask'] = roi['mask']['array'][idx0,:,:][:,idx1,:][:,:,idx2]  # this discards some metadata that we don't need

    nPlt = imageVolume.shape[0]
    pltRows = int(np.round(np.sqrt(2*nPlt/3))) + 1
    pltCols = int(np.ceil(nPlt/pltRows))

    # main figure and axes
    fPlt, axarr = plt.subplots(pltRows, pltCols, gridspec_kw={'wspace': 0.02, 'hspace': 0.02})

    # dummy figure so we can use contour() to get outline of the mask
    figContourDummy, axContourDummy = plt.subplots(1,1)

    # make each image and overlay all contours on this image
    colors = ['#2882ff', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    for slice, ax in enumerate(fPlt.axes):
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if slice<imageVolume.shape[0]:
            ax.imshow(imageVolume[slice,:,:], vmin=imageGrayLevelLimits[0], vmax=imageGrayLevelLimits[1], cmap='gray', interpolation='nearest')

            for k, roi in enumerate(roiList):
                maskHere = roi['mask'][slice,:,:].astype(int)
                if np.any(maskHere > 0):
                    # tricks to get the boundary of the outside of the mask pixels using contour()
                    ff = 5
                    res = cv2.resize(maskHere, dsize=(maskHere.shape[1] * ff, maskHere.shape[0] * ff), interpolation=cv2.INTER_NEAREST)
                    cc = axContourDummy.contour(res, levels=[0.5])
                    for pp in cc.allsegs[0]:
                        pp = (pp - (ff - 1) / 2) / ff
                        pp = np.round(pp - 0.5) + 0.5
                        ax.plot(pp[:, 0], pp[:, 1], colors[k], linewidth=100/np.sqrt(nPlt)/imageVolume.shape[1])

    plt.close(figContourDummy)

    # legend goes on last axes that shouldn't have any images in it
    for k, roi in enumerate(roiList):
        ax.plot(0, 0, colors[k], label = roi['ROIName'])
    ax.legend()

    fPlt.savefig(outputFileName,  orientation='landscape', format='pdf', dpi=1200)

