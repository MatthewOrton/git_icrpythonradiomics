from radiomics.featureextractor import RadiomicsFeatureExtractor
import collections, six, logging
from radiomics import generalinfo, imageoperations, getFeatureClasses, setVerbosity
from itertools import chain

geometryTolerance = None
logger = logging.getLogger(__name__)

class radiomicsFeatureExtractorEnhanced(RadiomicsFeatureExtractor):

    def setVerbosity(self, level):
        setVerbosity(level)

    def getProbabilityMatrices(self, imageFilepath, maskFilepath, label=None, label_channel=None, voxelBased=False):
        """
        Compute radiomics signature for provide image and mask combination. It comprises of the following steps:

        1. Image and mask are loaded and normalized/resampled if necessary.
        2. Validity of ROI is checked using :py:func:`~imageoperations.checkMask`, which also computes and returns the
           bounding box.
        3. If enabled, provenance information is calculated and stored as part of the result. (Not available in voxel-based
           extraction)
        4. Shape features are calculated on a cropped (no padding) version of the original image. (Not available in
           voxel-based extraction)
        5. If enabled, resegment the mask based upon the range specified in ``resegmentRange`` (default None: resegmentation
           disabled).
        6. Other enabled feature classes are calculated using all specified image types in ``_enabledImageTypes``. Images
           are cropped to tumor mask (no padding) after application of any filter and before being passed to the feature
           class.
        7. The calculated features is returned as ``collections.OrderedDict``.

        :param imageFilepath: SimpleITK Image, or string pointing to image file location
        :param maskFilepath: SimpleITK Image, or string pointing to labelmap file location
        :param label: Integer, value of the label for which to extract features. If not specified, last specified label
            is used. Default label is 1.
        :param label_channel: Integer, index of the channel to use when maskFilepath yields a SimpleITK.Image with a vector
            pixel type. Default index is 0.
        :param voxelBased: Boolean, default False. If set to true, a voxel-based extraction is performed, segment-based
            otherwise.
        :returns: dictionary containing calculated signature ("<imageType>_<featureClass>_<featureName>":value).
            In case of segment-based extraction, value type for features is float, if voxel-based, type is SimpleITK.Image.
            Type of diagnostic features differs, but can always be represented as a string.
        """
        global geometryTolerance, logger
        tolerance = self.settings.get('geometryTolerance')
        additionalInfo = self.settings.get('additionalInfo', False)
        resegmentShape = self.settings.get('resegmentShape', False)

        if label is not None:
            self.settings['label'] = label
        else:
            label = self.settings.get('label', 1)

        if label_channel is not None:
            self.settings['label_channel'] = label_channel

        if geometryTolerance != tolerance:
            self._setTolerance()

        if additionalInfo:
            generalInfo = generalinfo.GeneralInfo()
            generalInfo.addGeneralSettings(self.settings)
            generalInfo.addEnabledImageTypes(self.enabledImagetypes)
        else:
            generalInfo = None

        if voxelBased:
            self.settings['voxelBased'] = True
            kernelRadius = self.settings.get('kernelRadius', 1)
            logger.info('Starting voxel based extraction')
        else:
            kernelRadius = 0

        logger.info('Calculating features with label: %d', label)
        logger.debug('Enabled images types: %s', self.enabledImagetypes)
        logger.debug('Enabled features: %s', self.enabledFeatures)
        logger.debug('Current settings: %s', self.settings)

        # 1. Load the image and mask
        probabilityMatrices = collections.OrderedDict()
        image, mask = self.loadImage(imageFilepath, maskFilepath, generalInfo)

        # 2. Check whether loaded mask contains a valid ROI for feature extraction and get bounding box
        # Raises a ValueError if the ROI is invalid
        boundingBox, correctedMask = imageoperations.checkMask(image, mask, **self.settings)

        # Update the mask if it had to be resampled
        if correctedMask is not None:
            if generalInfo is not None:
                generalInfo.addMaskElements(image, correctedMask, label, 'corrected')
            mask = correctedMask

        logger.debug('Image and Mask loaded and valid, starting extraction')

        # 5. Resegment the mask if enabled (parameter regsegmentMask is not None)
        resegmentedMask = None
        if self.settings.get('resegmentRange', None) is not None:
            resegmentedMask = imageoperations.resegmentMask(image, mask, **self.settings)

            # Recheck to see if the mask is still valid, raises a ValueError if not
            boundingBox, correctedMask = imageoperations.checkMask(image, resegmentedMask, **self.settings)

            if generalInfo is not None:
                generalInfo.addMaskElements(image, resegmentedMask, label, 'resegmented')

        # 3. Add the additional information if enabled
        if generalInfo is not None:
            probabilityMatrices.update(generalInfo.getGeneralInfo())

        # if resegmentShape is True and resegmentation has been enabled, update the mask here to also use the
        # resegmented mask for shape calculation (e.g. PET resegmentation)
        if resegmentShape and resegmentedMask is not None:
            mask = resegmentedMask

        # (Default) Only use resegemented mask for feature classes other than shape
        # can be overridden by specifying `resegmentShape` = True
        if not resegmentShape and resegmentedMask is not None:
            mask = resegmentedMask

        # 6. Calculate other enabled feature classes using enabled image types
        # Make generators for all enabled image types
        logger.debug('Creating image type iterator')
        imageGenerators = []
        for imageType, customKwargs in six.iteritems(self.enabledImagetypes):
            args = self.settings.copy()
            args.update(customKwargs)
            logger.info('Adding image type "%s" with custom settings: %s' % (imageType, str(customKwargs)))
            imageGenerators = chain(imageGenerators,
                                    getattr(imageoperations, 'get%sImage' % imageType)(image, mask, **args))

        logger.debug('Extracting features')
        # Calculate features for all (filtered) images in the generator
        for inputImage, imageTypeName, inputKwargs in imageGenerators:
            logger.info('Calculating features for %s image', imageTypeName)
            inputImage, inputMask = imageoperations.cropToTumorMask(inputImage, mask, boundingBox,
                                                                    padDistance=kernelRadius)
            probabilityMatrices.update(self.get_PMatrix(inputImage, inputMask, imageTypeName, **inputKwargs))

        logger.debug('Features extracted')

        return probabilityMatrices


    def get_PMatrix(self, image, mask, imageTypeName, **kwargs):
        r"""
        Compute signature using image, mask and \*\*kwargs settings.

        This function computes the signature for just the passed image (original or derived), it does not pre-process or
        apply a filter to the passed image. Features / Classes to use for calculation of signature are defined in
        ``self.enabledFeatures``. See also :py:func:`enableFeaturesByName`.

        :param image: The cropped (and optionally filtered) SimpleITK.Image object representing the image used
        :param mask: The cropped SimpleITK.Image object representing the mask used
        :param imageTypeName: String specifying the filter applied to the image, or "original" if no filter was applied.
        :param kwargs: Dictionary containing the settings to use for this particular image type.
        :return: collections.OrderedDict containing the calculated features for all enabled classes.
          If no features are calculated, an empty OrderedDict will be returned.

        .. note::

          shape descriptors are independent of gray level and therefore calculated separately (handled in `execute`). In
          this function, no shape features are calculated.
        """
        global logger
        featureClasses = getFeatureClasses()

        enabledFeatures = self.enabledFeatures

        # Calculate feature classes
        featureProbabilityMatrices = {}
        for featureClassName, featureNames in six.iteritems(enabledFeatures):
            # Handle calculation of shape features separately
            if featureClassName.startswith('shape'):
                continue

            if featureClassName in featureClasses:
                logger.info('Computing %s', featureClassName)

                featureClass = featureClasses[featureClassName](image, mask, **kwargs)
                featureClass.execute()
                newFeatureClassName = '%s_%s' % (imageTypeName, featureClassName)
                if featureClassName == 'glcm':
                    featureProbabilityMatrices[newFeatureClassName] = featureClass.P_glcm
                if featureClassName == 'gldm':
                    featureProbabilityMatrices[newFeatureClassName] = featureClass.P_gldm
                if featureClassName == 'glrlm':
                    featureProbabilityMatrices[newFeatureClassName] = featureClass.P_glrlm
                if featureClassName == 'glszm':
                    featureProbabilityMatrices[newFeatureClassName] = featureClass.P_glszm
                if featureClassName == 'ngtdm':
                    featureProbabilityMatrices[newFeatureClassName] = featureClass.P_ngtdm


        return featureProbabilityMatrices
