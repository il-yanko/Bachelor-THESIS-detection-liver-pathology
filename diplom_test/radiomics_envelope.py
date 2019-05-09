#!/usr/bin/env python

from radiomics import featureextractor  # This module is used for interaction with pyradiomics
import numpy as np
import os.path
import cv2
import nrrd


def rgb_to_gray(rgb):
    # scalar product of colors with certain theoretical coefficients according to the YUV system
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114]).round(3).astype(int)


# Declare the source of the data
folder = "/data/" + "auh" + "/"
name = "1.bmp"
path_from = os.getcwd() + folder + name

# Load our 2D image
image = cv2.imread(path_from)
image = np.array(rgb_to_gray(image))

# Add 1 additional axis for future Radiomics processing
image = image[..., np.newaxis]
label = np.ones(shape=image.shape)

# Declare the source of the data
folder = "/data/nrrd/" + "auh" + "/"
name_image = "auh" + "_image_" + "1" + ".nrrd"
name_label = "auh" + "_label_" + "1" + ".nrrd"
image_path = os.getcwd() + folder + name_image
label_path = os.getcwd() + folder + name_label

# Save the image as NRRD
nrrd.write(image_path, image)
nrrd.write(label_path, label)

readdata, header = nrrd.read(image_path)


# Instantiate the extractor
extractor = featureextractor.RadiomicsFeaturesExtractor()

print ("Extraction parameters:\n\t", extractor.settings)
print ("Enabled filters:\n\t", extractor._enabledImagetypes)
print ("Enabled features:\n\t", extractor._enabledFeatures)

result = extractor.execute(image_path ,label_path)

print ("Calculated features:")
for key, value in result.items():
    print ("\t", key, ":", value)