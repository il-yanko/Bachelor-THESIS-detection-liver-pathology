#!/usr/bin/env python

# This module is used for work data uploading and processing
import pandas as pd
import numpy as np
from data_reader import ImgReader
# This module is used for access to files in the system
import os.path
# This module is used for converting from PNG to NRRD needed for radiomics
import nrrd
# This module is used for interaction with pyradiomics
from radiomics import featureextractor


# Create target directory if don't exist
def create_directory(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)
        print("Directory ", directory, " Created ")
    else:
        print("Directory ", directory, " already exists")


sl = "/"
folderNames = ["norm", "auh", "hpb", "hpc", "wls"]
# add "dsh" to add dysholia (almost like a norm)
folderNumber = len(folderNames)

# Save all PNG-s as NRRD-s
'''
# Declare the source of the data
data = dict()
for i in range(folderNumber):
    path = os.getcwd() + "/data/png/" + folderNames[i] + sl
    data[folderNames[i]] = ImgReader.read_directory(path, "png")


for i in range (folderNumber):
    size = len(data[folderNames[i]])
    for j in range (size):
        image = data[folderNames[i]][j]

        # Add 1 additional axis for future Radiomics processing
        image = image[..., np.newaxis]
        label = np.ones(shape=image.shape)

        # Declare the destination of the data
        folder = "data/nrrd/" + folderNames[i]
        create_directory(folder)
        name_image = folderNames[i] + "_image_" + str(j) + ".nrrd"
        name_label = folderNames[i] + "_label_" + str(j) + ".nrrd"
        image_path_to = os.getcwd() + sl + folder + sl + name_image
        label_path_to = os.getcwd() + sl + folder + sl + name_label

        # Save the image as NRRD
        nrrd.write(image_path_to, image)
        nrrd.write(label_path_to, label)
'''

# Alternative for 1 image
'''
# Declare the destination of the data
folder = "/data/bmp/" + "auh" + "/"
name = "2" + ".bmp"
path_from = os.getcwd() + folder + name

# Load our 2D image
image = cv2.imread(path_from)
image = np.array(rgb_to_gray(image))

# Add 1 additional axis for future Radiomics processing
image = image[..., np.newaxis]
label = np.ones(shape=image.shape)

# Declare the source of the data
folder = "data/nrrd/" + "auh"
name_image = "auh" + "_image_" + "2" + ".nrrd"
name_label = "auh" + "_label_" + "2" + ".nrrd"
image_path_to = os.getcwd() + sl + folder + sl + name_image
label_path_to = os.getcwd() + sl + folder + sl + name_label

create_directory(folder)

# Save the image as NRRD
nrrd.write(image_path_to, image)
nrrd.write(label_path_to, label)
'''

data_array = list()
# Declare the source of NRRD data

for i in range(folderNumber):
    # folderNumber
    path = os.getcwd() + "/data/nrrd/" + folderNames[i] + sl
    # Calculate number of files in the folder
    fileNumber = int (len(next(os.walk(path))[2]) / 2)

    for j in range (fileNumber):
        # fileNumber
        name_image = folderNames[i] + "_image_" + str(j) + ".nrrd"
        name_label = folderNames[i] + "_label_" + str(j) + ".nrrd"
        image_path_to = path + name_image
        label_path_to = path + name_label

        # Instantiate the extractor
        extractor = featureextractor.RadiomicsFeaturesExtractor()
        extractor.disableAllFeatures()
        extractor.enableFeatureClassByName('firstorder')
        extractor.enableFeatureClassByName('glcm')
        extractor.enableFeatureClassByName('glrlm')
        extractor.enableFeatureClassByName('ngtdm')
        extractor.enableFeatureClassByName('gldm')

        #print("Extraction parameters:\n\t", extractor.settings)
        #print("Enabled filters:\n\t", extractor._enabledImagetypes)
        #print("Enabled features:\n\t", extractor._enabledFeatures)

        # result -> ordered dict
        result = extractor.execute(image_path_to, label_path_to)
        result['data_source'] = folderNames[i]
        result['diagnosis_code'] = i
        # for each special disease 0 or 1
        for k in range (folderNumber):
            column = str("is" + str(folderNames[k]))
            if folderNames[i] == folderNames[k]:
                result[column] = 1
            else:
                result[column] = 0
        data_array.append(result)

toBeDeleted = ['diagnostics_Versions_PyRadiomics', 'diagnostics_Versions_Numpy',
               'diagnostics_Versions_SimpleITK', 'diagnostics_Versions_PyWavelet',
               'diagnostics_Versions_Python','diagnostics_Configuration_Settings',
               'diagnostics_Configuration_EnabledImageTypes','diagnostics_Image-original_Hash',
               'diagnostics_Image-original_Spacing','diagnostics_Image-original_Size',
               'diagnostics_Mask-original_Hash', 'diagnostics_Mask-original_Spacing',
               'diagnostics_Mask-original_Size', 'diagnostics_Mask-original_BoundingBox',
               'diagnostics_Mask-original_VoxelNum', 'diagnostics_Mask-original_VolumeNum',
               'diagnostics_Mask-original_CenterOfMassIndex','diagnostics_Mask-original_CenterOfMass']
for index in range (len(data_array)):
    for feature in range(len(toBeDeleted)):
        del(data_array[index][toBeDeleted[feature]])
df =  pd.DataFrame(data_array)

df.to_csv(os.getcwd() + '/data/result/' + 'features.csv', sep=";",float_format=None)
