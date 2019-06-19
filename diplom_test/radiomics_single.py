#!/usr/bin/env python

# This module is used for work data uploading and processing
import pandas as pd
import numpy as np
from PIL import Image
# This module is used for access to files in the system
import os.path
# This module is used for converting from PNG to NRRD needed for radiomics
import nrrd
# This module is used for interaction with pyradiomics
from radiomics import featureextractor
# This module is used for I/O of the serialized algorithms
import pickle
# This module is used for our algorithms
import sklearn
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn import linear_model,tree
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn import svm
import sklearn.utils._cython_blas
import sklearn.neighbors.typedefs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

# Create target directory if don't exist
def create_directory(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)
        print("Directory ", directory, " Created ")
    else:
        print("Directory ", directory, " already exists")

# Declare the source of the data
format = "bmp"
folderName = "tmp"
sl = "/"
img_path = os.getcwd() + "/data/" + format + sl + folderName + sl + "1." + format

def signle_prediction(path=img_path):
    image = np.asarray(Image.open(path).convert('L'), dtype=np.uint8)
    #image = ImgReader.read_directory(path, "png")

    # Add 1 additional axis for future Radiomics processing
    image = image[..., np.newaxis]
    label = np.ones(shape=image.shape)

    # Declare the destination of the data
    folder = "data/nrrd/" + folderName
    create_directory(folder)
    name_image = folderName + "_image_1.nrrd"
    name_label = folderName + "_label_1.nrrd"
    image_path_to = os.getcwd() + "/data/nrrd/" + folderName + sl + name_image
    label_path_to = os.getcwd() + "/data/nrrd/" + folderName + sl + name_label

    # Save the PNG-image as NRRD
    nrrd.write(image_path_to, image)
    nrrd.write(label_path_to, label)

    # Instantiate the extractor
    extractor = featureextractor.RadiomicsFeatureExtractor()

    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName(str('firstorder'))
    extractor.enableFeatureClassByName(str('glcm'))
    extractor.enableFeatureClassByName(str('glrlm'))
    extractor.enableFeatureClassByName(str('ngtdm'))
    extractor.enableFeatureClassByName(str('gldm'))
    extractor.enableFeatureClassByName(str('glszm'))


    #print("Extraction parameters:\n\t", extractor.settings)
    #print("Enabled filters:\n\t", extractor._enabledImagetypes)
    #print("Enabled features:\n\t", extractor._enabledFeatures)

    # result -> ordered dict
    result = extractor.execute(image_path_to, label_path_to)
    toBeDeleted = ['diagnostics_Image-original_Dimensionality', 'diagnostics_Versions_PyRadiomics',
                   'diagnostics_Versions_Numpy',
                   'diagnostics_Versions_SimpleITK', 'diagnostics_Versions_PyWavelet',
                   'diagnostics_Versions_Python', 'diagnostics_Configuration_Settings',
                   'diagnostics_Configuration_EnabledImageTypes', 'diagnostics_Image-original_Hash',
                   'diagnostics_Image-original_Spacing', 'diagnostics_Image-original_Size',
                   'diagnostics_Mask-original_Hash', 'diagnostics_Mask-original_Spacing',
                   'diagnostics_Mask-original_Size', 'diagnostics_Mask-original_BoundingBox',
                   'diagnostics_Mask-original_VoxelNum', 'diagnostics_Mask-original_VolumeNum',
                   'diagnostics_Mask-original_CenterOfMassIndex', 'diagnostics_Mask-original_CenterOfMass']
    for feature in range(len(toBeDeleted)):
        del(result[toBeDeleted[feature]])

    # Have a look at the current data
    '''
    for key,val in result.items():
        print(key,":",val)
    '''

    df =  pd.DataFrame(result, index=[0])
    df.to_csv(os.getcwd() + '/data/result/' + 'single.csv', sep=";",float_format=None)


    data = pd.read_csv("data/result/single.csv", ";")
    data = data.iloc[0:, 1:data.shape[1]]



    # load the model from disk
    model_name = 'Multi-layer Perceptron'
    accuracy = [77,83,90,86,84,85,85]
    label_ukr = [["norm", "autoimmune hepatitis", "hepatitis В", "hepatitis С", "Wilson disease", "cystic fibrosis"],
                 ['NOT Wilson disease', 'Wilson disease'], ['NOT hepatitis В', 'hepatitis В'], ['NOT hepatitis С', 'hepatitis С'],
                 ['NOT autoimmune hepatitis', 'autoimmune hepatitis'], ['pathology', 'norm'], ['NOT cystic fibrosis', 'cystic fibrosis']]

    kind_ukr = ["all diseases", "Wilson disease VS all", "hepatitis В VS all", "hepatitis С VS all",
               "autoimmune hepatitis VS all", "norm VS pathologies", "cystic fibrosis VS all"]
    poolParam = ["diagnosis_code", "iswls", "ishpb", "ishpc", "isauh", "isnorm", "iscf"]
    text = "Classifier: <b>{0}</b><br><br>\n\n".format(model_name)





    models = [0,5]
    for number in models:
        filename = 'data/result/model/' + model_name + ' ' + poolParam[number] + '.sav'
        file = open(filename, 'rb')
        loaded = pickle.load(file)
        print("Model <" + model_name + " " + poolParam[number]  + "> was loaded")

        # Test the classifier
        y_pred = int(loaded.predict(data))
        part = 'Accuracy: <b>{1}</b>%<br>\nClassification: [{0}]<br>\n[RESULTS]: <b>{2}</b><br><br>\n\n'.format(kind_ukr[number], accuracy[number],label_ukr[number][y_pred])
        text = text + part
    text = '<font size="2"> {0} </font>'.format(text)
    print(text)
    return text
#signle_predition()