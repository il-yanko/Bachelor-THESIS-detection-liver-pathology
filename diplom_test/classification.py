#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#from sklearn.ensemble import RandomForestClassfier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split

path = "data/result/features.csv"
chosen = [#"original_glrlm_RunEntropy",
          #"original_glrlm_GrayLevelNonUniformity",
          "original_glrlm_LongRunHighGrayLevelEmphasis",
          "original_glrlm_RunLengthNonUniformity",
          "diagnosis_code"]

data = pd.read_csv(path, ";")

red_blue = ["#ff0000", "#1240ab"]
sns.pairplot(
    data,
    vars=[
        #'original_glrlm_RunEntropy',
        #'original_glrlm_GrayLevelNonUniformity',
        #'original_glrlm_GrayLevelVariance',
        'original_glrlm_LongRunHighGrayLevelEmphasis',
        #'original_glrlm_LongRunLowGrayLevelEmphasis',
        'original_glrlm_RunLengthNonUniformity',
    ],
    #hue='data_source',
    hue='isnorm',
    aspect=0.3,
    palette=red_blue,
    #kind="skatter"
    #markers="."
)

plt.show()
plt.tight_layout()






'''
#X_train,y_train,X_test,y_test = train_test_split(data,test_size=0.3)
#print("% is X_train, % is y_train",X_train,y_train)
'''




