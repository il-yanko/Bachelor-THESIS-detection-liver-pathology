#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#from sklearn.ensemble import RandomForestClassfier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn import linear_model

path = "data/result/features.csv"
chosen = [#"original_glrlm_RunEntropy",
          #"original_glrlm_GrayLevelNonUniformity",
          "original_glrlm_LongRunHighGrayLevelEmphasis",
          "original_glrlm_RunLengthNonUniformity"
          ]
          #,"diagnosis_code"]
#data = pd.read_csv(path, ";")
# visualize SEABORN
'''
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
    #hue='data_source', 'isnorm'
    hue='data_source',
    aspect=0.3,
    palette="Paired",
    #kind="skatter"
    #markers="."
)
plt.show()
plt.tight_layout()
'''


# Make Logistic Regression (Logit) and test it
test_path = "data/result/test.csv"
test = pd.read_csv(test_path, ";")
train_path = "data/result/train.csv"
train = pd.read_csv(train_path, ";")
X = train[chosen]
y = train["isnorm"]

# Fit the classifier
clf = linear_model.LogisticRegression(C=1e5, solver='lbfgs')
clf.fit(X, y)
# Test the classifier
clf.predict_proba(test[chosen])
print(clf.score(X, y))

