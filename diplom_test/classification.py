#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
#from sklearn.ensemble import RandomForestClassfier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier, plot_importance

path = "data/result/features.csv"
chosen = ["original_glrlm_RunEntropy",
          "original_glrlm_GrayLevelNonUniformity",
          "original_firstorder_10Percentile",
          "original_gldm_GrayLevelNonUniformity",
          "diagnostics_Image-original_Mean"
          ]
          #,"diagnosis_code"]
chosen1 = ["original_glcm_Imc1",
          "original_glcm_ClusterShade",
          "original_glrlm_LongRunHighGrayLevelEmphasis",
          "original_glrlm_RunLengthNonUniformity"
          ]

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

# download train and test data
test_path = "data/result/test.csv"
test = pd.read_csv(test_path, ";")
train_path = "data/result/train.csv"
train = pd.read_csv(train_path, ";")
#X = train[chosen1]
X_train = train.iloc[:, 1:train.shape[1] - 7]
y_train = train["isnorm"]
X_test = test.iloc[:, 1:train.shape[1] - 7]
y_test = test["isnorm"]

# Make Logistic Regression (Logit) and test it
'''
clf = linear_model.LogisticRegression(C=1e5, solver='lbfgs')
clf.fit(X_train, y_train)
# Test the classifier
clf.predict_proba(X_test)
print(clf.score(X_train, y_train))
'''

# XGBoost
# fit model on all training data
model = XGBClassifier()
model.fit(X_train, y_train)

# make predictions for test data and evaluate
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# Fit model using each importance as a threshold
thresholds = np.sort(model.feature_importances_)
print("%.2f%", thresholds)
for thresh in thresholds:
    # select features using threshold
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    select_X_train = selection.transform(X_train)
    # train model
    selection_model = XGBClassifier()
    selection_model.fit(select_X_train, y_train)
    # eval model
    select_X_test = selection.transform(X_test)
    y_pred = selection_model.predict(select_X_test)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_test, predictions)
    print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0))
