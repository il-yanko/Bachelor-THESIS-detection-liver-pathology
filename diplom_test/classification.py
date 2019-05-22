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
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier, plot_importance

import warnings
warnings.filterwarnings("ignore")

path = "data/result/features.csv"
chosen = ["original_glrlm_RunEntropy",
          "original_glrlm_GrayLevelNonUniformity",
          "original_firstorder_10Percentile",
          "original_gldm_GrayLevelNonUniformity",
          "diagnostics_Image-original_Mean"
          ]

chosen1 = [
            "diagnostics_Image-original_Mean",
            "diagnostics_Image-original_Minimum",
            "diagnostics_Image-original_Maximum",
            "original_firstorder_10Percentile",
            "original_firstorder_90Percentile",
          ]
chosen2 = [
            'original_glrlm_RunEntropy',
            'original_glrlm_GrayLevelNonUniformity',
            'original_glrlm_GrayLevelVariance',
            'original_glrlm_LongRunHighGrayLevelEmphasis',
            'original_glrlm_LongRunLowGrayLevelEmphasis',
            'original_glrlm_RunLengthNonUniformity'
          ]
bestwls = [
            'original_glrlm_RunEntropy',
            'original_glrlm_RunLengthNonUniformity',
            "diagnostics_Image-original_Mean",
            "original_firstorder_90Percentile",
          ]

data = pd.read_csv(path, ";")
# крутая штука показывает важность многих фич на 2д картинке
'''
choice = chosen2
choice.append('isnorm')
from pandas.plotting import radviz
plt.figure()
radviz(data[choice], 'isnorm', color=['blue','red'])
plt.show()
'''

# visualize SEABORN
'''
red_blue = ["#ff0000", "#1240ab"]
sns.pairplot(
    data,
    vars=bestwls,
    #hue='data_source', 'isnorm'
    hue='iswls',
    aspect=0.3,
    palette=red_blue,
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

X_train = train.iloc[:, 1:train.shape[1] - 7]
#X_train = train[bestwls]
y_train = train["isnorm"]
X_test = test.iloc  [:, 1:train.shape[1] - 7]
#X_test = test[bestwls]
y_test = test["isnorm"]

all = ["norm", "auh", "hpb", "hpc", "wls"]
wls = ['notwls','wls']
hpb = ['notHPB','HPB']
hpc = ['notHPC','HPC']
auh = ['notAuh','auh']
norma = ['patho','norma']
current = y_test.name


# Make Multinomial Logistic Regression (Logit) and test it

print("\nMultinomial Logistic Regression:\n===================\nPredictable attribute: ",current)
clf = linear_model.LogisticRegression(max_iter=10000, C=1e5, solver='lbfgs')#,multi_class='multinomial')
clf.fit(X_train, y_train)
# Test the classifier
y_pred = clf.predict(X_test)
print("Accuracy:%.2f%%" % (float(accuracy_score(y_test, y_pred))*100))
print("Prediction:\n",y_pred)
print("Real test:\n",y_test.to_numpy())
print(classification_report(y_test, y_pred, target_names=norma))

# choose best features and show new model
'''
# calculate the features importance
coefs,number,arr = list(),list(),list()
for i in range(len(clf.coef_[0])):
    a = float(np.std(X_train, 0)[i] * clf.coef_[0][i])
    b = (a, i)
    coefs.append(b)
dtype = [('coef',float), ('number',int)]
arr = np.sort(np.array(coefs, dtype=dtype), order='coef', kind='mergesort')[::-1]
# choose most important features
best = list()
modelSize = 5
for i in range (modelSize):
    best.append(X_test.columns[arr[i][1]])

# recalculate model
X_train = X_train[best]
X_test = X_test[best]
print("OPTIMIZED MODEL:\n")
print(X_test)
print(X_train)
clf = linear_model.LogisticRegression(max_iter=10000, C=1e5, solver='lbfgs')#,multi_class='multinomial')
clf.fit(X_train, y_train)
# Test the classifier
y_pred = clf.predict(X_test)
print("Accuracy:%.2f%%" % (float(accuracy_score(y_test, y_pred))*100))
print("Prediction:\n",y_pred)
print("Real test:\n",y_test.to_numpy())
print(classification_report(y_test, y_pred, target_names=norma))
'''









# XGBoost
'''
print("\nXGBoost Classification:\n===================\nPredictable attribute: ",current)
# fit model on all training data
model = XGBClassifier()
model.fit(X_train, y_train)

# make predictions for test data and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

#print("Accuracy: %.2f%%" % (accuracy * 100.0))
#print("Prediction:\n",y_pred)
#print("Real data:\n",y_test.to_numpy())

# Fit model using each importance as a threshold
thresholds = np.sort(model.feature_importances_)
#print("thresholds:", thresholds)
'''

# cycle
'''
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
'''
# hand-made threshold
'''
# select features using threshold
threshold = 0.06
selection = SelectFromModel(model, threshold=threshold, prefit=True)
select_X_train = selection.transform(X_train)
# train model
selection_model = XGBClassifier()
selection_model.fit(select_X_train, y_train)
# eval model
select_X_test = selection.transform(X_test)
y_pred = selection_model.predict(select_X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (threshold, select_X_train.shape[1], accuracy*100.0))

print("Prediction:\n",y_pred)
print("Real data:\n",y_test.to_numpy())
'''