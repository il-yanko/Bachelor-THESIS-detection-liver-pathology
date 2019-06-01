#!/usr/bin/env python

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn import linear_model,tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.multiclass import unique_labels
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


#from boruta import BorutaPy

import warnings
warnings.filterwarnings("ignore")

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False, title=None,
                          cmap=plt.cm.get_cmap('Reds')):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = "NORMALIZED CONFUSION MATRIX"
        else:
            title = "NON-NORMALIZED CONFUSION MATRIX"

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("NORMALIZED CONFUSION MATRIX")
    else:
        print("NON-NORMALIZED CONFUSION MATRIX")

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           ylabel='TRUE CLASS',
           xlabel='PREDICTED CLASS'
           )
    ax.set_title(title, fontsize=22)

    # Rotate the tick labels and set their alignment.
    #plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #         rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center", fontsize=20,
                    color="white" if cm[i, j] > thresh else "black")
            ax.tick_params(labelsize=20)
    fig.tight_layout()
    return ax

path = "data/result/features.csv"
bestnorm = [
            "original_glrlm_RunEntropy",
            "original_glrlm_GrayLevelNonUniformity",
            "original_firstorder_10Percentile",
            #"original_gldm_GrayLevelNonUniformity",
            #"diagnostics_Image-original_Mean"
           ]
bestwls = [
            'original_glrlm_RunEntropy',
            #'original_glrlm_RunLengthNonUniformity',
            #"diagnostics_Image-original_Mean",
            "original_firstorder_90Percentile",
          ]
besthpc = [
            #"diagnostics_Image-original_Mean",
            #"diagnostics_Image-original_Minimum",
            #"diagnostics_Image-original_Maximum",
            #"original_firstorder_10Percentile",
            #"original_firstorder_90Percentile",
            #"original_gldm_GrayLevelNonUniformity",
            "original_glcm_ClusterShade",
            "original_firstorder_RobustMeanAbsoluteDeviation",
            #"original_firstorder_TotalEnergy",
            "original_glrlm_RunEntropy",
            #"original_gldm_DependenceNonUniformity",
            #"original_glrlm_LongRunHighGrayLevelEmphasis",
            "original_gldm_LargeDependenceEmphasis"
          ]
besthpb = [
            "original_gldm_DependenceVariance",
            #"diagnostics_Image-original_Mean",
            "original_glcm_ClusterShade",
            #"original_gldm_LargeDependenceLowGrayLevelEmphasis",
            "original_glcm_Idmn",
            "original_firstorder_Skewness",
            "original_ngtdm_Strength",
            #"original_gldm_DependenceNonUniformity",
            #"original_firstorder_Kurtosis",
            #"original_firstorder_Energy",
            #"original_glrlm_GrayLevelNonUniformity",
]
bestauh = [
            'original_firstorder_TotalEnergy',
            'original_firstorder_Energy',
            'original_glcm_ClusterProminence',
            'original_glcm_Imc1'
          ]

data = pd.read_csv(path, ";")

# radviz (Dimensional Anchor)
'''
# крутая штука показывает важность многих фич на 2д картинке
choice = bestnorm
choice.append('isnorm')
from pandas.plotting import radviz
plt.figure()
radviz(data[choice], 'isnorm', color=['blue','red'])
plt.show()
'''

# seaborn
'''
red_blue = ["#ff0000", "#1240ab"]
sns.pairplot(
    data,
    vars=besthpb,
    hue='ishpb',
    aspect=0.3,
    palette=red_blue,
    #kind="skatter"
    #markers="."
)
plt.show()
plt.tight_layout()
'''

#====================================================================
# download train and test data
test_path = "data/result/test.csv"
test = pd.read_csv(test_path, ";")
train_path = "data/result/train.csv"
train = pd.read_csv(train_path, ";")

all = ["norm", "auh", "hpb", "hpc", "wls"]
wls = ['notwls','wls']
hpb = ['notHPB','HPB']
hpc = ['notHPC','HPC']
auh = ['notAuh','auh']
norma = ['patho','norma']

cols_to_drop = ['id','data_source','diagnosis_code','isnorm','isauh','ishpb','ishpc','iswls']
model_features = [col for col in train.columns if col not in cols_to_drop]

# pool of all classification settings
poolParam = ["diagnosis_code"]#,"iswls","ishpb","ishpc","isauh","isnorm"]
poolLabel = [all]#, wls, hpb, hpc, auh, norma]
poolTests = {poolParam[a]:poolLabel[a] for a in range (len(poolParam))}

# single classification setting
#model_parameter = "diagnosis_code"
#model_labels = all

#====================================================================

def predict_and_show(X_train, y_train, X_test, y_test, clf, plt, names, clf_name):
    print("\n", clf_name, ":\n================================================\nPredictable attribute: ", current)
    clf.fit(X_train, y_train)
    # Test the classifier
    y_pred = clf.predict(X_test)
    print("Accuracy:%.2f%%" % (float(accuracy_score(y_test, y_pred)) * 100))
    print("Prediction:\n", y_pred)
    print("Real test:\n", y_test.to_numpy())
    # print(classification_report(y_test, y_pred, target_names=names))
    # Plot normalized confusion matrix
    # if you need numbers: classes=np.asarray(unique_labels(y_test), dtype=int)
    plot_confusion_matrix(y_test, y_pred, classes=names, normalize=True, title=clf_name)
    plt.show()

clf_names,clf_models  = list(), list()


clf_models.append(make_pipeline (#PCA(n_components=2),
                                 StandardScaler(),
                                 tree.DecisionTreeClassifier(random_state=0,criterion='gini',max_features=2)))
clf_names.append("Decision Tree Classifier")


clf_models.append(make_pipeline (PCA(n_components=5), #StandardScaler(),
                                 linear_model.LogisticRegression(max_iter=1000000, C=1e3,
                                                     solver='newton-cg',penalty="l2" ,multi_class='multinomial'
                                                                 )))
clf_names.append("Logistic Regression")

clf_models.append(make_pipeline (PCA(n_components=5), StandardScaler(),
                                 RandomForestClassifier(max_depth=10, n_estimators=100,
                                            max_features=2, random_state=0,
                                            criterion='gini',bootstrap=False)))
clf_names.append("Random Forest Classifier")



clf_models.append(make_pipeline (PCA(n_components=3), #StandardScaler(),
                                 svm.SVC(gamma='scale', kernel='rbf')))
clf_names.append("C-Support Vector Machine")

clf_models.append(make_pipeline (PCA(n_components=2), StandardScaler(),
                                 GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
                                                max_depth=8, random_state=0)))
clf_names.append("Gradient Boosting")

clf_models.append(make_pipeline (PCA(n_components=3), StandardScaler(),
                                 KNeighborsClassifier(5)))
clf_names.append("k-Nearest Neighbors")




clfs = {clf_names[a]:clf_models[a] for a in range(len(clf_names))}
for name,model in clfs.items():
    for param, label in poolTests.items():

        # X_train = train.iloc[:, 1:train.shape[1] - 7]
        X_train = train[model_features]
        y_train = train[param].astype(int)

        # X_test = test.iloc  [:, 1:train.shape[1] - 7]
        X_test = test[model_features]
        y_test = test[param].astype(int)

        current = param
        predict_and_show(X_train, y_train, X_test, y_test, model, plt, label, name)



# model saving

# TODO: save all one-vs-all model's and their accuracies
#prepare 1 (!!!) model for saving
'''
name = "Logistic Regression"
model = clfs[name]


# save the model to disk
filename = 'data/result/model/'+ name +'.sav'
file = open(filename, 'wb')
pickle.dump(model, file)
print("Model of", name, "was saved")
file.close()
'''


for name,model in clfs.items():
    filename = 'data/result/model/'+ name +'.sav'
    file = open(filename, 'wb')
    pickle.dump(model, file)
    print("Model of", name, "was saved")
    file.close()






# Different additional unused code
'''
# Multi-Logit: choose best features and show new model
clf = make_pipeline (PCA(n_components=5),StandardScaler(),
                     linear_model.LogisticRegression(max_iter=10000, C=1e5, solver='lbfgs',multi_class='multinomial'))
print("MODEL:")
for i in range(len(model_features)):
    print(model_features[i],clf.coef_[0][i])
# calculate the features importance
coefs,arr = list(),list()
for i in range(len(clf.coef_[0])):
    a = float(np.std(X_train, 0)[i] * clf.coef_[0][i])
    b = (a, i)
    coefs.append(b)
dtype = [('coef',float), ('number',int)]
arr = np.sort(np.array(coefs, dtype=dtype), order='coef', kind='mergesort')[::-1]
# choose most important features
best = list()
modelSize = 7
for i in range (modelSize):
    best.append(X_test.columns[arr[i][1]])
# recalculate model
X_train = X_train[best]
X_test = X_test[best]
print("OPTIMIZED MODEL:\n")
print('best=',best)
clf1 = linear_model.LogisticRegression(max_iter=10000, C=1e5, solver='lbfgs')#,multi_class='multinomial')
clf1.fit(X_train, y_train)
# Test the classifier
y_pred = clf1.predict(X_test)
print("Accuracy:%.2f%%" % (float(accuracy_score(y_test, y_pred))*100))
print("Prediction:\n",y_pred)
print("Real test:\n",y_test.to_numpy())
print(classification_report(y_test, y_pred, target_names=model_names))


# XGBoost
from xgboost import XGBClassifier, plot_importance
print("\nXGBoost Classification:\n===================\nPredictable attribute: ",current)
# fit model on all training data
model = XGBClassifier()
model.fit(X_train, y_train)
plot_importance(model)
plt.show()

# make predictions for test data and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy: %.2f%%" % (accuracy * 100.0))
print("Prediction:\n",y_pred)
print("Real data:\n",y_test.to_numpy())

# Fit model using each importance as a threshold
thresholds = np.sort(model.feature_importances_)
#print("thresholds:", thresholds)

# XGB: cycle
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

# XGB: hand-made threshold
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

# Build correlation between all model features
'''
data = pd.read_csv(path, ";")
X_all = data[model_features]
# Draw the full plot
sns.clustermap(X_all.corr(), center=0, cmap="vlag",
               linewidths=.75, figsize=(13, 13))
plt.show()
'''
