# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 11:37:08 2022

@author: R
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold,GridSearchCV,cross_val_score,KFold,GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import accuracy_score,roc_auc_score,confusion_matrix,classification_report
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis


glass=pd.read_csv(r"G:\Ddrive\PG DBDA/12 Practical Machine Learning_/Cases/Glass Identification/Glass.csv")
# Image_Segmention.csv has several Response, all are categorical. so dummy will not be supported by train_test_splits
F = glass.iloc[:,:-1]
R = glass.iloc[:,-1]
# LabelEncoder() will numberify all the respnses as 1,2,3.... It is now can be processed by train_test_splits
lda=LinearDiscriminantAnalysis()
qda=QuadraticDiscriminantAnalysis()

##LDA##
kfold=StratifiedKFold(n_splits=(5),shuffle=(True),random_state=(2024))
cv=cross_val_score(lda,F,R,scoring='neg_log_loss',cv=kfold)
print(cv.mean()) #-1.3163363939664063

##QDA##
qda=QuadraticDiscriminantAnalysis()
kfold=StratifiedKFold(n_splits=(5),shuffle=(True),random_state=(2024))
cv=cross_val_score(qda,F,R,scoring='neg_log_loss',cv=kfold)
print(cv.mean()) #"Variables are collinear"

##Linear##
svm=SVC(random_state=(2024),kernel='linear',probability=(True))
param={'C':[0,0.01,0.1,0.2,0.5,1],'decision_function_shape':['ovo','ovr']}
kfold=StratifiedKFold(n_splits=5,random_state=(2024),shuffle=(True))
cv=GridSearchCV(svm, param_grid=param,scoring='neg_log_loss',cv=kfold,verbose=2)
cv.fit(F,R)
print(cv.best_score_) # -0.9358111223013502
print(cv.best_params_) #{'C': 1, 'decision_function_shape': 'ovo'}

##Radial##
svm=SVC(random_state=(2024),kernel='rbf',probability=(True))
param={'C':[0,0.01,0.1,0.2,0.5,1],'gamma':[0.1,0.5,1],'decision_function_shape':['ovo','ovr']}
kfold=StratifiedKFold(n_splits=5,random_state=(2024),shuffle=(True))
cv=GridSearchCV(svm, param_grid=param,scoring='neg_log_loss',cv=kfold,verbose=2)
cv.fit(F,R)
print(cv.best_score_) #-0.8597117423915852
print(cv.best_params_) #{'C': 1, 'decision_function_shape': 'ovo', 'gamma': 1}

##DTC##
dtc=DecisionTreeClassifier()
depth_range=[None,8,10]
minsplit=[2,5,10]
minleaf=[1,5,10]
param=dict(max_depth=depth_range,min_samples_leaf=minleaf,min_samples_split=minsplit)
kfold=KFold(n_splits=5,shuffle=(True),random_state=(2024))
cv=GridSearchCV(dtc, param_grid=param,scoring='roc_auc_ovr',cv=kfold,verbose=2)
cv.fit(F,R)
res=cv.cv_results_
print(cv.best_params_) # {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2}
print(cv.best_score_) # nan

import matplotlib.pyplot as plt
best_model= cv.best_estimator_

plt.figure(figsize=(15,10))
plot_tree(best_model,feature_names=F.columns,class_names=['1', '2', '3', '5','6', '7'],filled=True,rounded=True)
plt.show()

plt.figure(figsize=(15,10))
ind=np.arange(9)
plt.bar(ind,best_model.feature_importances_)
plt.xticks(ind,F.columns,rotation=90)
plt.title('Features')
plt.xlabel('Variables')
plt.show()

