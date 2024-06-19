# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 16:35:36 2022

@author: R
"""

import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split,KFold,GridSearchCV
from sklearn.metrics import accuracy_score,roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis

df=pd.read_csv(r"G:/Ddrive/PG DBDA/12 Practical Machine Learning_/Cases/Sonar/Sonar.csv")
df_dum=pd.get_dummies(df,drop_first=True)
F = df_dum.iloc[:,:-1]
R = df_dum.iloc[:,-1]

##train_test_split##
X_train,X_test,y_train,y_test=train_test_split(F,R,train_size=0.7,stratify=R,random_state=2024,shuffle=True)

log_reg=LogisticRegression(random_state=2024)
lda=LinearDiscriminantAnalysis()
qda=QuadraticDiscriminantAnalysis()
dtc=DecisionTreeClassifier(max_depth=None,random_state=2024)
svc_rad=SVC(probability=(True),random_state=2024) #default=’rbf’
svc_lin=SVC(kernel='linear',probability=(True),random_state=2024)

model_bg=BaggingClassifier(base_estimator=svc_lin,oob_score=True,n_estimators=15,max_samples=X_train.shape[0],max_features=X_train.shape[1],random_state=2024,verbose=2)
model_bg.fit(X_train,y_train)
print(model_bg.oob_score_) #0.7034482758620689
y_pred_proba=model_bg.predict_proba(X_test)[:,1]
#print(model_bg.oob_score_)
print(roc_auc_score(y_test, y_pred_proba)) #0.8853955375253548

model_bg=BaggingClassifier(base_estimator=svc_rad,oob_score=True,n_estimators=15,max_samples=X_train.shape[0],max_features=X_train.shape[1],random_state=2024,verbose=2)
model_bg.fit(X_train,y_train)
print(model_bg.oob_score_) #0.7862068965517242
y_pred_proba=model_bg.predict_proba(X_test)[:,1]
#print(model_bg.oob_score_)
print(roc_auc_score(y_test, y_pred_proba))#0.9320486815415822

model_bg=BaggingClassifier(base_estimator=lda,oob_score=True,n_estimators=15,max_samples=X_train.shape[0],max_features=X_train.shape[1],random_state=2024,verbose=2)
model_bg.fit(X_train,y_train)
print(model_bg.oob_score_) #0.8068965517241379
y_pred_proba=model_bg.predict_proba(X_test)[:,1]
#print(model_bg.oob_score_)
print(roc_auc_score(y_test, y_pred_proba))#0.7890466531440163

model_bg=BaggingClassifier(base_estimator=qda,oob_score=True,n_estimators=15,max_samples=X_train.shape[0],max_features=X_train.shape[1],random_state=2024,verbose=2)
model_bg.fit(X_train,y_train)
print(model_bg.oob_score_) #0.6620689655172414
y_pred_proba=model_bg.predict_proba(X_test)[:,1]
#print(model_bg.oob_score_)
print(roc_auc_score(y_test, y_pred_proba))#0.8504056795131846

model_bg=BaggingClassifier(base_estimator=log_reg,oob_score=True,n_estimators=15,max_samples=X_train.shape[0],max_features=X_train.shape[1],random_state=2024,verbose=2)
model_bg.fit(X_train,y_train)
print(model_bg.oob_score_) #0.7034482758620689
y_pred_proba=model_bg.predict_proba(X_test)[:,1]
#print(model_bg.oob_score_)
print(roc_auc_score(y_test, y_pred_proba))#0.8762677484787018

model_bg=BaggingClassifier(base_estimator=dtc,oob_score=True,n_estimators=15,max_samples=X_train.shape[0],max_features=X_train.shape[1],random_state=2024,verbose=2)
model_bg.fit(X_train,y_train)
print(model_bg.oob_score_) #0.7655172413793103
y_pred_proba=model_bg.predict_proba(X_test)[:,1]
#print(model_bg.oob_score_)
print(roc_auc_score(y_test, y_pred_proba))#0.8940162271805274


