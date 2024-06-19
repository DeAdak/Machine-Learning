# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 16:09:33 2022

@author: R
"""

import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split,KFold,GridSearchCV
from sklearn.metrics import accuracy_score,roc_auc_score,log_loss
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


df=pd.read_csv(r"G:\Ddrive\PG DBDA\12 Practical Machine Learning_/Cases/Bankruptcy/Bankruptcy.csv")
F = df.iloc[:,2:]
R = df.iloc[:,1]

##train_test_split##
X_train,X_test,y_train,y_test=train_test_split(F,R,train_size=0.7,stratify=R,random_state=2024,shuffle=True)

log_reg=LogisticRegression(random_state=2024)
lda=LinearDiscriminantAnalysis()
dtc=DecisionTreeClassifier(max_depth=None,random_state=2024)
svc=SVC(kernel='rbf',probability=(True),random_state=2024)

model_bg=BaggingClassifier(base_estimator=log_reg,oob_score=True,n_estimators=15,max_samples=X_train.shape[0],max_features=X_train.shape[1],random_state=2024,verbose=2)
model_bg.fit(X_train,y_train)
print(model_bg.oob_score_) #0.8152173913043478
y_pred_proba=model_bg.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test, y_pred_proba)) # 0.85
y_pred=model_bg.predict(X_test)
print(log_loss(y_test, y_pred)) #7.20873067782343

model_bg=BaggingClassifier(base_estimator=lda,oob_score=True,n_estimators=15,max_samples=X_train.shape[0],max_features=X_train.shape[1],random_state=2024,verbose=2)
model_bg.fit(X_train,y_train)
print(model_bg.oob_score_) #0.782608695652174
y_pred_proba=model_bg.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test, y_pred_proba)) #0.655
y_pred=model_bg.predict(X_test)
print(log_loss(y_test, y_pred)) #11.714187351463075

model_bg=BaggingClassifier(base_estimator=dtc,oob_score=True,n_estimators=15,max_samples=X_train.shape[0],max_features=X_train.shape[1],random_state=2024,verbose=2)
model_bg.fit(X_train,y_train)
print(model_bg.oob_score_) #0.8152173913043478
y_pred_proba=model_bg.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test, y_pred_proba)) #0.8325
y_pred=model_bg.predict(X_test)
print(log_loss(y_test, y_pred)) #9.912004682007218

model_bg=BaggingClassifier(base_estimator=svc,oob_score=True,n_estimators=15,max_samples=X_train.shape[0],max_features=X_train.shape[1],random_state=2024,verbose=2)
model_bg.fit(X_train,y_train)
print(model_bg.oob_score_) #0.45652173913043476
y_pred_proba=model_bg.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test, y_pred_proba)) #0.31
y_pred=model_bg.predict(X_test)
print(log_loss(y_test, y_pred)) #19.824009364014437
















