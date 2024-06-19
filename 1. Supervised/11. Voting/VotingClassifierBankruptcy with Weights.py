# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 12:38:42 2022

@author: R
"""

import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.ensemble import VotingClassifier,VotingRegressor
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split,KFold,GridSearchCV
from sklearn.metrics import accuracy_score,roc_auc_score,log_loss
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


df=pd.read_csv(r"G:/Ddrive/PG DBDA/12 Practical Machine Learning_/Cases/Bankruptcy/Bankruptcy.csv")
F = df.iloc[:,2:]
R = df.iloc[:,1]

##train_test_split##
X_train,X_test,y_train,y_test=train_test_split(F,R,train_size=0.7,random_state=2024,shuffle=True)

##avg model##
log_reg=LogisticRegression(random_state=2024)
lda=LinearDiscriminantAnalysis()
dtc=DecisionTreeClassifier(max_depth=None,random_state=2024)
svc=SVC(kernel='rbf',probability=(True),random_state=2024)
models=[('LogR',log_reg),('LDA',lda),('DTC',dtc),('SVC',svc)]
vote=VotingClassifier(estimators=models,voting='soft')
vote.fit(X_train,y_train)
y_pred = vote.predict(X_test)
y_pred_prob=vote.predict_proba(X_test)[:,1]
print(log_loss(y_test, y_pred)) #8.526455640436316
print(roc_auc_score(y_test, y_pred_prob)) # 0.8648148148148149


##weightrd avg model##
log_reg.fit(X_train,y_train)
y_pred_log_reg=log_reg.predict_proba(X_test)[:,1]
e1=roc_auc_score(y_test, y_pred_log_reg) #0.8933333333333334

lda.fit(X_train,y_train)
y_pred_lda=lda.predict_proba(X_test)[:,1]
e2=roc_auc_score(y_test, y_pred_lda) #0.8853333333333333

dtc.fit(X_train,y_train)
y_pred_dtc=dtc.predict_proba(X_test)[:,1]
e3=roc_auc_score(y_test, y_pred_dtc) #0.78

svc.fit(X_train,y_train)
y_pred_svc=svc.predict_proba(X_test)[:,1]
e4=roc_auc_score(y_test, y_pred_svc) #0.34933333333333333

w1=e1/(e1+e2+e3+e4)
w2=e2/(e1+e2+e3+e4)
w3=e3/(e1+e2+e3+e4)
w4=e4/(e1+e2+e3+e4)

vote=VotingClassifier(estimators=models,voting='soft',weights=[w1,w2,w3,w4])
vote.fit(X_train,y_train)
y_pred=vote.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test, y_pred)) #0.8986666666666667















