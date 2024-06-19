# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 08:43:27 2022

@author: R
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,StratifiedKFold,GridSearchCV
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,roc_auc_score
df=pd.read_csv(r"G:\Ddrive\PG DBDA\12 Practical Machine Learning_/Cases/Image Segmentation/Image_Segmention.csv")
# Image_Segmention.csv has several Response, all are categorical. so dummy will not be supported by train_test_splits
X = df.iloc[:,1:]
y = df.iloc[:,0]
# LabelEncoder() will numberify all the respnses as 1,2,3.... It is now can be processed by train_test_splits
le = LabelEncoder()
y_le = le.fit_transform(y)
X_train,X_test,y_train,y_test=train_test_split(X,y_le,test_size=0.3,random_state=2024,stratify=y_le)

# solver{‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’}, default=’lbfgs’
# Algorithm to use in the optimization problem. Default is ‘lbfgs’.
# To choose a solver, you might want to consider the following aspects:
# For small datasets, ‘liblinear’ is a good choice, whereas ‘sag’ and ‘saga’ are faster for large ones;
# For multiclass problems, only ‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’ handle multinomial loss;
# ‘liblinear’ is limited to one-versus-rest schemes.
## Use random_state when solver == ‘sag’, ‘saga’ or ‘liblinear’ to shuffle the data.

log_reg=LogisticRegression(solver='liblinear', random_state=2024)
log_reg.fit(X_train,y_train)
y_pred=log_reg.predict(X_test)
# y_inv=le.inverse_transform(y_pred)
y_pred_prob=log_reg.predict_proba(X_test)
print(accuracy_score(y_test, y_pred)) #0.9047619047619048
# roc_auc_score is binary, to do a multiclass, we need to apecify it.
print(roc_auc_score(y_test, y_pred_prob,multi_class='ovr')) #0.9861845972957084
print(roc_auc_score(y_test, y_pred_prob,multi_class='ovo')) #0.9861845972957084
##################################################################################################
log_reg=LogisticRegression(random_state=2024)
param={'multi_class':['ovr', 'multinomial','auto'],'solver':['sag','lbfgs', 'liblinear', 'saga','newton-cg']}
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kfold=StratifiedKFold(n_splits=5,random_state=2024,shuffle=(True))
cv=GridSearchCV(log_reg, param_grid=param,scoring='roc_auc_ovr',cv=kfold)
cv.fit(X_scaled,y_le)
print(cv.best_params_) #{'multi_class': 'multinomial', 'solver': 'newton-cg'}
print(cv.best_score_) #0.9812849584278155

best_model=cv.best_estimator_
y_pred=best_model.predict(X_test)
y_inv=le.inverse_transform(y_pred)
print(cv.best_score_) #0.9812849584278155




















