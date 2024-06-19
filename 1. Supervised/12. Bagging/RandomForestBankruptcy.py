# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 17:24:02 2022

@author: R
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold,GridSearchCV

df=pd.read_csv(r"G:/Ddrive/PG DBDA/12 Practical Machine Learning_/Cases/Bankruptcy/Bankruptcy.csv")
F = df.iloc[:,2:]
R = df.iloc[:,1]
rfc=RandomForestClassifier(random_state=2024)
param={'max_features':[5,10,15,20]}
kfold=KFold(n_splits=(5),shuffle=(True),random_state=2024)
cv=GridSearchCV(rfc, param_grid=param,scoring='roc_auc',cv=kfold,verbose=2)
cv.fit(F,R)
print(cv.best_score_) #0.9162853385930309
print(cv.best_params_) #{'max_features': 5}

