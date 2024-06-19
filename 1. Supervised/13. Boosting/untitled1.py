# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 09:57:10 2024

@author: R
"""

import pandas as pd
from sklearn.model_selection import KFold,GridSearchCV
from xgboost import XGBRFRegressor

otto=pd.read_csv(r'G:/Ddrive/PG DBDA/12 Practical Machine Learning_/Cases/otto-group-product-classification/train.csv')
F = df.iloc[:,:-1]
R = df.iloc[:,-1]


xgbr=XGBRFRegressor(random_state=2024)
lr=[0.01,0.1,0.3,0.5,0.6]
n_est=[10,25,50]
max_d=[3,5,10]
params=dict(learning_rate=lr,max_depth=max_d,n_estimators=n_est)
kfold=KFold(n_splits=(5),shuffle=(True),random_state=(2024))
cv=GridSearchCV(xgbr, param_grid=params,scoring='r2',verbose=2,cv=kfold)
cv.fit(F,R)
cv.best_score_ #0.7489972996983496
cv.best_params_ #{'learning_rate': 0.6, 'max_depth': 10, 'n_estimators': 10}