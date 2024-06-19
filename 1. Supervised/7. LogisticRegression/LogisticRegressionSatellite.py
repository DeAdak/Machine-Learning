# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 10:28:45 2022

@author: R
"""

import pandas as pd
from sklearn.model_selection import StratifiedKFold,GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
df=pd.read_csv(r"G:/Ddrive/PG DBDA/12 Practical Machine Learning_/Cases/Satellite Imaging/Satellite.csv",sep=';')
# Image_Segmention.csv has several Response, all are categorical. so dummy will not be supported by train_test_splits
X = df.iloc[:,:-1]
y = df.iloc[:,-1]
# LabelEncoder() will numberify all the respnses as 1,2,3.... It is now can be processed by train_test_splits
le = LabelEncoder()
y_le = le.fit_transform(y)
log_reg=LogisticRegression(random_state=2024)
param={'multi_class':['ovr', 'multinomial','auto'],'solver':['sag','lbfgs', 'liblinear', 'saga','newton-cg']}
kfold=StratifiedKFold(n_splits=5,random_state=2024,shuffle=(True))
cv=GridSearchCV(log_reg, param_grid=param,scoring='roc_auc_ovr',cv=kfold,verbose=2)
cv.fit(X,y_le)
print(cv.best_params_) #{'multi_class': 'multinomial', 'solver': 'newton-cg'}
print(cv.best_score_) #0.9763606817087631
print(cv.best_estimator_)
model = cv.best_estimator_

param={'multi_class':['multinomial'],'solver':['newton-cg']}
kfold=StratifiedKFold(n_splits=5,random_state=2022,shuffle=(True))
cv=GridSearchCV(model, param_grid=param,scoring='neg_log_loss',cv=kfold,verbose=2)
cv.fit(X,y_le)
print(cv.best_params_)
print(cv.best_score_) #-0.36588607239474824
print(cv.best_estimator_)
