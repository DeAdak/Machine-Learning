# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 21:32:04 2022

@author: R
"""


import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV,StratifiedKFold
from sklearn.preprocessing import LabelEncoder

df=pd.read_csv(r"G:\Ddrive\PG DBDA/12 Practical Machine Learning_/Cases/Satellite Imaging/Satellite.csv",sep=';')
#df_dum=pd.get_dummies(df,drop_first=True)
X=df.iloc[:,:-1]
y=df.iloc[:,-1]
le=LabelEncoder()
y_le=le.fit_transform(y)
##Linear##
svm=SVC(random_state=(2024),kernel='linear',probability=(True))
param={'C':[0.001,0.1,0.3,0.8,1,1.5,2],'decision_function_shape':['ovo','ovr']}
kfold=StratifiedKFold(n_splits=5,random_state=(2024),shuffle=(True))
cv=GridSearchCV(svm, param_grid=param,scoring='neg_log_loss',cv=kfold,verbose=2)
cv.fit(X,y_le)
print(cv.best_score_) #-0.32900340937750394
print(cv.best_params_) # {'C': 0.001, 'decision_function_shape': 'ovo'}

##Poly##
svm=SVC(random_state=(2024),kernel='poly',probability=(True))
param={'C':[0.001,0.01,0.1,0.5,0.7,1,1.4,2],'coef0':[0,0.5,1,2],'degree':[2,3],'decision_function_shape':['ovo','ovr']}
kfold=StratifiedKFold(n_splits=5,random_state=(2024),shuffle=(True))
cv=GridSearchCV(svm, param_grid=param,scoring='neg_log_loss',cv=kfold,verbose=2)
cv.fit(X,y_le)
print(cv.best_score_) # -0.27249435236526015
print(cv.best_params_) # {'C': 2, 'coef0': 2, 'decision_function_shape': 'ovo', 'degree': 3}

##Radial##
svm=SVC(random_state=(2024),kernel='rbf',probability=(True))
param={'C':[0.001,0.1,0.3,0.8,1,1.5,2],'gamma':[0.001,0.01,0.3,0.7,1,1.9,3],'decision_function_shape':['ovo','ovr']}
kfold=StratifiedKFold(n_splits=5,random_state=(2024),shuffle=(True))
cv=GridSearchCV(svm, param_grid=param,scoring='neg_log_loss',cv=kfold,verbose=2)
cv.fit(X,y_le)
print(cv.best_score_) # -0.2232726785431256
print(cv.best_params_) #{'C': 2, 'decision_function_shape': 'ovo', 'gamma': 0.001}
