# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 21:27:04 2022

@author: R
"""

import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV,StratifiedKFold

df=pd.read_csv(r"G:/Ddrive/PG DBDA/12 Practical Machine Learning_/Cases/Sonar/Sonar.csv")
df_dum=pd.get_dummies(df,drop_first=True)
X=df_dum.iloc[:,:-1]
y=df_dum.iloc[:,-1]

##Linear##
svm=SVC(random_state=(2024),kernel='linear',probability=(True))
param={'C':[0.001,0.1,0.3,0.8,1,1.5,2]}
kfold=StratifiedKFold(n_splits=5,random_state=(2022),shuffle=(True))
cv=GridSearchCV(svm, param_grid=param,scoring='roc_auc',cv=kfold,verbose=2)
cv.fit(X,y)
print(cv.best_score_) #0.8256001664239652
print(cv.best_params_) #{'C': 1}

##Poly##
svm=SVC(random_state=(2024),kernel='poly',probability=(True))
param={'C':[0.001,0.01,0.1,0.5,0.7,1,1.4,2],'coef0':[0,0.5,1,2],'degree':[2,3]}
kfold=StratifiedKFold(n_splits=5,random_state=(2022),shuffle=(True))
cv=GridSearchCV(svm, param_grid=param,scoring='roc_auc',cv=kfold,verbose=2)
cv.fit(X,y)
print(cv.best_score_) #0.9009808612440191
print(cv.best_params_) #{'C': 2, 'coef0': 2, 'degree': 3}

##Radial##
svm=SVC(random_state=(2024),kernel='rbf',probability=(True))
param={'C':[0.001,0.1,0.3,0.8,1,1.5,2],'gamma':[0.001,0.01,0.3,0.7,1,1.9,3]}
kfold=StratifiedKFold(n_splits=5,random_state=(2022),shuffle=(True))
cv=GridSearchCV(svm, param_grid=param,scoring='roc_auc',cv=kfold,verbose=2)
cv.fit(X,y)
print(cv.best_score_) # 0.9535458706053672
print(cv.best_params_) # {'C': 2, 'gamma': 1}
