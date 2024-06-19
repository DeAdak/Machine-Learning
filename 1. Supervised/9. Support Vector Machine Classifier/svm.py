# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 17:55:13 2022

@author: R
"""

import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV,StratifiedKFold

df=pd.read_csv(r"G:/Ddrive/PG DBDA/12 Practical Machine Learning_/Cases/Wisconsin/BreastCancer.csv")
df_dum=pd.get_dummies(df,drop_first=True)
X=df_dum.iloc[:,1:-1]
y=df_dum.iloc[:,-1]

#####################   Linear   #############################
svm=SVC(random_state=(2024),kernel='linear',probability=(True))
# probability=(True) as it is needed for roc_auc, to use predict_proba output
param={'C':[0.001,0.01,0.1,0.5,0.7,1,1.4,2]}
kfold=StratifiedKFold(n_splits=5,random_state=(2022),shuffle=(True))
cv=GridSearchCV(svm, param_grid=param,scoring='roc_auc',cv=kfold,verbose=2)
cv.fit(X,y)
print(cv.best_score_) #0.9956557999453961
print(cv.best_params_) #{'C': 0.01}

####################   Poly   ###############################
svm=SVC(random_state=(2024),kernel='poly',probability=(True))
param={'C':[0.001,0.01,0.1,0.5,0.7,1,1.4,2],'coef0':[0,0.5,1,2],'degree':[2,3]}
kfold=StratifiedKFold(n_splits=5,random_state=(2022),shuffle=(True))
cv=GridSearchCV(svm, param_grid=param,scoring='roc_auc',cv=kfold,verbose=2)
cv.fit(X,y)
print(cv.best_score_) # 0.9957006530514738
print(cv.best_params_) # {'C': 0.1, 'coef0': 0.5, 'degree': 2}

##################   Radial   ############################
svm=SVC(random_state=(2024),kernel='rbf',probability=(True))
param={'C':[0.001,0.01,0.1,0.5,0.7,1,1.4,2],'gamma':['auto','scale']}
kfold=StratifiedKFold(n_splits=5,random_state=(2022),shuffle=(True))
cv=GridSearchCV(svm, param_grid=param,scoring='roc_auc',cv=kfold,verbose=2)
cv.fit(X,y)
print(cv.best_score_) #0.9913110920431241
print(cv.best_params_) #{'C': 0.001, 'gamma': 'scale'}
