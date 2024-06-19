# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 12:54:31 2022

@author: R
"""
import pandas as pd
from sklearn.model_selection import KFold,GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor

df=pd.read_csv(r"G:/Ddrive/PG DBDA/12 Practical Machine Learning_/Cases/Concrete Strength/Concrete_Data.csv")
F = df.iloc[:,:-1]
R = df.iloc[:,-1]

##train_test_split##
#X_train,X_test,y_train,y_test=train_test_split(F,R,train_size=0.3,stratify=R,random_state=2022,shuffle=True)

gbr=GradientBoostingRegressor(random_state=(2024),verbose=2)
lr=[0.01,0.1,0.3,0.5,0.6]
n_est=[10,25,50]
max_d=[3,5,None]
params=dict(learning_rate=lr,max_depth=max_d,n_estimators=n_est)
kfold=KFold(n_splits=(5),shuffle=(True),random_state=(2024))
cv=GridSearchCV(gbr, param_grid=params,scoring='r2',verbose=2,cv=kfold)
cv.fit(F,R)
cv.best_score_ #0.9309593370700606
cv.best_params_ #{'learning_rate': 0.3, 'max_depth': 5, 'n_estimators': 50}
















