# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 19:20:31 2022

@author: R
"""

import pandas as pd
#import numpy as np
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

ChemicalProcess = pd.read_csv(r"G:\Ddrive\PG DBDA\12 Practical Machine Learning_\Cases\Chemical Process Data\ChemicalProcess.csv")
# no need for dummy as all fields are numbers only

############# SimpleImputer(strategy='mean') #################################
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(strategy='mean')
cp=imputer.fit_transform(ChemicalProcess)
X=cp[:,1:]
y=cp[:,0]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=2024)
lin_reg=LinearRegression()
lin_reg.fit(X_train, y_train)
print(f"intercept: {lin_reg.intercept_}")
print(f"coefficient: {lin_reg.coef_}")
y_pred=lin_reg.predict(X_test)
print(f"R2 score: {r2_score(y_test, y_pred)}")

########### SimpleImputer(strategy='mean'), StandardScaler ########################################
from sklearn.preprocessing import StandardScaler 
Scaler=StandardScaler()
X=cp[:,1:]
X_scaled=Scaler.fit_transform(X)
y=cp[:,0]
X_train,X_test,y_train,y_test=train_test_split(X_scaled,y,test_size=0.3,random_state=2024)
lin_reg=LinearRegression()
lin_reg.fit(X_train, y_train)
print(f"intercept: {lin_reg.intercept_}")
print(f"coefficient: {lin_reg.coef_}")
y_pred=lin_reg.predict(X_test)
print(f"R2 score: {r2_score(y_test, y_pred)}")


################# SimpleImputer(strategy='median') ######################################
imputer=SimpleImputer(strategy='median')
cp=imputer.fit_transform(ChemicalProcess)
X=cp[:,1:]
y=cp[:,0]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=2022)
lin_reg=LinearRegression()
lin_reg.fit(X_train, y_train)
print(f"intercept: {lin_reg.intercept_}")
print(f"coefficient: {lin_reg.coef_}")
y_pred=lin_reg.predict(X_test)
print(f"R2 score: {r2_score(y_test, y_pred)}")

############## SimpleImputer(strategy='median'), StandardScaler ######################
stdScale=StandardScaler()
X_scaled=stdScale.fit_transform(X)
X_train,X_test,y_train,y_test=train_test_split(X_scaled,y,test_size=0.3,random_state=2022)
lin_reg=LinearRegression()
lin_reg.fit(X_train, y_train)
print(f"intercept: {lin_reg.intercept_}")
print(f"coefficient: {lin_reg.coef_}")
y_pred=lin_reg.predict(X_test)
print(f"R2 score: {r2_score(y_test, y_pred)}")

############## KFold, SimpleImputer, StandardScaler ###################################
from sklearn.model_selection import KFold,cross_val_score
from sklearn.preprocessing import StandardScaler 

imputer=SimpleImputer(strategy='mean')
cp=imputer.fit_transform(ChemicalProcess)
Scaler=StandardScaler()
X=cp[:,1:]
X_scaled=Scaler.fit_transform(X)
y=cp[:,0]
lin_reg=LinearRegression()
kfold=KFold(n_splits=5,shuffle=True,random_state=(2022))
res=cross_val_score(lin_reg,X_scaled,y,cv=kfold,scoring='r2')
print(f"R2 score: {res.mean()}")

imputer=SimpleImputer(strategy='median')
cp=imputer.fit_transform(ChemicalProcess)
Scaler=StandardScaler()
X=cp[:,1:]
X_scaled=Scaler.fit_transform(X)
y=cp[:,0]
lin_reg=LinearRegression()
kfold=KFold(n_splits=5,shuffle=True,random_state=(2022))
res=cross_val_score(lin_reg,X_scaled,y,cv=kfold,scoring='r2')
print(f"R2 score: {res.mean()}")

#r2_score is negative so this model is also not suitable
















