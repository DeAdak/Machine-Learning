# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 16:08:48 2022

@author: R
"""

import pandas as pd
import numpy as np




ChemicalProcess = pd.read_csv(r"C:\Users\R\Downloads\12 Practical Machine Learning_\Cases\Chemical Process Data\ChemicalProcess.csv")
# no need for dummy as all fields are numbers only

from sklearn.impute import SimpleImputer
imputer=SimpleImputer(strategy='mean')
CP=imputer.fit_transform(ChemicalProcess)
cpdf=pd.DataFrame(CP,columns=ChemicalProcess.columns)
X=cpdf.iloc[:,1:].values
y=cpdf.iloc[:,0].values

from sklearn.preprocessing import StandardScaler
stdScale=StandardScaler()
X_scaled=stdScale.fit_transform(X)
########################### Grid Search CV ########################
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
param_grid={'n_neighbors':np.arange(1,20)}
#param_grid is a dictonary, its values will be used as no of neighbors, don't go beyond 25, it overfits the data
kFold = KFold(n_splits = 5, shuffle = True, random_state = 2022)
knn=KNeighborsRegressor()
gscv=GridSearchCV(knn, param_grid,scoring='r2',cv=kFold)
gscv.fit(X_scaled,y)
# .fit will exchage the values as no of neighbor and compute
pd_df=pd.DataFrame(gscv.cv_results_)
#Dictonary --> df conversion as the result produced as Dictonary 
gscv.best_params_
gscv.best_score_

imputer=SimpleImputer(strategy='median')
CP=imputer.fit_transform(ChemicalProcess)
cpdf=pd.DataFrame(CP,columns=ChemicalProcess.columns)
X=cpdf.iloc[:,1:].values
y=cpdf.iloc[:,0].values
stdScale=StandardScaler()
X_scaled=stdScale.fit_transform(X)
########################### Grid Search CV ########################
param_grid={'n_neighbors':np.arange(1,20)}
#param_grid is a dictonary, its values will be used as no of neighbors, don't go beyond 25, it overfits the data
kFold = KFold(n_splits = 5, shuffle = True, random_state = 2022)
knn=KNeighborsRegressor()
gscv=GridSearchCV(knn, param_grid,scoring='r2',cv=kFold)
gscv.fit(X_scaled,y)
# .fit will exchage the values as no of neighbor and compute
pd_df=pd.DataFrame(gscv.cv_results_)
#Dictonary --> df conversion as the result produced as Dictonary 
gscv.best_params_
gscv.best_score_













