# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 14:51:16 2022

@author: R
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score,KFold, GridSearchCV
#from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score,accuracy_score,r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
ConcreteStrength = pd.read_csv(r"C:\Users\R\Downloads\12 Practical Machine Learning_\Cases\Concrete Strength\Concrete_Data.csv")
# no need for dummy as all fields are numbers only
X=ConcreteStrength.iloc[:,:-1].values
y=ConcreteStrength.iloc[:,-1].values
stdScale=StandardScaler()
X_scaled=stdScale.fit_transform(X)
########################### Grid Search CV ########################
param_grid={'n_neighbors':np.arange(1,25)}
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
########################### Random Search CV ########################
from sklearn.model_selection import RandomizedSearchCV
param_dist={'n_neighbors':np.arange(1,100)}
#param_grid is a dictonary, its values will be used as no of neighbors, don't go beyond 25, it overfits the data
kFold = KFold(n_splits = 5, shuffle = True, random_state = 2022)
knn=KNeighborsRegressor()
rcv=RandomizedSearchCV(knn, param_dist,scoring='r2',cv=kFold,random_state=2022,n_iter=15)
rcv.fit(X_scaled,y)
# .fit will exchage the values as no of neighbor and compute
pd_df=pd.DataFrame(rcv.cv_results_)
#Dictonary --> df conversion as the result produced as Dictonary 

rcv.best_params_
rcv.best_score_



