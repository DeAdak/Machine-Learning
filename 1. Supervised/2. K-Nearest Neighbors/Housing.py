# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 13:30:07 2022

@author: R
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV, train_test_split,kFold
#from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.neighbors import KNeighborsRegressor
Housing = pd.read_csv(r"C:\Users\R\Downloads\12 Practical Machine Learning_\Cases\Real Estate\Housing.csv")
HousingDummy=pd.get_dummies(Housing,drop_first=True)
X = HousingDummy.iloc[:,1:].values
y = HousingDummy.iloc[:,0].values
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=2022)
knn=KNeighborsRegressor(n_neighbors=1)
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)
print(mean_squared_error(y_test, y_pred))
print(r2_score(y_test, y_pred))
########################### Grid Search CV ########################
param_grid={'n_neighbors':np.arange(1,26)}
#param_grid is a dictonary, its values will be used as no of neighbors, don't go beyond 25, it overfits the data
kFold = KFold(n_splits = 5, shuffle = True, random_state = 2022)
knn=KNeighborsRegressor()
gscv=GridSearchCV(knn, param_grid,scoring='r2',cv=kFold)
gscv.fit(X,y)
# .fit will exchage the values as no of neighbor and compute
pd_df=pd.DataFrame(gscv.cv_results_)
#Dictonary --> df conversion as the result produced as Dictonary 
gscv.best_params_
gscv.best_score_
########################### Random Search CV ########################
from sklearn.model_selection import RandomizedSearchCV
param_dist={'n_neighbors':np.arange(1,101)}
#param_grid is a dictonary, its values will be used as no of neighbors, don't go beyond 25, it overfits the data
kFold = KFold(n_splits = 5, shuffle = True, random_state = 2022)
knn=KNeighborsRegressor()
rcv=RandomizedSearchCV(knn, param_dist,scoring='r2',cv=kFold,random_state=2022,n_iter=10)
rcv.fit(X,y)
# .fit will exchage the values as no of neighbor and compute
pd_df=pd.DataFrame(rcv.cv_results_)
#Dictonary --> df conversion as the result produced as Dictonary 

rcv.best_params_
rcv.best_score_



