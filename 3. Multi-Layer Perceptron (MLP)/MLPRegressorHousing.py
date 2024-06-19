# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 14:25:09 2022

@author: R
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split,GridSearchCV,KFold
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
housing=pd.read_csv(r"C:/Users/R/Downloads/12 Practical Machine Learning_/Cases/Real Estate/Housing.csv")
housing_dum=pd.get_dummies(housing,drop_first=True)
F = housing_dum.iloc[:,1:]
R = housing_dum.iloc[:,0]
scalarX = MinMaxScaler()
scalarY = MinMaxScaler()
F = F.values
R = R.values
X_train,X_test,y_train,y_test=train_test_split(F,R,test_size=0.3,random_state=2022)
X_train = scalarX.fit_transform(X_train)
y_train = scalarY.fit_transform(y_train.reshape(-1,1))

mlc=MLPRegressor(hidden_layer_sizes=(7,4,2),activation="tanh",random_state=2022)
mlc.fit(X_train,y_train.ravel())
#np.ravel() which is used to change a 2-dimensional array or a multi-dimensional array into a contiguous flattened array.
y_pred=mlc.predict(X_test)
y_pred_inv = scalarY.inverse_transform(y_pred.reshape(-1, 1))
print(mean_absolute_error(y_test, y_pred))
print(mean_absolute_error(y_test, y_pred_inv))
print(mean_squared_error(y_test, y_pred))
print(mean_squared_error(y_test, y_pred_inv))
print(r2_score(y_test, y_pred))
print(r2_score(y_test, y_pred_inv))

##################TUNING###########
F = housing_dum.iloc[:,1:]
R = housing_dum.iloc[:,0]
F = F.values
R = R.values
F= scalarX.fit_transform(F)
R = scalarY.fit_transform(R.reshape(-1,1))

params=dict(hidden_layer_sizes=[(4,3,2),(3,2),(5,4,3,2),(10,5,3)],
            activation=["logistic","tanh"],
            learning_rate_init=np.linspace(0.001,0.8,5),
            learning_rate=["constant","adaptive","invscaling"])
mlpc=MLPRegressor(random_state=2022)
kfold=KFold(n_splits=5,shuffle=(True),random_state=(2022))
cv=GridSearchCV(mlpc, param_grid=params,scoring='r2',cv=kfold,verbose=2)

cv.fit(F,R.ravel())
print(cv.best_score_)
print(cv.best_params_)
model=cv.best_estimator_
print(model.coefs_)
print(model.intercepts_)

