# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 15:41:41 2022

@author: R
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split,GridSearchCV,KFold
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
housing=pd.read_csv(r"C:/Users/R/Downloads/12 Practical Machine Learning_/Cases/Medical Cost Personal/insurance.csv")
R = housing.iloc[:,-1]
F_dum = housing.drop("charges",axis=1)
F=pd.get_dummies(F_dum,drop_first=True)
scalarX = MinMaxScaler()
scalarY = MinMaxScaler()

F = F.values
R = R.values
X_train,X_test,y_train,y_test=train_test_split(F,R,test_size=0.3,random_state=2022)
X_train = scalarX.fit_transform(X_train)
y_train = scalarY.fit_transform(y_train.reshape(-1,1))

mlpr=MLPRegressor(hidden_layer_sizes=(7,4,2),activation="tanh",random_state=2022)
mlpr.fit(X_train,y_train.ravel())
y_pred=mlpr.predict(X_test)
print(mean_absolute_error(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))
print(r2_score(y_test, y_pred))

##################TUNING###########
params=dict(hidden_layer_sizes=[ (20,10,5), (15,7,2), (12,6) ],
            learning_rate_init=[0.001,0.1,0.4,0.5],
            learning_rate=["constant","adaptive","invscaling"])
mlpr=MLPRegressor(random_state=2022)
kfold=KFold(n_splits=5,shuffle=(True),random_state=(2022))
cv=GridSearchCV(mlpr, param_grid=params,scoring='r2',cv=kfold,verbose=2)
R = housing.iloc[:,-1]
F = housing.drop("charges",axis=1)
F=pd.get_dummies(F,drop_first=True)
scalarX = MinMaxScaler()
scalarY = MinMaxScaler()

F = F.values
R = R.values
F= scalarX.fit_transform(F)
R = scalarY.fit_transform(R.reshape(-1,1))
cv.fit(F,R.ravel())
print(cv.best_score_)
print(cv.best_params_)
model=cv.best_estimator_
print(model.coefs_)
print(model.intercepts_)
####################################
test=pd.read_csv(r"C:/Users/R/Downloads/12 Practical Machine Learning_/Cases/Medical Cost Personal/tst_insure.csv")
test_dum=pd.get_dummies(test,drop_first=True)
#test=test.values
scaled_test= scalarX.transform(test_dum)
y_pred=model.predict(scaled_test)
y_decode = scalarY.inverse_transform(y_pred.reshape(-1,1))
