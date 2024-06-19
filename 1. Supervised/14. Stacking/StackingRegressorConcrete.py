# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 10:35:32 2022

@author: R
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import StackingRegressor,RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

house=pd.read_csv("G:/Ddrive/PG DBDA/12 Practical Machine Learning_/Cases/Concrete Strength/Concrete_Data.csv")
F = house.iloc[:,:-1]
R = house.iloc[:,-1]
X_train,X_test,y_train,y_test=train_test_split(F,R,test_size=0.3,random_state=2024)

lr=LinearRegression()
lasso=Lasso(random_state=2024)
ridge=Ridge(random_state=2024)
rfr=RandomForestRegressor(random_state=2024)
dtr=DecisionTreeRegressor(random_state=(2024))
xgbr=XGBRegressor(random_state=2024)

stack=StackingRegressor(estimators=[('LR',lr),('LASSO',lasso),('RIDGE',ridge),('RFR',rfr),('DTR',dtr)],final_estimator=xgbr)
stack.fit(X_train,y_train)
y_pred_stack=stack.predict(X_test)
print(r2_score(y_test,y_pred_stack)) #0.9015488276763075

stack=StackingRegressor(estimators=[('LR',lr),('LASSO',lasso),('RIDGE',ridge),('RFR',rfr),('DTR',dtr)],final_estimator=xgbr
                        ,passthrough=(True))
stack.fit(X_train,y_train)
y_pred_stack=stack.predict(X_test)
print(r2_score(y_test,y_pred_stack)) #0.9279293293670872