# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 09:37:41 2022

@author: R
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score

house=pd.read_csv("G:/Ddrive/PG DBDA/12 Practical Machine Learning_/Cases/Real Estate/Housing.csv")
F = house.iloc[:,1:]
R = house.iloc[:,0]
X = pd.get_dummies(F,drop_first=True)
X_train,X_test,y_train,y_test=train_test_split(X,R,test_size=0.3,random_state=2024)

lr=LinearRegression()
lasso=Lasso(random_state=2024)
ridge=Ridge(random_state=2024)
enet=ElasticNet(random_state=2024)
dtr=DecisionTreeRegressor(random_state=(2024))
gbr=GradientBoostingRegressor(random_state=2024)

stack=StackingRegressor(estimators=[('LR',lr),
                                    ('LASSO',lasso),
                                    ('RIDGE',ridge),
                                    ('ENET',enet),
                                    ('DTR',dtr)],
                        final_estimator=gbr,verbose=2)
stack.fit(X_train,y_train)
y_pred_stack=stack.predict(X_test)
print(r2_score(y_test,y_pred_stack))# 0.5911403316516949

stack=StackingRegressor(estimators=[('LR',lr),
                                    ('LASSO',lasso),
                                    ('RIDGE',ridge),
                                    ('ENET',enet),
                                    ('DTR',dtr)],
                        final_estimator=gbr
                        ,passthrough=(True))
stack.fit(X_train,y_train)
y_pred_stack=stack.predict(X_test)
print(r2_score(y_test,y_pred_stack)) #0.5884644886108255


