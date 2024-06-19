# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 17:55:14 2022

@author: R
"""
import pandas as pd

Housing = pd.read_csv(r"G:\Ddrive\PG DBDA\12 Practical Machine Learning_/Cases\Real Estate\Housing.csv")
HousingDummy=pd.get_dummies(Housing,drop_first=True)
X = HousingDummy.iloc[:,1:]
y = HousingDummy.iloc[:,0]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=2022)

from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X_train, y_train)
print(f"intercept: {lin_reg.intercept_}")
print(f"coefficient: {lin_reg.coef_}")
y_pred=lin_reg.predict(X_test)
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
print(f"R2 score: {r2_score(y_test, y_pred)}")
print(f'mean_absolute_error: {mean_absolute_error(y_test, y_pred)}')
print(f'mean_squared_error: {mean_squared_error(y_test, y_pred)}')
