# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 19:40:55 2022

@author: R
"""

import pandas as pd
from sklearn.model_selection import cross_val_score,KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

ConcreteStrength = pd.read_csv(r"G:\Ddrive\PG DBDA\12 Practical Machine Learning_/Cases\Concrete Strength\Concrete_Data.csv")
# no need for dummy as all fields are numbers only
X=ConcreteStrength.iloc[:,:-1].values
y=ConcreteStrength.iloc[:,-1].values
Scaler=StandardScaler()
X_scaled=Scaler.fit_transform(X)
lin_reg=LinearRegression()

X_train,X_test,y_train,y_test=train_test_split(X_scaled,y,test_size=0.3,random_state=2024)
lin_reg.fit(X_train,y_train)
y_pred=lin_reg.predict(X_test)
res=r2_score(y_test, y_pred)
print(f"R2 score: {res.mean()}")
#R2 score: 0.6395029542692768


kfold=KFold(n_splits=5,shuffle=True,random_state=(2024))
res=cross_val_score(lin_reg,X_scaled,y,cv=kfold,scoring='r2')
print(f"R2 score: {res.mean()}")
#R2 score: 0.604193074066034