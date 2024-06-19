# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 14:42:21 2022

@author: R
"""

import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV,train_test_split,cross_val_score
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
ConcreteStrength = pd.read_csv(r"G:\Ddrive\PG DBDA\12 Practical Machine Learning_/Cases\Concrete Strength\Concrete_Data.csv")
# no need for dummy as all fields are numbers only
X=ConcreteStrength.iloc[:,:8]
y=ConcreteStrength.iloc[:,8]
poly2=PolynomialFeatures(degree=2)
X_poly2=poly2.fit_transform(X)
poly3=PolynomialFeatures(degree=3)
X_poly3=poly3.fit_transform(X)
X_train,X_test,y_train,y_test=train_test_split(X_poly2,y,random_state=2024,train_size=0.3)

reg=LinearRegression()
reg.fit(X_train, y_train)
y_pred=reg.predict(X_test)
print(r2_score(y_test, y_pred)) #0.7216303797572006

kfold=KFold(n_splits=5,shuffle=True,random_state=2024)
print(f"R2 score: {cross_val_score(reg, X_poly2,y,scoring='r2',cv=kfold).mean()}")
#R2 score: 0.780175469774023

############################################################
ridge=Ridge()
lasso=Lasso()
enet=ElasticNet()
param_grid={'alpha':[0,0.001,0.1,0.5,0.8,1,1.5,2]}
kfold=KFold(n_splits=5,shuffle=True,random_state=2024)

############################# Lasso with Degree 2 and 3 ###############################
res=GridSearchCV(lasso,param_grid=param_grid, cv=kfold,scoring='r2')
res.fit(X_poly2,y)
res.cv_results_
res.best_params_
#{'alpha': 0.1}
print(f"R2 score: {res.best_score_}")
#R2 score: 0.7759139387956807

res.fit(X_poly3,y)
res.cv_results_
res.best_params_
#{'alpha': 0.1}
print(f"R2 score: {res.best_score_}")
#R2 score: 0.8491753581624369

############################# Ridge with Degree 2 and 3 ###############################
res=GridSearchCV(ridge,param_grid=param_grid, cv=kfold,scoring='r2')
res.fit(X_poly2,y)
res.cv_results_
res.best_params_
#{'alpha': 0} --> use LinearRegression
print(f"R2 score: {res.best_score_}")
#R2 score: 0.7802431756203717

res=GridSearchCV(ridge,param_grid=param_grid, cv=kfold,scoring='r2')
res.fit(X_poly3,y)
res.cv_results_
res.best_params_
#{'alpha': 2}
print(f"R2 score: {res.best_score_}")
#R2 score: 0.8486958853535669

############################# ElasticNet with Degree 2 and 3 ###############################
param_grid=dict(alpha=[0,0.001,0.1,0.5,0.8,1,1.5,2],l1_ratio=[0.001,0.5,0.7,1])
res=GridSearchCV(enet,param_grid=param_grid, cv=kfold,scoring='r2')
res.fit(X_poly2,y)
res.cv_results_
res.best_params_
#{'alpha': 0.1, 'l1_ratio': 0.7}
print(f"R2 score: {res.best_score_}")
#R2 score: 0.7760024495974511


res.fit(X_poly3,y)
res.cv_results_
res.best_params_
#{'alpha': 0.1, 'l1_ratio': 1}
print(f"R2 score: {res.best_score_}")
#R2 score: 0.8491753581624369