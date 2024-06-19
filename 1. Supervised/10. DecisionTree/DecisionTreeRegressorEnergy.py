# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 21:42:30 2022

@author: R
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 21:15:19 2022

@author: R
"""

import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split,KFold,GridSearchCV,cross_val_score
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,plot_confusion_matrix
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.tree import DecisionTreeRegressor,plot_tree
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge,Lasso,ElasticNet

energy=pd.read_excel(r"G:\Ddrive\PG DBDA/12 Practical Machine Learning_/Cases/Energy Efficiency/ENB2012_data.xlsx")
F = energy.iloc[:,:-2]
R1 = energy.iloc[:,-2]
R2= energy.iloc[:,-1]

####kFold####
##DecisionTreeRegressor
depth_range=[None,8,3]
minsplit=[2,10,20]
minleaf=[1,5,10]
dtr=DecisionTreeRegressor()
param=dict(max_depth=depth_range,min_samples_leaf=minleaf,min_samples_split=minsplit)
kfold=KFold(n_splits=5,shuffle=(True),random_state=(2024))
cv1=GridSearchCV(dtr, param_grid=param,scoring='r2',cv=kfold,verbose=2,)
cv1.fit(F,R1)
print(cv1.best_params_) #{'max_depth': 8, 'min_samples_leaf': 1, 'min_samples_split': 2}
print(cv1.best_score_) # 0.9973533048865484
cv2=GridSearchCV(dtr, param_grid=param,scoring='r2',cv=kfold,verbose=2,)
cv2.fit(F,R2)
print(cv2.best_params_)#{'max_depth': None, 'min_samples_leaf': 10, 'min_samples_split': 2}
print(cv2.best_score_)# 0.9628729531246805


best_model1=cv1.best_estimator_
best_model2=cv2.best_estimator_
imp=np.argsort(best_model1.feature_importances_)
F.columns[imp]
imp=np.argsort(best_model2.feature_importances_)
F.columns[imp]

####Feature Importances#####
import matplotlib.pyplot as plt
print(best_model1.feature_importances_)
ind=np.arange(8)
plt.figure(figsize=(15,10))
plt.bar(ind,best_model1.feature_importances_)
plt.xticks(ind,F.columns,rotation=45)
plt.title('Features')
plt.xlabel('Variables')
plt.show()

print(best_model2.feature_importances_)
ind=np.arange(8)
plt.figure(figsize=(15,10))
plt.bar(ind,best_model2.feature_importances_)
plt.xticks(ind,F.columns,rotation=45)
plt.title('Features')
plt.xlabel('Variables')
plt.show()

##Linear Regression K-Fold CV##
lin_reg=LinearRegression()
kfold=KFold(n_splits=5,shuffle=True,random_state=(2024))
res1=cross_val_score(lin_reg,F,R1,cv=kfold)
print(res1.mean()) #0.9129315626717103
res2=cross_val_score(lin_reg,F,R2,cv=kfold)
print(res2.mean()) #0.8830951985578002

##Elasticnet##
enet=ElasticNet()
param_grid=dict(alpha=[0,0.1,0.5,1,1.5],l1_ratio=[0,0.5,1])
res=GridSearchCV(enet,param_grid=param_grid, cv=kfold,scoring='r2')
res.fit(F,R1)
#r2_score is negative so this model is algo in not suitable
res.best_params_ #{'alpha': 0, 'l1_ratio': 0}
res.best_score_ #0.9139809131344506
res.fit(F,R2)
#r2_score is negative so this model is algo in not suitable
res.best_params_ #{'alpha': 0, 'l1_ratio': 0}
res.best_score_ #0.8829845018181333

