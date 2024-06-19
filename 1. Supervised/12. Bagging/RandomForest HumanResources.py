# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 17:40:13 2022

@author: R
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold,GridSearchCV

df=pd.read_csv(r"G:/Ddrive/PG DBDA/12 Practical Machine Learning_/Cases/human-resources-analytics/HR_comma_sep.csv",sep=',')
F_dum = df.drop(['left'],axis = 1)
R = df['left']
F = pd.get_dummies(F_dum,drop_first=True)
rfc=RandomForestClassifier(random_state=2024)
param={'max_features':[3,4,6,8,10]}
kfold=KFold(n_splits=(5),shuffle=(True),random_state=2024)
cv=GridSearchCV(rfc, param_grid=param,scoring='roc_auc',cv=kfold,verbose=2)
cv.fit(F,R)
print(cv.best_score_)
print(cv.best_params_)
best_model=cv.best_estimator_

import matplotlib.pyplot as plt
ind = np.arange(F.shape[1])
plt.figure(figsize=(15,10))
plt.bar(ind,best_model.feature_importances_)
plt.xticks(ind,(F.columns),rotation=90)
plt.title('Feature Importance')
plt.xlabel("Variables")
plt.show()

imp_ind=np.argsort(best_model.feature_importances_)
F.columns[imp_ind]
