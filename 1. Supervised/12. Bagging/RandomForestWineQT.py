# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 18:18:44 2022

@author: R
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold,GridSearchCV

df=pd.read_csv(r"G:/Ddrive/PG DBDA/12 Practical Machine Learning_/Cases/WineQT.csv")
F = df.iloc[:,:-2]
R = df.iloc[:,-2]
rfc=RandomForestClassifier(random_state=2024)
param={'max_features':[1,2,3,4,6,8,10]}
kfold=KFold(n_splits=(5),shuffle=(True),random_state=2024)
cv=GridSearchCV(rfc, param_grid=param,scoring='neg_log_loss',cv=kfold,verbose=2)
cv.fit(F,R)
print(cv.best_score_)
print(cv.best_params_) #{'max_features': 1}

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
