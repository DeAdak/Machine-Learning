# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 12:05:56 2022

@author: R
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 11:31:03 2022

@author: R
"""

import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
#from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score,accuracy_score
from sklearn.neighbors import KNeighborsClassifier
BankruptcyData = pd.read_csv(r"C:\Users\R\Downloads\12 Practical Machine Learning_\Cases\Bankruptcy\Bankruptcy.csv")
# no need for dummy as all fields are numbers only
BankruptcyData
X = BankruptcyData.iloc[:,2:]
y = BankruptcyData.iloc[:,1]
X_np=X.values
y_np=y.values
# pandas to numpy conversion, pd_df --> np_array
param_grid={'n_neighbors':[1,3,5,7,9,11,13,15,17,19,21,23,25,27,29]}
#param_grid is a dictonary, its values will be used as no of neighbors, don't go beyond 25, it overfits the data
kFold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 2022)
knn=KNeighborsClassifier()
gscv=GridSearchCV(knn, param_grid,scoring='roc_auc',cv=kFold)
gscv.fit(X_np,y_np)
# .fit will exchage the values as no of neighbor and compute
pd_df=pd.DataFrame(gscv.cv_results_)
#Dictonary --> df conversion as the result produced as Dictonary 

gscv.best_params_
gscv.best_score_
