# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 09:34:25 2022

@author: Debabrata Adak
"""
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
#from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
cancerDataDummy = pd.read_csv(r"C:\Users\R\Downloads\12 Practical Machine Learning_\Cases\Wisconsin\BreastCancer.csv")
cancerData = pd.get_dummies(cancerDataDummy, drop_first = True)
cancerData
X = cancerData.iloc[:,1:-1]
y = cancerData.iloc[:,-1]
X_np=X.values
y_np=y.values
# pandas to numpy conversion, pd_df --> np_array
kFold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 2022)
knn=KNeighborsClassifier(n_neighbors=1)
result = cross_val_score(knn, X_np,y_np,scoring="roc_auc",cv=kFold)
# KNeighborsClassifier needs numpy array to operate.
# cv = cross validation
result
result.mean()

