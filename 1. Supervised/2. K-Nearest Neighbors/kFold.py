# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 09:34:25 2022

@author: Debabrata Adak
"""
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
cancerDataDummy = pd.read_csv(r"C:\Users\R\Downloads\12 Practical Machine Learning_\Cases\Wisconsin\BreastCancer.csv")
cancerData = pd.get_dummies(cancerDataDummy, drop_first = True)
cancerData
X = cancerData.iloc[:,1:-1]
y = cancerData.iloc[:,-1]
X
y
kFold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 2022)
gnb=GaussianNB()
result = cross_val_score(gnb, X,y,scoring="roc_auc",cv=kFold)
# cv = cross validation
result.mean()

