# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 09:34:25 2022

@author: Debabrata Adak
"""
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold,train_test_split
#from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,roc_auc_score
cancerDataDummy = pd.read_csv(r"C:\Users\R\Downloads\12 Practical Machine Learning_\Cases\Wisconsin\BreastCancer.csv")
cancerData = pd.get_dummies(cancerDataDummy, drop_first = True)
X = cancerData.iloc[:,1:-1]
y = cancerData.iloc[:,-1]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=2022,stratify=y)

knn = KNeighborsClassifier(n_neighbors=3)
# n_neighbors value to get a suitable result and choose that value for k
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)
y_pred_prob=knn.predict_proba(X_test)[:,1]
print(accuracy_score(y_test, y_pred))
print(roc_auc_score(y_test, y_pred))


