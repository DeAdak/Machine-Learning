# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 22:43:28 2022

@author: R
"""

import pandas as pd
from sklearn import tree
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split,KFold,GridSearchCV
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,roc_auc_score
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB


df=pd.read_csv(r"G:\Ddrive\PG DBDA/12 Practical Machine Learning_/Cases/Wisconsin/BreastCancer.csv")
df_dum=pd.get_dummies(df,drop_first=True)
F = df_dum.iloc[:,1:-1]
R = df_dum.iloc[:,-1]
X_train,X_test,y_train,y_test=train_test_split(F,R,train_size=0.3,stratify=R,random_state=2024,shuffle=True)

svm=SVC(random_state=(2024),probability=(True))
gnb=GaussianNB()
log_reg=LogisticRegression()

models=[('svm',svm),('GNB',gnb),('log_reg',log_reg)]
vote=VotingClassifier(estimators=models,voting='soft',verbose=(2))
vote.fit(X_train,y_train)
y_pred=vote.predict(X_test)
y_pred_prob=vote.predict_proba(X_test)[:,1]
print(accuracy_score(y_test, y_pred)) # 0.9673469387755103
print(roc_auc_score(y_test, y_pred_prob)) #0.9931980313001161
