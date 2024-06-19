# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 11:53:31 2022

@author: R
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder


glass=pd.read_csv("G:/Ddrive/PG DBDA/12 Practical Machine Learning_/Cases/Glass Identification/Glass.csv")
F = glass.iloc[:,:-1].values
R = glass.iloc[:,-1].values
X_train,X_test,y_train,y_test=train_test_split(F,R,test_size=0.2,random_state=2024)

log_reg=LogisticRegression()
svc_rad=SVC(kernel='rbf',random_state=2024,probability=(True))
svc_lin=SVC(kernel='linear',random_state=2024,probability=(True))
dtc=DecisionTreeClassifier(random_state=(2024))
knn=KNeighborsClassifier()
xgbc=XGBClassifier(random_state=2024)

stack=StackingClassifier(stack_method='predict_proba',estimators=[('LOG_REG',log_reg),
                                                                  ('SVC_RAD',svc_rad),
                                                                  ('SVD_LIN',svc_lin),
                                                                  ('DTC',dtc),
                                                                  ('KNN',knn)],final_estimator=xgbc)
stack.fit(X_train,y_train)
label=[1, 2, 3, 5, 6, 7]
y_pred_stack=stack.predict(X_test)
y_pred_stack_prob=stack.predict_proba(X_test)
print(log_loss(y_test,y_pred_stack_prob,labels=label)) # 0.8735067563869787

stack=StackingClassifier(stack_method='predict_proba',estimators=[('LOG_REG',log_reg),
                                                                  ('SVC_RAD',svc_rad),
                                                                  ('SVD_LIN',svc_lin),
                                                                  ('DTC',dtc),
                                                                  ('KNN',knn)],final_estimator=xgbc,passthrough=(True))
stack.fit(X_train,y_train)
y_pred_stack_prob=stack.predict_proba(X_test)
print(log_loss(y_test,y_pred_stack_prob,labels=label)) # 0.8515152255356521

