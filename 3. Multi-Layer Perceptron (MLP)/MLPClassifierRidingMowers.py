# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 12:42:37 2022

@author: R
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.model_selection import train_test_split,GridSearchCV,StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

RidingMowers=pd.read_csv(r"G:\Ddrive\PG DBDA/12 Practical Machine Learning_/Datasets/RidingMowers.csv")
RidingMowers_dum=pd.get_dummies(RidingMowers)
RidingMowers_dum.drop("Response_Not Bought",axis=1,inplace=True)
F = RidingMowers_dum.iloc[:,:-1]
R_le = RidingMowers_dum.iloc[:,-1]
le = LabelEncoder()
R = le.fit_transform(R_le)


scalar=MinMaxScaler()
F=scalar.fit_transform(F)
X_train,X_test,y_train,y_test=train_test_split(F,R,test_size=0.3,random_state=2024,stratify=R)

mlc=MLPClassifier(hidden_layer_sizes=(3,2),activation="tanh",random_state=2024)
mlc.fit(X_train,y_train)
y_pred=mlc.predict(X_test)
print(accuracy_score(y_test, y_pred)) #0.9074074074074074
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

##################TUNING###########
params=dict(hidden_layer_sizes=[(4,3,2),(3,2),(5,4,3,2),(10,5,3)],
            activation=["logistic","tanh"],
            learning_rate_init=np.linspace(0.001,0.8,5),
            learning_rate=["constant","adaptive","invscaling"])
mlpc=MLPClassifier(random_state=2022)
kfold=StratifiedKFold(n_splits=5,shuffle=(True),random_state=(2022))
cv=GridSearchCV(mlpc, param_grid=params,scoring='roc_auc',cv=kfold,verbose=2)
cv.fit(F,R)
print(cv.best_score_) #0.9564213564213565
print(cv.best_params_) #{'activation': 'tanh', 'hidden_layer_sizes': (5, 4, 3, 2), 'learning_rate': 'constant', 'learning_rate_init': 0.20075}
model=cv.best_estimator_
print(model.coefs_)
print(model.intercepts_)








