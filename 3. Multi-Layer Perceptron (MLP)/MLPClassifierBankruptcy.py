# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 15:14:32 2022

@author: R
"""
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split,GridSearchCV,StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

bankruptcy=pd.read_csv(r"G:\Ddrive\PG DBDA/12 Practical Machine Learning_/Cases/Bankruptcy/Bankruptcy.csv")
#RidingMowers_dum=pd.get_dummies(RidingMowers,drop_first=True)
F = bankruptcy.iloc[:,2:]
R = bankruptcy.iloc[:,1]

scalar=MinMaxScaler()
F=scalar.fit_transform(F)
X_train,X_test,y_train,y_test=train_test_split(F,R,test_size=0.3,random_state=2024,stratify=R)

mlc=MLPClassifier(hidden_layer_sizes=(3,2),activation="tanh",random_state=2024)
mlc.fit(X_train,y_train)
y_pred=mlc.predict(X_test)
print(accuracy_score(y_test, y_pred)) #0.65
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

##################TUNING###########
params=dict(hidden_layer_sizes=[ (20,10,5), (15,7,2), (12,6) ],
            activation=["logistic","tanh"],
            learning_rate_init=[0.001,0.1,0.4,0.5],
            learning_rate=["constant","adaptive","invscaling"])
mlpc=MLPClassifier(random_state=2022)
kfold=StratifiedKFold(n_splits=5,shuffle=(True),random_state=(2024))
cv=GridSearchCV(mlpc, param_grid=params,scoring='roc_auc',cv=kfold,verbose=2)
cv.fit(F,R)
print(cv.best_score_) #0.8765004226542688
print(cv.best_params_) #{'activation': 'tanh', 'hidden_layer_sizes': (20, 10, 5), 'learning_rate': 'constant', 'learning_rate_init': 0.001}
model=cv.best_estimator_
print(model.coefs_)
print(model.intercepts_)








