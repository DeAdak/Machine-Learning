# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 16:57:53 2022

@author: R
"""

import pandas as pd
from sklearn.model_selection import train_test_split,StratifiedKFold,cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,roc_auc_score,confusion_matrix,classification_report
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis
df=pd.read_csv(r"G:\Ddrive\PG DBDA\12 Practical Machine Learning_/Cases/Vehicle Silhouettes/Vehicle.csv")
# Image_Segmention.csv has several Response, all are categorical. so dummy will not be supported by train_test_splits
X = df.iloc[:,:-1]
y = df.iloc[:,-1]
# LabelEncoder() will numberify all the respnses as 1,2,3.... It is now can be processed by train_test_splits
le = LabelEncoder()
y_le = le.fit_transform(y)
X_train,X_test,y_train,y_test=train_test_split(X,y_le,test_size=0.3,random_state=2024,stratify=y_le)
lda=LinearDiscriminantAnalysis()
qda=QuadraticDiscriminantAnalysis()
gnb=GaussianNB()
lda.fit(X_train,y_train)
qda.fit(X_train,y_train)
gnb.fit(X_train,y_train)

y_pred=lda.predict(X_test)
y_pred_prob = lda.predict_proba(X_test)
print(roc_auc_score(y_test,y_pred_prob,multi_class='ovr')) #0.9363119245363718
print(accuracy_score(y_test, y_pred)) #0.7637795275590551
#print(confusion_matrix(y_test, y_pred))
#print(classification_report(y_test, y_pred))

y_pred1=qda.predict(X_test)
y_pred_prob1 = qda.predict_proba(X_test)
print(roc_auc_score(y_test,y_pred_prob1,multi_class='ovr')) #0.9640891676499381
print(accuracy_score(y_test, y_pred1)) #0.8503937007874016
#print(confusion_matrix(y_test, y_pred1))
#print(classification_report(y_test, y_pred1))

y_pred2=gnb.predict(X_test)
y_pred_prob2 = gnb.predict_proba(X_test)
print(roc_auc_score(y_test,y_pred_prob2,multi_class='ovr')) #0.7403179698961034
print(accuracy_score(y_test, y_pred2)) #0.452755905511811
#print(confusion_matrix(y_test, y_pred2))
#print(classification_report(y_test, y_pred2))

############################### TASK ##########################################
lda=LinearDiscriminantAnalysis()
kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=2024)
cv=cross_val_score(lda,X,y_le,scoring='neg_log_loss',cv=kfold,verbose=2)
print(cv.mean()) #-0.4891667496513392

qda=QuadraticDiscriminantAnalysis()
kfold=StratifiedKFold(n_splits=(5),shuffle=(True),random_state=2024)
cv=cross_val_score(qda,X,y_le,scoring='neg_log_loss',cv=kfold,verbose=2)
print(cv.mean()) #-0.4029702890518914

gnb=GaussianNB()
kfold=StratifiedKFold(n_splits=(5),shuffle=(True),random_state=2024)
cv=cross_val_score(gnb,X,y_le,scoring='neg_log_loss',cv=kfold,verbose=2)
print(cv.mean()) #-2.5289795460152122



