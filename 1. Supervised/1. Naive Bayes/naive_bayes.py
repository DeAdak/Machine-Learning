# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 21:25:52 2024

@author: R
"""
import pandas as pd
telecom = pd.read_csv(r"G:\Ddrive\PG DBDA\12 Practical Machine Learning_/Cases/Telecom/Telecom.csv")
from sklearn.naive_bayes import MultinomialNB,BernoulliNB,GaussianNB
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,roc_auc_score
telecom=pd.get_dummies(telecom,drop_first=True)
X = telecom.iloc[:,:-1]
Y = telecom.iloc[:,-1]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,train_size=0.7,random_state=2022,stratify=Y)

mnb = MultinomialNB()
mnb.fit(x_train, y_train)
y_pred = mnb.predict(x_test)

print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(roc_auc_score(y_test, y_pred))

cancer = pd.read_csv(r'G:\Ddrive\PG DBDA\12 Practical Machine Learning_\Cases\Cancer\Cancer.csv')
cancer = pd.get_dummies(cancer,drop_first=True)
X = cancer.iloc[:,1:-1]
Y = cancer.iloc[:,-1]
x_train,x_test,y_train,y_test = train_test_split(X,Y,train_size=0.7,random_state=2022,stratify=Y)

mnb = MultinomialNB()
mnb.fit(x_train, y_train)
y_pred_prob = mnb.predict_proba(x_test)[:,-1]
y_pred = mnb.predict(x_test)

print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(roc_auc_score(y_test, y_pred_prob))

bnb = BernoulliNB()
bnb.fit(x_train,y_train)
y_pred_prob = mnb.predict_proba(x_test)[:,-1]
y_pred=bnb.predict(x_test)

print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(roc_auc_score(y_test, y_pred_prob))

b_cancer = pd.read_csv(r'G:/Ddrive/PG DBDA/12 Practical Machine Learning_/Cases/Wisconsin/BreastCancer.csv')
X=b_cancer.iloc[:,1:-1]
Y=b_cancer.iloc[:,-1]
x_train,x_test,y_train,y_test = train_test_split(X,Y,train_size=0.7,random_state=2022,stratify=Y)
gnb = GaussianNB()
gnb.fit(x_train,y_train)

y_pred=gnb.predict(x_test)
y_pred_prob = gnb.predict_proba(x_test)[:,-1]
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(roc_auc_score(y_test, y_pred_prob))


b_cancer = pd.read_csv(r'G:/Ddrive/PG DBDA/12 Practical Machine Learning_/Cases/Wisconsin/BreastCancer.csv')
b_cancer = pd.get_dummies(b_cancer,drop_first=True)
X=b_cancer.iloc[:,1:-1]
Y=b_cancer.iloc[:,-1]
x_train,x_test,y_train,y_test = train_test_split(X,Y,train_size=0.7,random_state=2022,stratify=Y)
gnb = GaussianNB()
gnb.fit(x_train,y_train)

y_pred=gnb.predict(x_test)
y_pred_prob = gnb.predict_proba(x_test)[:,-1]
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(roc_auc_score(y_test, y_pred_prob))

from sklearn.preprocessing import LabelEncoder
img=pd.read_csv(r'G:/Ddrive/PG DBDA/12 Practical Machine Learning_/Cases/Image Segmentation/Image_Segmention.csv')
X = img.iloc[:,1:]
Y = img.iloc[:,0]
le = LabelEncoder()
Y = le.fit_transform(Y)
x_train,x_test,y_train,y_test=train_test_split(X,Y,train_size=0.7,random_state=2024,stratify=Y)
gnb=GaussianNB()
gnb.fit(x_train,y_train)

y_pred=gnb.predict(x_test)
y_pred_prob = gnb.predict_proba(x_test)
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(roc_auc_score(y_test, y_pred_prob,multi_class='ovo'))
print(roc_auc_score(y_test, y_pred_prob,multi_class='ovr'))


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,roc_auc_score

glass = pd.read_csv(r'G:/Ddrive/PG DBDA/12 Practical Machine Learning_/Cases/Glass Identification/Glass.csv')
X=glass.iloc[:,:-1]
Y=glass.iloc[:,-1]
x_train,x_test,y_train,y_test = train_test_split(X,Y,random_state=2022,stratify=Y,train_size=0.7)
gnb = GaussianNB()
gnb.fit(x_train,y_train)
y_pred=gnb.predict(x_test)
y_pred_prob=gnb.predict_proba(x_test)
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(roc_auc_score(y_test, y_pred_prob,multi_class='ovo'))
print(roc_auc_score(y_test, y_pred_prob,multi_class='ovr'))

























































