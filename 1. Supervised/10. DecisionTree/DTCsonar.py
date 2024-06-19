# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 18:32:27 2022

@author: R
"""

import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split,KFold,GridSearchCV
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,plot_confusion_matrix
from sklearn.tree import DecisionTreeClassifier,plot_tree
import matplotlib.pyplot as plt
import graphviz

df=pd.read_csv(r"G:\Ddrive\PG DBDA/12 Practical Machine Learning_/Cases/Sonar/Sonar.csv")
df.Class.value_counts()
# Class
# M    111
# R     97
df_dum=pd.get_dummies(df,drop_first=True)
F = df_dum.iloc[:,:-1]
R = df_dum.iloc[:,-1]
dtc=DecisionTreeClassifier(random_state=2024,max_depth=5)

####kFold####
depth_range=[3,4,5,6,7,8,9]
minsplit=[5,10,15,20,25,30]
minleaf=[5,10,15]
param=dict(max_depth=depth_range,min_samples_leaf=minleaf,min_samples_split=minsplit)
kfold=KFold(n_splits=5,shuffle=(True),random_state=(2024))
cv=GridSearchCV(dtc, param_grid=param,scoring='roc_auc',cv=kfold,verbose=2)
cv.fit(F,R)
print(cv.best_params_) # {'max_depth': 4, 'min_samples_leaf': 15, 'min_samples_split': 5}
print(cv.best_score_) # 0.7939841176282825
best_model = cv.best_estimator_

plt.figure(figsize=(15,10))
plot_tree(best_model,feature_names=F.columns,class_names=['R','M'],filled=True,rounded=True)
plt.show()

dot_data=tree.export_graphviz(best_model,feature_names=F.columns,class_names=['Benign','Malignant'],filled=(True),rounded=(True),special_characters=(True))
graph=graphviz.Source(dot_data)
graph

####Feature Importances#####
## Feature influence on Gini's index ##
print(best_model.feature_importances_)

plt.figure(figsize=(15,10))
ind=np.arange(60)
plt.bar(ind,best_model.feature_importances_)
plt.xticks(ind,F.columns,rotation=90)
plt.title('Features')
plt.xlabel('Variables')
plt.show()