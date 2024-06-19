# -*- coding: utf-8 -*-
"""
Created on Sat May 11 12:20:55 2024

@author: R
"""
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold,cross_val_score

bankruptcy = pd.read_csv(r"G:/Ddrive/PG DBDA/12 Practical Machine Learning_/Cases/Bankruptcy/Bankruptcy.csv")
X = bankruptcy.iloc[:,2:]
y = bankruptcy.iloc[:,1]

scaler = StandardScaler()
XScaled=scaler.fit_transform(X)

#############  LogisticRegression  ####################

log_reg=LogisticRegression(random_state=2024)
kfold=StratifiedKFold(n_splits=5,random_state=2024,shuffle=(True))
res = cross_val_score(log_reg,XScaled,y,scoring='roc_auc',cv = kfold,verbose=3)
print(res.mean()) #0.8755705832628908

#############  LinearRegression with KMeans  ####################
clustNos = [2,3,4,5,6,7,8,9,10]
Inertia = []

for i in clustNos :
    model = KMeans(n_clusters=i,random_state=2024)
    model.fit(XScaled)
    Inertia.append(model.inertia_)

plt.plot(clustNos, Inertia, '-o')
plt.title("Scree Plot")
plt.xlabel('Number of clusters, k')
plt.ylabel('Inertia')
plt.xticks(clustNos)
plt.show()

# Create a KMeans instance with clusters: Best k model
model = KMeans(n_clusters=5,random_state=2024)

# Fit model to points
model.fit(XScaled)

#model.n_init
# Determine the cluster labels of new_points: labels
labels = model.predict(XScaled)

clusterID = pd.DataFrame({'ClustID':labels})

XScaled_pd = pd.DataFrame(XScaled,columns=X.columns,index=X.index)
clusteredData = pd.concat([XScaled_pd,clusterID],
                          axis='columns')

clusteredData['ClustID'] = clusteredData['ClustID'].astype('category')

X_new = pd.get_dummies(clusteredData,drop_first=True)

log_reg=LogisticRegression(random_state=2024)
kfold=StratifiedKFold(n_splits=5,random_state=2024,shuffle=(True))
res = cross_val_score(log_reg,X_new,y,scoring='roc_auc',cv = kfold,verbose=3)
print(res.mean()) #0.8934911242603552
