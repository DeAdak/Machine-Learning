# -*- coding: utf-8 -*-
"""
Created on Sat May 11 20:27:45 2024

@author: R
"""
from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd

USArrests = pd.read_csv(r"G:/Ddrive/PG DBDA/12 Practical Machine Learning_/Datasets/USArrests.csv",index_col=0)

from sklearn.preprocessing import StandardScaler
# Create scaler: scaler
scaler = StandardScaler()
USArrestsscaled=scaler.fit_transform(USArrests)

clust_DB = DBSCAN(eps=0.5, min_samples=2)
clust_DB.fit(USArrestsscaled)
print(clust_DB.labels_) # -1, 0, 1, 2, 3. -1 is outlier

clusterID = pd.DataFrame({'ClustID':clust_DB.labels_},index=USArrests.index)
clusteredData = pd.concat([USArrests,clusterID],axis='columns')

clusteredData.groupby('ClustID').mean()
clusteredData.sort_values('ClustID')

from sklearn.metrics import silhouette_score
silhouette_score(USArrestsscaled,clust_DB.labels_)

eps_range = [0.2,0.4,0.6,1,1.1,1.5]
mp_range = [2,3,4,5,6,7]
cnt = 0
a =[]
for i in eps_range:
    for j in mp_range:
        clust_DB = DBSCAN(eps=i, min_samples=j)
        clust_DB.fit(USArrestsscaled)
        if len(set(clust_DB.labels_)) >= 2:
            cnt = cnt + 1
            sil_sc = silhouette_score(USArrestsscaled,clust_DB.labels_)
            a.append([cnt,i,j,sil_sc])
            print(f'eps_range:{i},mp_range:{j},silhouette_score:{sil_sc}')
    
a = np.array(a)
pa = pd.DataFrame(a,columns=['Sr','eps','min_pt','sil'])
print("Best Paramters:")
pa[pa['sil'] == pa['sil'].max()]
#       Sr  eps  min_pt       sil
# 14  15.0  1.1     5.0  0.390737

clust_DB = DBSCAN(eps=1.1, min_samples=5)
clust_DB.fit(USArrestsscaled)
print(clust_DB.labels_) # -1, 0. -1 is outlier

clusterID = pd.DataFrame({'ClustID':clust_DB.labels_},index=USArrests.index)
clusteredData = pd.concat([USArrests,clusterID],axis='columns')

clusteredData.groupby('ClustID').mean()
clusteredData.sort_values('ClustID')
