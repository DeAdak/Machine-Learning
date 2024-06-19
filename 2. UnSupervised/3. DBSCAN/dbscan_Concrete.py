# -*- coding: utf-8 -*-
"""
Created on Sat May 11 20:36:51 2024

@author: R
"""

# -*- coding: utf-8 -*-
"""
Created on Sat May 11 20:27:45 2024

@author: R
"""
from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd

concrete = pd.read_csv(r"G:/Ddrive/PG DBDA/12 Practical Machine Learning_/Cases/Concrete Strength/Concrete_Data.csv")

from sklearn.preprocessing import StandardScaler
# Create scaler: scaler
scaler = StandardScaler()
concreteScaled=scaler.fit_transform(concrete)

clust_DB = DBSCAN(eps=0.5, min_samples=2)
clust_DB.fit(concreteScaled)
print(clust_DB.labels_) # -1, 0, 1, 2, 3. -1 is outlier

clusterID = pd.DataFrame({'ClustID':clust_DB.labels_},index=concrete.index)
clusteredData = pd.concat([concrete,clusterID],axis='columns')
clusteredData['ClustID'].unique() # -1 to 228
clusteredData.groupby('ClustID').mean()
clusteredData.sort_values('ClustID')

from sklearn.metrics import silhouette_score
silhouette_score(concreteScaled,clust_DB.labels_)

eps_range = [0.2,0.4,0.6,1,1.1,1.5]
mp_range = [2,3,4,5,6,7]
cnt = 0
a =[]
for i in eps_range:
    for j in mp_range:
        clust_DB = DBSCAN(eps=i, min_samples=j)
        clust_DB.fit(concreteScaled)
        if len(set(clust_DB.labels_)) >= 2:
            cnt = cnt + 1
            sil_sc = silhouette_score(concreteScaled,clust_DB.labels_)
            a.append([cnt,i,j,sil_sc])
            #print(f'eps_range:{i},mp_range:{j},silhouette_score:{sil_sc}')
    
a = np.array(a)
pa = pd.DataFrame(a,columns=['Sr','eps','min_pt','sil'])
print("Best Paramters:")
pa[pa['sil'] == pa['sil'].max()]
#       Sr  eps  min_pt       sil
# 10  11.0  0.6     2.0  0.203616

clust_DB = DBSCAN(eps=0.6, min_samples=2)
clust_DB.fit(concreteScaled)
print(clust_DB.labels_) # -1, 0. -1 is outlier

clusterID = pd.DataFrame({'ClustID':clust_DB.labels_},index=concrete.index)
clusteredData = pd.concat([concrete,clusterID],axis='columns')
clusteredData['ClustID'].unique() # -1 to 222
clusteredData.groupby('ClustID').mean()
clusteredData.sort_values('ClustID')
