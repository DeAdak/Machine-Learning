# -*- coding: utf-8 -*-
"""
Created on Fri May 10 23:08:51 2024

@author: R
"""

# Perform the necessary imports
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
\
freq = pd.read_csv(r"G:/Ddrive/PG DBDA/12 Practical Machine Learning_/Cases/Recency Frequency Monetary/rfm_data_customer.csv",index_col=0)

freq.drop(['most_recent_visit'],axis = 1,inplace = True)
scaler = StandardScaler()
freqscaled=scaler.fit_transform(freq)

################ Elbow ############################
clustNos = [2,3,4,5,6,7,8,9,10]
Inertia = []

for i in clustNos :
    model = KMeans(n_clusters=i,random_state=2024)
    model.fit(freqscaled)
    Inertia.append(model.inertia_)

import matplotlib.pyplot as plt
plt.plot(clustNos, Inertia, '-o')
plt.title("Scree Plot")
plt.xlabel('Number of clusters, k')
plt.ylabel('Inertia')
plt.xticks(clustNos)
plt.show()  #n_clusters=5

################ silhouette_score ############################
clustNos = [2,3,4,5,6,7,8,9,10]
silhouette = []

for i in clustNos :
    model = KMeans(n_clusters=i,random_state=2024)
    model.fit(freqscaled)
    labels = model.predict(freqscaled)
    sil_score = silhouette_score(freqscaled,labels)
    silhouette.append(sil_score)

import matplotlib.pyplot as plt
plt.plot(clustNos, silhouette, '-o')
plt.title("Scree Plot")
plt.xlabel('Number of clusters, k')
plt.ylabel('silhouette')
plt.xticks(clustNos)
plt.show() #n_clusters=3
###############################################
# Create a KMeans instance with clusters: Best k model
model = KMeans(n_clusters=5,random_state=2024)

# Fit model to points
model.fit(freqscaled)

#model.n_init
# Determine the cluster labels of new_points: labels
labels = model.predict(freqscaled)


clusterID = pd.DataFrame({'ClustID':labels},index=freq.index)
clusteredData = pd.concat([freq,clusterID],
                          axis=1)

clusteredData.groupby('ClustID').mean()
clusteredData.sort_values('ClustID').head()
clusteredData.sort_values('ClustID').to_csv(r'G:\Ddrive\PG DBDA\12 Practical Machine Learning_\dayWise\UnSupervised\K-Means\Recency_Frequency.csv')





