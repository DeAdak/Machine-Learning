# Perform the necessary imports
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

milk = pd.read_csv(r"G:/Ddrive/PG DBDA/12 Practical Machine Learning_/Datasets/milk.csv",index_col=0)

scaler = StandardScaler()
milkscaled=scaler.fit_transform(milk)

milkscaled = pd.DataFrame(milkscaled,
                          columns=milk.columns,
                          index=milk.index)


################ Elbow ############################
clustNos = [2,3,4,5,6,7,8,9,10]
Inertia = []

for i in clustNos :
    model = KMeans(n_clusters=i,random_state=2022)
    model.fit(milkscaled)
    Inertia.append(model.inertia_)

import matplotlib.pyplot as plt
plt.plot(clustNos, Inertia, '-o')
plt.title("Scree Plot")
plt.xlabel('Number of clusters, k')
plt.ylabel('Inertia')
plt.xticks(clustNos)
plt.show() #n_clusters=4

################ silhouette_score ############################
clustNos = [2,3,4,5,6,7,8,9,10]
silhouette = []

for i in clustNos :
    model = KMeans(n_clusters=i,random_state=2022)
    model.fit(milkscaled)
    labels = model.predict(milkscaled)
    sil_score = silhouette_score(milkscaled,labels)
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
model = KMeans(n_clusters=4,random_state=2024)

# Fit model to points
model.fit(milkscaled)

#model.n_init
# Determine the cluster labels of new_points: labels
labels = model.predict(milkscaled)


clusterID = pd.DataFrame({'ClustID':labels},index=milk.index)
clusteredData = pd.concat([milk,clusterID],
                          axis=1)

clusteredData.groupby('ClustID').mean()
clusteredData.sort_values('ClustID')





