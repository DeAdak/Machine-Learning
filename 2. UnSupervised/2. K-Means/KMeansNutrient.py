# Perform the necessary imports
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

nutrient = pd.read_csv(r"G:/Ddrive/PG DBDA/12 Practical Machine Learning_/Datasets/nutrient.csv",index_col=0)

scaler = StandardScaler()
nutrientscaled=scaler.fit_transform(nutrient)

nutrientscaled = pd.DataFrame(nutrientscaled,
                          columns=nutrient.columns,
                          index=nutrient.index)


################ Elbow ############################
clustNos = [2,3,4,5,6,7,8,9,10]
Inertia = []

for i in clustNos :
    model = KMeans(n_clusters=i,random_state=2024)
    model.fit(nutrientscaled)
    Inertia.append(model.inertia_)

import matplotlib.pyplot as plt
plt.plot(clustNos, Inertia, '-o')
plt.title("Scree Plot")
plt.xlabel('Number of clusters, k')
plt.ylabel('Inertia')
plt.xticks(clustNos)
plt.show()
###############################################
# Create a KMeans instance with clusters: Best k model
model = KMeans(n_clusters=5,random_state=2024)

# Fit model to points
model.fit(nutrientscaled)

#model.n_init
# Determine the cluster labels of new_points: labels
labels = model.predict(nutrientscaled)


clusterID = pd.DataFrame({'ClustID':labels},index=nutrient.index)
clusteredData = pd.concat([nutrient,clusterID],axis=1)

clusteredData.groupby('ClustID').mean()
clusteredData.sort_values('ClustID')





