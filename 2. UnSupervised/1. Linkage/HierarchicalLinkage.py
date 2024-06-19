# Perform the necessary imports
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

milk = pd.read_csv(r"G:/Ddrive/PG DBDA/12 Practical Machine Learning_/Datasets/milk.csv",index_col=0)

scaler = StandardScaler()
milkscaled=scaler.fit_transform(milk)

milkscaled = pd.DataFrame(milkscaled,
                          columns=milk.columns,
                          index=milk.index)

# Calculate the linkage: mergings
mergings = linkage(milkscaled,method='average')

# method=’single’
# This is also known as the Nearest Point Algorithm.
# method=’complete’
# This is also known by the Farthest Point Algorithm or Voor Hees Algorithm.
# method=’average’
# This is also known as the UPGMA algorithm.
# method=’weighted’
# This is also known as the WPGMA algorithm.
# method=’centroid’
# This is also known as the UPGMC algorithm.
# method=’median’
# This is also known as the WPGMC algorithm.
# method=’ward’
# This is also known as the incremental algorithm.

fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 12
fig_size[1] = 8
plt.rcParams["figure.figsize"] = fig_size
# Plot the dendrogram, using varieties as labels
dendrogram(mergings,
           labels=np.array(milk.index),
           leaf_rotation=45,
           leaf_font_size=10,
)

plt.show()

####### Using Mahalonobis Distance Method #############

# Calculate the linkage: mergings
mergings = linkage(milkscaled,method='average',
                   metric='mahalanobis')

# Plot the dendrogram, using varieties as labels
dendrogram(mergings,
           labels=np.array(milk.index),
           leaf_rotation=60,
           leaf_font_size=10,
)
plt.show()



######################################################################


USArrests = pd.read_csv(r"G:/Ddrive/PG DBDA/12 Practical Machine Learning_/Datasets/USArrests.csv",index_col=0)
scaler = StandardScaler()
USArrests_scaled=scaler.fit_transform(USArrests)

USArrests_scaled = pd.DataFrame(USArrests_scaled,
                          columns=USArrests.columns,
                          index=USArrests.index)

# Calculate the linkage: mergings
mergings = linkage(USArrests_scaled,method='average')
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 12
fig_size[1] = 8
plt.rcParams["figure.figsize"] = fig_size
# Plot the dendrogram, using varieties as labels
dendrogram(mergings,
           labels=np.array(USArrests.index),
           leaf_rotation=90,
           leaf_font_size=10,
)

plt.show()

####### Using Mahalonobis Distance Method #############

# Calculate the linkage: mergings
mergings = linkage(USArrests_scaled,method='average',
                   metric='mahalanobis')

# Plot the dendrogram, using varieties as labels
dendrogram(mergings,
           labels=np.array(USArrests_scaled.index),
           leaf_rotation=90,
           leaf_font_size=10,
)
plt.show()