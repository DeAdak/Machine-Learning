# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 09:25:01 2024

@author: R
"""

from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
companies = pd.read_csv(r"G:/Ddrive/PG DBDA/12 Practical Machine Learning_/dayWise/UnSupervised/Linkage/Australia_largest_companies.csv",index_col=1)

scaler = StandardScaler()
companiesScaled=scaler.fit_transform(companies)

companiesScaled = pd.DataFrame(companiesScaled,
                          columns=companies.columns,
                          index=companies.index)

# Calculate the linkage: mergings
mergings = linkage(companiesScaled,method='average')
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 12
fig_size[1] = 8
plt.rcParams["figure.figsize"] = fig_size
# Plot the dendrogram, using varieties as labels
dendrogram(mergings,
           labels=np.array(companies.index),
           leaf_rotation=90,
           leaf_font_size=10,
)

plt.show()

####### Using Mahalonobis Distance Method #############

# Calculate the linkage: mergings
mergings = linkage(companiesScaled,method='average',
                   metric='mahalanobis')

# Plot the dendrogram, using varieties as labels
dendrogram(mergings,
           labels=np.array(companies.index),
           leaf_rotation=90,
           leaf_font_size=10,
)
plt.show()
