# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 19:42:25 2022

@author: R
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 


nutrient=pd.read_csv(r"C:/Users/R/Downloads/12 Practical Machine Learning_/Datasets/nutrient.csv",index_col=(0))
Scaler=StandardScaler()
std_nutrient=Scaler.fit_transform(nutrient)
pca=PCA()
principleComponents = pca.fit_transform(std_nutrient)
pca.explained_variance_ #individual variance
np.sum(pca.explained_variance_) #total variance
pca.explained_variance_ratio_ #individual variance/total variance
pca.explained_variance_ratio_ * 100

ys=pca.explained_variance_ratio_ * 100
xs=np.arange(1,6)
plt.plot(xs,ys)
plt.show()

ys=np.cumsum(pca.explained_variance_ratio_ * 100)
xs=np.arange(1,6)
plt.plot(xs,ys)
plt.show()

df = pd.DataFrame(principleComponents,columns=['PC1','PC2','PC3','PC4','PC5'],index=nutrient.index)
pca_loadings = pd.DataFrame(pca.components_.T, index=nutrient.columns, columns=['V1', 'V2','V3','V4','V5'] )
pca_loadings

##########################################################
nutrientscaled = pd.DataFrame(std_nutrient,columns=nutrient.columns, index=nutrient.index)
##########################################################
### From: https://github.com/teddyroland/python-biplot/blob/master/biplot.py
import seaborn as sns
 
# Scatter plot based and assigne color based on 'label - y'
sns.lmplot('PC1', 'PC2', data=df, fit_reg = False, size = 15, scatter_kws={"s": 100})
 
# set the maximum variance of the first two PCs
# this will be the end point of the arrow of each **original features**
xvector = pca.components_[0]
yvector = pca.components_[1]
 
# value of the first two PCs, set the x, y axis boundary
xs = df['PC1']
ys = df['PC2']
 
## visualize projections
 
## Note: scale values for arrows and text are a bit inelegant as of now,
##       so feel free to play around with them
for i in range(len(xvector)):
    # arrows project features (ie columns from csv) as vectors onto PC axes
    # we can adjust length and the size of the arrow
    plt.arrow(0, 0, xvector[i]*max(xs), yvector[i]*max(ys),
              color='r', width=0.005, head_width=0.05)
    plt.text(xvector[i]*max(xs)*1.1, yvector[i]*max(ys)*1.1,
             list(nutrient.columns.values)[i], color='r')
 
for i in range(len(xs)):
    plt.text(xs[i]*1.08, ys[i]*1.08, list(nutrient.index)[i], color='b') # index number of each observations
plt.title('PCA Plot of first PCs')
plt.show()