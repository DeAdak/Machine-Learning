# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 19:51:56 2022

@author: R
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,roc_auc_score
from sklearn.linear_model import LogisticRegression
import seaborn as sns

satellite=pd.read_csv(r"G:/Ddrive/PG DBDA/12 Practical Machine Learning_/Cases/Satellite Imaging/Satellite.csv",sep=';')

F = satellite.iloc[:,:-1]
R = satellite.iloc[:,-1]
X_train,X_test,y_train,y_test=train_test_split(F,R,test_size=0.3,random_state=2022,stratify=(R))
scalar = StandardScaler()
std_satellite = scalar.fit_transform(X_train)
pca=PCA()
principleComponents = pca.fit_transform(std_satellite)
principleFs=pd.DataFrame(principleComponents[:,:10])
pca.explained_variance_ #individual variance
np.sum(pca.explained_variance_) #total variance
pca.explained_variance_ratio_ #individual variance/total variance
pca.explained_variance_ratio_ * 100

ys=pca.explained_variance_ratio_ * 100
xs=np.arange(1,37)
plt.plot(xs,ys)
plt.show()

ys=np.cumsum(pca.explained_variance_ratio_ * 100)
xs=np.arange(1,37)
plt.plot(xs,ys)
plt.show()

X_test_scaled=scalar.transform(X_test)
X_test_pca=pca.transform(X_test_scaled)
X_test_pca_pd=pd.DataFrame(X_test_pca[:,:10])
pca.explained_variance_ #individual variance
np.sum(pca.explained_variance_) #total variance
pca.explained_variance_ratio_ #individual variance/total variance
pca.explained_variance_ratio_ * 100




ys=np.cumsum(pca.explained_variance_ratio_ * 100)
xs=np.arange(1,37)
plt.plot(xs,ys)
plt.show()
# 1st 10 Fs are contributing more than 95%, so we can ditch the rest Fs

################### Plotting the PCs ######################
df_PC = pd.DataFrame(principleComponents[:,:2],
                     index=X_train.index,
                     columns=["PC1","PC2"])
df_PC = pd.concat([df_PC,y_train],axis=1)


sns.scatterplot(x="PC1", y="PC2", hue=R ,data=df_PC)
plt.show()

log_reg=LogisticRegression()
log_reg.fit(principleFs,y_train)
y_pred=log_reg.predict(X_test_pca_pd)
y_pred_proba = log_reg.predict_proba(X_test_pca_pd)
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(roc_auc_score(y_test, y_pred_proba,multi_class='ovr'))
