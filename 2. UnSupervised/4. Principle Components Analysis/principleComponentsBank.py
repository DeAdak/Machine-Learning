# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 10:55:03 2022

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

bankruptcy=pd.read_csv(r"G:/Ddrive/PG DBDA/12 Practical Machine Learning_/Cases/Bankruptcy/Bankruptcy.csv")

F = bankruptcy.iloc[:,2:]
R = bankruptcy.iloc[:,1]
X_train,X_test,y_train,y_test=train_test_split(F,R,test_size=0.3,random_state=2022,stratify=(R))
stdscl=StandardScaler()
std_bank=stdscl.fit_transform(X_train)
pca=PCA()
principleComponents = pca.fit_transform(std_bank)
principleFs=pd.DataFrame(principleComponents[:,:10])
pca.explained_variance_ #individual variance
np.sum(pca.explained_variance_) #total variance
pca.explained_variance_ratio_ #individual variance/total variance
pca.explained_variance_ratio_ * 100
np.cumsum(pca.explained_variance_ratio_ * 100)

X_test_scaled=stdscl.transform(X_test)
X_test_pca=pca.transform(X_test_scaled)
X_test_pca_pd=pd.DataFrame(X_test_pca[:,:10])
pca.explained_variance_ #individual variance
np.sum(pca.explained_variance_) #total variance
pca.explained_variance_ratio_ #individual variance/total variance
pca.explained_variance_ratio_ * 100


ys=pca.explained_variance_ratio_ * 100
xs=np.arange(1,26)
plt.plot(xs,ys)
plt.show()

ys=np.cumsum(pca.explained_variance_ratio_ * 100)
xs=np.arange(1,26)
plt.plot(xs,ys)
plt.show()
# 1st 10 Fs are contributing more than 95%, so we can ditch the rest Fs

################### Plotting the PCs ######################
df_PC = pd.DataFrame(principleComponents[:,:2],
                     index=X_train.index,
                     columns=["PC1","PC2"])
df_PC = pd.concat([df_PC,y_train],axis=1)


sns.scatterplot(x="PC1", y="PC2", hue='D',data=df_PC)
plt.show()

log_reg=LogisticRegression()
log_reg.fit(principleFs,y_train)
y_pred=log_reg.predict(X_test_pca_pd)
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(roc_auc_score(y_test, y_pred))


