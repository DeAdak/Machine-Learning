# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 19:48:59 2022

@author: R
"""

import pandas as pd

df=pd.read_csv(r"C:/Users/R/Downloads/12 Practical Machine Learning_/Cases/MNIST - Digits/full_mnist_train.csv")
df.head()
label = df.iloc[:,0]
df_np = df.iloc[:,1:].values
df_np= df_np.reshape(60000,28,28)
imageID = 599
image = df_np[imageID]
label[imageID]
import matplotlib.pyplot as plt
import matplotlib.cm as cm
plt.imshow(image,cmap = cm.gray)
plt.show()

df_image = pd.DataFrame(df_np[imageID,:,:])
df_image.to_csv(r"C:\Users\R\Downloads\12 Practical Machine Learning_\dayWise\Day22\7.csv")
