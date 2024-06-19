# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 22:41:03 2022

@author: R
"""


import pandas as pd
import tensorflow as tf
import numpy as np

df = pd.read_csv("C:/Users/R/Downloads/12 Practical Machine Learning_/Cases/Glass Identification/Glass.csv")
df.head()

y = df.iloc[:,-1]
y
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y_encoded = le.fit_transform(y)
y_encoded

X = df.iloc[:,:-1]
X

from sklearn.preprocessing import MinMaxScaler
scalerX = MinMaxScaler()
X_scaled = scalerX.fit_transform(X)
X_scaled

#y=y.values

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.3, 
                                                    random_state=2022)

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from keras import metrics
import keras

tf.random.set_seed(seed=2022)
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(9, activation='relu',input_shape=(X_train.shape[1], )), 
    tf.keras.layers.Dropout(rate=0.2,seed=2022),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(alpha=0.1),
    tf.keras.layers.Dense(7),
    tf.keras.layers.Dropout(rate=0.1,seed=2022),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ELU(alpha=0.1),
    tf.keras.layers.Dense(5),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(3, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(6, activation='softmax')  
])

print(model.summary())
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit( X_train,y_train,validation_data=(X_test,y_test),verbose=2,epochs=100)

from sklearn.metrics import log_loss,accuracy_score
y_pred_prob = model.predict(X_test)
log_loss(y_true=y_test,y_pred=y_pred_prob)

y_pred_proba = np.argmax(y_pred_prob,axis =1)
y_test=y_test.values
print(accuracy_score(y_test,y_pred_proba))

loss, acc = model.evaluate(X_test, y_test,verbose=0)
print('Test loss = {:.4f} '.format(loss))
print('Test acc = {:.4f} '.format(acc))

tf.random.set_seed(seed=2022)
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(25, activation='relu',input_shape=(X_train.shape[1], )), 
    tf.keras.layers.Dropout(rate=0.2,seed=2022),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dropout(rate=0.1,seed=2022),
    tf.keras.layers.Dense(10, activation='relu'), 
    tf.keras.layers.Dense(5, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  
])
model.compile(optimizer='sgd', loss='binary_crossentropy',metrics=['accuracy'])

print(model.summary())

model.fit( X_train,y_train,validation_data=(X_test,y_test),verbose=2,epochs=200)

from sklearn.metrics import log_loss
y_pred_prob = model.predict(X_test)
log_loss(y_true=y_test,y_pred=y_pred_prob)
