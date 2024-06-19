# -*- coding: utf-8 -*-
"""Untitled6.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1FnLGd9gAAmkvPD9lrqEBrWKBNYIWJmHI
"""

import pandas as pd
import numpy as np
import tensorflow as tf

df=pd.read_csv("/content/train.csv")
df

X = df.iloc[:,1:-3]
X

y = df.iloc[:,-3:-1]
y

from sklearn.preprocessing import MinMaxScaler
scalerx=MinMaxScaler()
scalery=MinMaxScaler()
X_scaled=scalerx.fit_transform(X)
y_scaled=scalery.fit_transform(y)

X_scaled

y_scaled

y1 = y_scaled[:,0]
y2 = y_scaled[:,1]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X_scaled,y_scaled,test_size=0.1,random_state=2022)

tf.random.set_seed(2022)
model=tf.keras.models.Sequential([
                tf.keras.layers.Dense(8,activation="relu",input_shape=(X_train.shape[1],)),
                tf.keras.layers.Dense(6,activation="relu"),
                tf.keras.layers.Dense(3,activation="relu"),
                tf.keras.layers.Dense(2,activation="relu")
                                  ])

model.compile(optimizer="adam",loss=tf.keras.losses.MeanSquaredError())

history=model.fit(X_train,y_train,validation_data=(X_test,y_test),verbose=2,epochs=500)

import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

model.compile(optimizer="adam",loss=tf.keras.losses.MeanSquaredError())

from tensorflow.keras.callbacks import EarlyStopping
monitor = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=20, verbose=2, mode='auto',
        restore_best_weights=True)
history2 = model.fit(X_train,y_train,validation_data=(X_test,y_test),callbacks=[monitor],verbose=2,epochs=500)

import matplotlib.pyplot as plt
plt.plot(history2.history['loss'])
plt.plot(history2.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

y_pred = model.predict(X_test)

y_pred_inv=scalery.inverse_transform(y_pred)

from sklearn.metrics import mean_absolute_error
print(f"MAE:{np.sqrt(mean_absolute_error(y_test[:,0],y_pred[:,0]))}")

from sklearn.metrics import mean_squared_error
print(f"RMSE:{np.sqrt(mean_squared_error(y_test[:,0],y_pred[:,0]))}")

from sklearn.metrics import r2_score
print(f"R2_Score:{r2_score(y_test[:,0],y_pred[:,0])}")

y_test_inv=scalery.inverse_transform(y_test)

from sklearn.metrics import mean_absolute_error
print(f"MAE:{np.sqrt(mean_absolute_error(y_test_inv[:,0],y_pred_inv[:,0]))}")

from sklearn.metrics import mean_squared_error
print(f"RMSE:{np.sqrt(mean_squared_error(y_test_inv[:,0],y_pred_inv[:,0]))}")

from sklearn.metrics import r2_score
print(f"R2:{r2_score(y_test_inv[:,0],y_pred_inv[:,0])}")