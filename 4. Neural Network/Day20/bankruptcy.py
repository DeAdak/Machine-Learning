# -*- coding: utf-8 -*-
"""Bankruptcy.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1l_pKgp19YLHue51IQOpbhH5jxeRDt9NF
"""



from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import tensorflow as tf
import numpy as np

df = pd.read_csv(r"G:/Ddrive/PG DBDA/12 Practical Machine Learning_/Cases/Bankruptcy/Bankruptcy.csv")
df.head()

F = df.iloc[:,2:]
R = df.iloc[:,1]

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(F, R, test_size = 0.3, 
                                                    random_state=2024,stratify=R)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)    
X_test = scaler.transform(X_test)

(X_train.shape, y_train.shape)

y_train = y_train.values
y_test = y_test.values

tf.random.set_seed(2024)
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(4, activation='relu',input_shape=(X_train.shape[1], )), 
    tf.keras.layers.Dense(1, activation='sigmoid')  
])

model.variables

model.compile(optimizer='sgd', loss='binary_crossentropy',metrics=['accuracy'])

print(model.summary())

history = model.fit( X_train,y_train,validation_data=(X_test,y_test),verbose=2,epochs=500)

model.get_weights()

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline

import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

from sklearn.metrics import log_loss
y_pred_prob = model.predict(X_test)
log_loss(y_true=y_test,y_pred=y_pred_prob)

from sklearn.metrics import accuracy_score
predict_probs= model.predict(X_test)
predict_probs[:5]

predict_classes = np.where(predict_probs>=0.5,1,0)
predict_classes[:5]

acc = accuracy_score(y_test,predict_classes)
print(f"Accuracy: {acc}")