# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 20:48:49 2024

@author: R
"""

from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
import numpy as np 
y_test = np.array([0,0,0,1,0,1,1,0,1,0,0,1])
y_pred = np.array([0,0,0,0,1,1,0,0,1,1,0,0])
print(confusion_matrix(y_test, y_pred))
print(f'accuracy_score: {accuracy_score(y_test, y_pred)}')
print(classification_report(y_test, y_pred))
 