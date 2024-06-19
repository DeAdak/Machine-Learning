# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 08:22:04 2022

@author: R
"""

import h2o
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
h2o.init()

df=h2o.import_file("C:/Users/R/Downloads/12 Practical Machine Learning_/Cases/Bankruptcy/Bankruptcy.csv",destination_frame="Bankruptcy")

R = 'D'
F = df.col_names[2:]

df['D'] = df['D'].asfactor()
df['D'].levels()
train, test = df.split_frame(ratios=[0.7],seed=2022,destination_frames=['train', 'test'])
print(df.shape)
print(train.shape)
print(test.shape)
glm_model=H2OGeneralizedLinearEstimator(family='binomial',model_id='Logistic',training_frame=train,validation_frame=test)
glm_model.train(x=F,y=R)
glm_model.auc()
glm_model.confusion_matrix()
y_pred=glm_model.predict(test_data=test)
y_pred_df = y_pred.as_data_frame()
h2o.cluster().shutdown()
