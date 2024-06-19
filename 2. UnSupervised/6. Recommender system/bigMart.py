# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 09:25:53 2022

@author: R
"""


import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold,GridSearchCV
#from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from math import sqrt 


big_mart1=pd.read_csv(r"C:/Users/R/Downloads/12 Practical Machine Learning_/Cases/Big Mart Sales Prediction/train_v9rqX0R.csv")
big_mart=big_mart1
big_mart["Item_Weight"].fillna(big_mart["Item_Weight"].median(),inplace=True)
big_mart["Item_Fat_Content"].replace(['LF','reg'],['Low Fat','Regular'],inplace=True)
big_mart["Item_Visibility"].replace(0,big_mart["Item_Visibility"].median(),inplace=True)
big_mart.drop("Outlet_Establishment_Year",axis=1,inplace=True)
Outlet_Sizes=big_mart.pivot_table(values="Outlet_Size",columns="Outlet_Type",aggfunc=(lambda x:x.mode()[0]))
Outlet_Sizes
missing_Outlet_Size = big_mart["Outlet_Size"].isnull()
missing_Outlet_Size.value_counts()
big_mart.loc[missing_Outlet_Size,"Outlet_Size"]=big_mart.loc[missing_Outlet_Size,"Outlet_Type"].apply(lambda x:Outlet_Sizes[x])

#big_mart["Outlet_Size"].fillna(big_mart["Outlet_Size"].mode()[0],inplace=True)
#big_mart.isnull().sum()
#big_mart.describe()

big_mart["Outlet_Size"].value_counts()

#sns.countplot(y="Outlet_Size",data=big_mart)

encode=LabelEncoder()

big_mart["Item_Identifier"]=encode.fit_transform(big_mart["Item_Identifier"])
big_mart["Item_Fat_Content"]=encode.fit_transform(big_mart["Item_Fat_Content"])
big_mart["Item_Type"]=encode.fit_transform(big_mart["Item_Type"])
big_mart["Outlet_Identifier"]=encode.fit_transform(big_mart["Outlet_Identifier"])
big_mart["Outlet_Size"]=encode.fit_transform(big_mart["Outlet_Size"])
big_mart["Outlet_Location_Type"]=encode.fit_transform(big_mart["Outlet_Location_Type"])
big_mart["Outlet_Type"]=encode.fit_transform(big_mart["Outlet_Type"])

F=big_mart.drop("Item_Outlet_Sales",axis=1)
R=big_mart["Item_Outlet_Sales"]

##############################################
big_mart_test=pd.read_csv(r"C:/Users/R/Downloads/12 Practical Machine Learning_/Cases/Big Mart Sales Prediction/test_AbJTz2l.csv")
big_mart_test["Item_Weight"].fillna(big_mart["Item_Weight"].median(),inplace=True)
big_mart_test["Item_Fat_Content"].replace(['LF','reg'],['Low Fat','Regular'],inplace=True)
big_mart_test["Item_Visibility"].replace(0,big_mart["Item_Visibility"].median(),inplace=True)
big_mart_test.drop("Outlet_Establishment_Year",axis=1,inplace=True)
big_mart_test.loc[missing_Outlet_Size,"Outlet_Size"]=big_mart_test.loc[missing_Outlet_Size,"Outlet_Type"].apply(lambda x:Outlet_Sizes[x])
big_mart_test["Outlet_Size"].value_counts()
#big_mart_test["Outlet_Size"].fillna(big_mart_test["Outlet_Size"].mode()[0],inplace=True)


big_mart_test["Item_Identifier"]=encode.fit_transform(big_mart_test["Item_Identifier"])
big_mart_test["Item_Fat_Content"]=encode.fit_transform(big_mart_test["Item_Fat_Content"])
big_mart_test["Item_Type"]=encode.fit_transform(big_mart_test["Item_Type"])
big_mart_test["Outlet_Identifier"]=encode.fit_transform(big_mart_test["Outlet_Identifier"])
big_mart_test["Outlet_Size"]=encode.fit_transform(big_mart_test["Outlet_Size"])
big_mart_test["Outlet_Location_Type"]=encode.fit_transform(big_mart_test["Outlet_Location_Type"])
big_mart_test["Outlet_Type"]=encode.fit_transform(big_mart_test["Outlet_Type"])

xgbr=XGBRegressor()
xgbr.fit(F,R)

prediction = xgbr.predict(big_mart_test)
prediction_df = pd.DataFrame(prediction)

prediction_df.rename(columns={0:'Item_Outlet_Sales'},inplace=True)
# big_test=pd.read_csv(r"C:/Users/R/Downloads/12 Practical Machine Learning_/Cases/Big Mart Sales Prediction/test_AbJTz2l.csv")

# big_mart_pred=pd.concat([big_test,prediction_df],ignore_index=True)



