import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold,GridSearchCV

df=pd.read_csv(r"G:/Ddrive/PG DBDA/12 Practical Machine Learning_/Cases/King City Housing/kc_house_data.csv")
F = df.drop(['id', 'date', 'price','zipcode'],axis = 1)
R = df['price']
rfr=RandomForestRegressor(random_state=2024,verbose=3)
param={'max_features':[6,10,15,20]}
kfold=KFold(n_splits=(5),shuffle=(True),random_state=2024)
cv=GridSearchCV(rfr, param_grid=param,scoring='r2',cv=kfold,verbose=2)
cv.fit(F,R)
print(cv.best_score_) #0.8747915836339892
print(cv.best_params_) #{'max_features': 15}

best_model=cv.best_estimator_
import matplotlib.pyplot as plt
ind = np.arange(F.shape[1])
plt.figure(figsize=(15,10))
plt.bar(ind,best_model.feature_importances_)
plt.xticks(ind,(F.columns),rotation=90)
plt.title('Feature Importance')
plt.xlabel("Variables")
plt.show()

imp_ind=np.argsort(best_model.feature_importances_)
F.columns[imp_ind]
