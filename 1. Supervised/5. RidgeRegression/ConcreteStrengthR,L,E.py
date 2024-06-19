import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge,Lasso,ElasticNet
ConcreteStrength = pd.read_csv(r"G:\Ddrive\PG DBDA\12 Practical Machine Learning_/Cases\Concrete Strength\Concrete_Data.csv")
# no need for dummy as all fields are numbers only
X=ConcreteStrength.iloc[:,:-1].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y=ConcreteStrength.iloc[:,-1].values
kfold=KFold(n_splits=5,shuffle=True,random_state=(2024))
ridge=Ridge()
lasso=Lasso()
enet=ElasticNet()


param_grid={'alpha':np.linspace(0,10,40)}
res=GridSearchCV(lasso,param_grid=param_grid, cv=kfold,scoring='r2')
res.fit(X_scaled,y)
res.cv_results_
res.best_params_
#'alpha': 0.0
print(f"R2 score: {res.best_score_}")
#R2 score: 0.604193074066034


param_grid={'alpha':[1,1.5,2,2.5,3,3.5,4]}
#OR
param_grid={'alpha':np.linspace(0,10,30)}
res=GridSearchCV(ridge,param_grid=param_grid, cv=kfold,scoring='r2')
res.fit(X_scaled,y)
res.cv_results_
res.best_params_
#'alpha': 2.0689655172413794
print(f"R2 score: {res.best_score_}")
#R2 score: 0.6042962298409532


param_grid={'alpha':np.linspace(0,10,30),'l1_ratio':np.linspace(0.001,1,10)}
res=GridSearchCV(enet,param_grid=param_grid, cv=kfold,scoring='r2')
res.fit(X_scaled,y)
res_df=pd.DataFrame(res.cv_results_)
res.best_params_
#'alpha': 0.0, 'l1_ratio': 0.001
print(f"R2 score: {res.best_score_}")
#R2 score: 0.604193074066034













