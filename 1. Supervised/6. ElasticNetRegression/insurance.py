import pandas as pd
from sklearn.model_selection import cross_val_score,KFold, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
insurance_dummy = pd.read_csv(r"G:\Ddrive\PG DBDA\12 Practical Machine Learning_/Cases/Medical Cost Personal/insurance.csv")
y=insurance_dummy.iloc[:,-1]
insurance=pd.get_dummies(insurance_dummy,drop_first=True)
X=insurance.iloc[:,:-1]

reg=LinearRegression()
kfold=KFold(n_splits=5,random_state=2024,shuffle=True)
results=cross_val_score(reg, X,y,cv=kfold,scoring='r2')
print(results.mean())

poly=PolynomialFeatures(degree=2)
X_poly=poly.fit_transform(X)
reg=LinearRegression()
kfold=KFold(n_splits=5,random_state=2024,shuffle=True)
results=cross_val_score(reg, X_poly,y,cv=kfold,scoring='r2')
print(results.mean())

poly=PolynomialFeatures(degree=3)
X_poly=poly.fit_transform(X)
reg=LinearRegression()
kfold=KFold(n_splits=5,random_state=2024,shuffle=True)
results=cross_val_score(reg, X_poly,y,cv=kfold,scoring='r2')
print(results.mean())

poly=PolynomialFeatures(degree=4)
X_poly=poly.fit_transform(X)
reg=LinearRegression()
kfold=KFold(n_splits=5,random_state=2024,shuffle=True)
results=cross_val_score(reg, X_poly,y,cv=kfold,scoring='r2')
print(results.mean())
#########################################################################
ridge=Ridge()
lasso=Lasso()
enet=ElasticNet()
param_grid=dict(alpha=[0.001,0.1,0.5,0.8,1,1.5,2])
kfold=KFold(n_splits=5,shuffle=True,random_state=(2024))

res=GridSearchCV(ridge,param_grid=param_grid, cv=kfold,scoring='r2')
res.fit(X,y)
res.cv_results_
res.best_params_
res.best_score_

res=GridSearchCV(ridge,param_grid=param_grid, cv=kfold,scoring='r2')
res.fit(X_poly,y)
res.cv_results_
res.best_params_
res.best_score_

res=GridSearchCV(lasso,param_grid=param_grid, cv=kfold,scoring='r2')
res.fit(X,y)
res.cv_results_
res.best_params_
res.best_score_

res=GridSearchCV(lasso,param_grid=param_grid, cv=kfold,scoring='r2')
res.fit(X_poly,y)
res.cv_results_
res.best_params_
res.best_score_

param_grid=dict(alpha=[0.001,0.1,0.5,0.8,1,1.5,2],l1_ratio=[0.001,0.5,0.7,1])
res=GridSearchCV(enet,param_grid=param_grid, cv=kfold,scoring='r2')
res.fit(X,y)
res.cv_results_
res.best_params_
res.best_score_

param_grid=dict(alpha=[0.001,0.1,0.5,0.8,1,1.5,2],l1_ratio=[0.001,0.5,0.7,1])
res=GridSearchCV(enet,param_grid=param_grid, cv=kfold,scoring='r2')
res.fit(X_poly,y)
res.cv_results_
res.best_params_
res.best_score_

#########################################
insurance_dummy = pd.read_csv(r"G:\Ddrive\PG DBDA\12 Practical Machine Learning_/Cases/Medical Cost Personal/insurance.csv")
y=insurance_dummy.iloc[:,-1]
X_dummy=insurance_dummy.iloc[:,:-1]
X=pd.get_dummies(X_dummy,drop_first=True)

X_poly=poly.fit_transform(X)

ins_dum=pd.read_csv(r"G:\Ddrive\PG DBDA\12 Practical Machine Learning_/Cases/Medical Cost Personal/tst_insure.csv")
ins=pd.get_dummies(ins_dum,drop_first=True)
poly=PolynomialFeatures(degree=2)
ins_poly=poly.fit_transform(ins)

param_grid=dict(alpha=[0.001,0.1,0.5,0.8,1,1.5,2],l1_ratio=[0.001,0.5,0.7,1])
kfold=KFold(n_splits=5,random_state=2024,shuffle=True)
res=GridSearchCV(enet,param_grid=param_grid, cv=kfold,scoring='r2')
res.fit(X_poly,y)
y_pred=res.predict(ins_poly)
res.cv_results_
res.best_params_
res.best_score_
#0.836148852945103

y_pred = pd.DataFrame(y_pred,columns=['charges'])
insurance = pd.concat([ins_dum,y_pred],axis=1)


