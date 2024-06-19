import pandas as pd
train=pd.read_csv(r'G:/Ddrive/PG DBDA/12 Practical Machine Learning_/7. Regularized _ Polynomial Regression/bike-sharing-demand/train.csv',parse_dates=['datetime'])
train['year']=train['datetime'].dt.year
train['month']=train['datetime'].dt.month
train['day']=train['datetime'].dt.day
train['hour']=train['datetime'].dt.hour
train['season']=train['season'].astype('category')
train['weather']=train['weather'].astype('category')

X=train.drop(['datetime','casual','registered','count'],axis=1)
Y = train['count']
X_dum = pd.get_dummies(X,drop_first=True)

# GridSearch CV with 
# Ridge: alpha=[0,0.001,0.1,0.5,1,1.5,2]
# best_model = cv.best_estimator

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV,KFold
kfold=KFold(n_splits=5,shuffle=True,random_state=2024)
ridge = Ridge()
params = {'alpha':[0,0.001,0.1,0.5,1,1.5,2]}
res=GridSearchCV(ridge, param_grid=params,scoring='r2',cv=kfold)
res.fit(X_dum,Y)
res.best_score_
res.best_params_


test=pd.read_csv(r'G:/Ddrive/PG DBDA/12 Practical Machine Learning_/7. Regularized _ Polynomial Regression/bike-sharing-demand/test.csv',parse_dates=['datetime'])
test['year']=test['datetime'].dt.year
test['month']=test['datetime'].dt.month
test['day']=test['datetime'].dt.day
test['hour']=test['datetime'].dt.hour
test['season']=test['season'].astype('category')
test['weather']=test['weather'].astype('category')

X=test.drop(['datetime'],axis=1)
X_dum = pd.get_dummies(X,drop_first=True)
y_pred=res.predict(X_dum)
import numpy as np
y_pred=np.where(y_pred<0,0,y_pred)
y_pred=np.round(y_pred)

datetime=test['datetime']
submit=pd.DataFrame({'datetime':datetime,'count':y_pred})
submit.to_csv(r'G:/Ddrive/PG DBDA/12 Practical Machine Learning_/7. Regularized _ Polynomial Regression/bike-sharing-demand/submit.csv',index=False)


