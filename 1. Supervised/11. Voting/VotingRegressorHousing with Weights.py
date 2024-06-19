import pandas as pd
import numpy as np

Housing = pd.read_csv(r"G:/Ddrive/PG DBDA/12 Practical Machine Learning_/Cases/Real Estate/Housing.csv")
X_dum = Housing.iloc[:,1:]
y = Housing.iloc[:,0]
X = pd.get_dummies(X_dum, drop_first=True)

from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from sklearn.ensemble import VotingRegressor


# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, 
                                                    random_state=2024)

dtr = DecisionTreeRegressor(max_depth=4)
lr = LinearRegression()
lasso = Lasso()

dtr.fit(X_train,y_train)
y_pred_dtr = dtr.predict(X_test)
e1 = r2_score(y_test,y_pred_dtr) #0.4880275113349174

lr.fit(X_train,y_train)
y_pred_lr = lr.predict(X_test)
e2 = r2_score(y_test,y_pred_lr) #0.658581846353582

lasso.fit(X_train,y_train)
y_pred_lasso = lasso.predict(X_test)
e3 = r2_score(y_test,y_pred_lasso) #0.6586012804207333

Voting = VotingRegressor(estimators=[('DT',dtr),
                                     ('LR',lr),('Lasso',lasso)])
Voting.fit(X_train,y_train)
y_pred = Voting.predict(X_test)
print(r2_score(y_test,y_pred)) #0.6564186380701841

###### Weighted Avg #########
w1 = e1/(e1+e2+e3)
w2 = e2/(e1+e2+e3)
w3 = e3/(e1+e2+e3)
Voting = VotingRegressor(estimators=[('DT',dtr),
                                     ('LR',lr),('Lasso',lasso)],
                         weights=np.array([w1,w2,w3]))

Voting.fit(X_train,y_train)
y_pred = Voting.predict(X_test)
print(r2_score(y_test,y_pred)) #0.6610195028937342













