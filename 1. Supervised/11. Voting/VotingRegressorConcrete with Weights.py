import pandas as pd
import numpy as np
Concrete = pd.read_csv(r'G:/Ddrive/PG DBDA/12 Practical Machine Learning_/Cases/Concrete Strength/Concrete_Data.csv')
X = Concrete.iloc[:,:-1]
y = Concrete.iloc[:,-1]

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, 
                                                    random_state=2024)

from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.ensemble import VotingRegressor
lin_reg = LinearRegression()
lasso = Lasso()
ridge = Ridge()
e_net = ElasticNet()

lin_reg.fit(X_train,y_train)
lasso.fit(X_train,y_train)
ridge.fit(X_train,y_train)
e_net.fit(X_train,y_train)

ypred_lin_reg = lin_reg.predict(X_test)
ypred_lasso = lasso.predict(X_test)
ypred_ridge = ridge.predict(X_test)
ypred_e_net = e_net.predict(X_test)

from sklearn.metrics import r2_score
e1 = r2_score(y_test, ypred_lin_reg) #0.6395029542692768
e2 = r2_score(y_test, ypred_lasso) #0.6410263671288359
e3 = r2_score(y_test, ypred_ridge) #0.6395045700087036
e4 = r2_score(y_test, ypred_e_net) #0.6407735025172949

models = [('LinearRegression',lin_reg),
('Lasso',lasso),
('Ridge',ridge),
('ElasticNet',e_net)]
vote = VotingRegressor(estimators=models)
vote.fit(X_train,y_train)
y_pred = vote.predict(X_test)
print(r2_score(y_test, y_pred)) #0.6402504780968605

###### Weighted Avg #########
w1 = e1/(e1+e2+e3+e4)
w2 = e2/(e1+e2+e3+e4)
w3 = e3/(e1+e2+e3+e4)
w4 = e4/(e1+e2+e3+e4)

vote = VotingRegressor(estimators=models,weights=np.array([w1,w2,w3,w4]))
vote.fit(X_train,y_train)
y_pred = vote.predict(X_test)
print(r2_score(y_test, y_pred)) #0.6402512584045086









