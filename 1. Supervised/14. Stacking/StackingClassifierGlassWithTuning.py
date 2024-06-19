import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,KFold,GridSearchCV,StratifiedKFold
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import log_loss


glass=pd.read_csv("G:/Ddrive/PG DBDA/12 Practical Machine Learning_/Cases/Glass Identification/Glass.csv")
F = glass.iloc[:,:-1]
R_le = glass.iloc[:,-1]
le = LabelEncoder()
R = le.fit_transform(R_le)
X_train,X_test,y_train,y_test=train_test_split(F,R,test_size=0.2,stratify=R,random_state=2024)
kfold=StratifiedKFold(n_splits=(5),shuffle=(True),random_state=(2024))

log_reg=LogisticRegression()
params={'solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],'multi_class':['ovr','auto','multinomial']}
cv=GridSearchCV(log_reg,param_grid=params,cv=kfold,scoring='neg_log_loss',verbose=2) 
cv.fit(X_train,y_train)
print(cv.best_params_) # {'multi_class': 'auto', 'solver': 'lbfgs'} 
print(cv.best_score_) # -1.00462441155714
log_reg_best=cv.best_estimator_  
    
svc_lin=SVC(kernel='linear',random_state=2024,probability=(True))
params={'C':[0.001,0.05,0.5,1,1.5]}
cv=GridSearchCV(svc_lin,param_grid=params,scoring='neg_log_loss',cv=kfold,verbose=2)
cv.fit(X_train,y_train)
print(cv.best_params_) #{'C': 1}
print(cv.best_score_) # -1.012798518809061
svc_lin_best=cv.best_estimator_

svc_rad=SVC(random_state=2024,probability=(True))
params={'C':[0.001,0.05,0.5,1,1.5],'gamma':['auto','scale']}
cv=GridSearchCV(svc_rad,param_grid=params,scoring='neg_log_loss',cv=kfold,verbose=2)
cv.fit(X_train,y_train)
print(cv.best_params_) # {'C': 1.5, 'gamma': 'auto'}
print(cv.best_score_) # -0.8899492137433815
svc_rad_best=cv.best_estimator_

dtc=DecisionTreeClassifier(random_state=(2024))
params={'max_depth':[None,4],'min_samples_split':[2,10],'min_samples_leaf':[1,10]}
cv=GridSearchCV(dtc,param_grid=params,cv=kfold,scoring='neg_log_loss',verbose=2)
cv.fit(X_train,y_train)
print(cv.best_params_) #{'max_depth': 4, 'min_samples_leaf': 10, 'min_samples_split': 2}
print(cv.best_score_) # -3.6277187461988674
dtc_best=cv.best_estimator_

knn=KNeighborsClassifier()
params={'n_neighbors':[1,3,5,7]}
cv=GridSearchCV(knn,param_grid=params,cv=kfold,scoring='neg_log_loss',verbose=2)
cv.fit(X_train,y_train)
print(cv.best_params_) #{'n_neighbors': 7}
print(cv.best_score_) # -3.5031505293902425
knn_best=cv.best_estimator_

xgbc=XGBClassifier(random_state=2024)
params={'n_estimators':[ 10,20,30 ],'Learning_rate' : [ 0.001,0.1,0.5],'max_depth' : [4,10]}
cv=GridSearchCV(xgbc,param_grid=params,scoring='neg_log_loss',cv=kfold,verbose=2)
cv.fit(X_train,y_train)
print(cv.best_params_) # {'Learning_rate': 0.001, 'max_depth': 4, 'n_estimators': 20}
print(cv.best_score_) # -0.7402065065644423
xgbc_best=cv.best_estimator_

stack=StackingClassifier(stack_method='predict_proba',estimators=[('LOG_REG',log_reg_best),
                                                                  ('SVC_RAD',svc_rad_best),
                                                                  ('SVD_LIN',svc_lin_best),
                                                                  ('DTC',dtc_best),
                                                                  ('KNN',knn_best)],final_estimator=xgbc)
stack.fit(X_train,y_train)
y_pred_stack=stack.predict(X_test)
y_pred_stack_prob=stack.predict_proba(X_test)
print(log_loss(y_test,y_pred_stack_prob)) # 1.3339264058599332

stack=StackingClassifier(stack_method='predict_proba',estimators=[('LOG_REG',log_reg_best),
                                                                  ('SVC_RAD',svc_rad_best),
                                                                  ('SVD_LIN',svc_lin_best),
                                                                  ('DTC',dtc_best),
                                                                  ('KNN',knn_best)],final_estimator=xgbc,passthrough=(True))
stack.fit(X_train,y_train)
y_pred_stack_prob=stack.predict_proba(X_test)
print(log_loss(y_test,y_pred_stack_prob)) # 0.9926613680574498
