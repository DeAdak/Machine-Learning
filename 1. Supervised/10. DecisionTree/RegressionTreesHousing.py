import pandas as pd
import numpy as np

Housing = pd.read_csv(r"G:\Ddrive\PG DBDA\12 Practical Machine Learning_\Cases\Real Estate\Housing.csv")
dum_Housing = pd.get_dummies(Housing.iloc[:,1:11], drop_first=True)

from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeRegressor,plot_tree
X = dum_Housing
y = Housing.iloc[:,0]

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, 
                                                    random_state=2024)

dtr = DecisionTreeRegressor(max_depth=3,random_state=2024)
cv = dtr.fit(X_train, y_train)
X.columns
columns1 = ['lotsize', 'bedrooms', 'bathrms', 'stories', 'garagepl', 'driveway_yes',
       'recroom_yes', 'fullbase_yes', 'gashw_yes', 'airco_yes']

import matplotlib.pyplot as plt
plt.figure(figsize=(15,10))
plot_tree(cv.best_estimator_)
plt.show()

###################################################################
import graphviz 
from sklearn import tree
# =============================================================================
# dot_data = tree.export_graphviz(clf2, out_file=None) 
# graph = graphviz.Source(dot_data) 
# graph.render("Housing") 
# =============================================================================
dot_data = tree.export_graphviz(cv, out_file=None, 
                         feature_names=list(X_train),  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = graphviz.Source(dot_data)  
graph 

###################################################################

y_pred = cv.predict(X_test)

from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
print(mean_squared_error(y_test, y_pred) ) #358913364.614852
print(mean_absolute_error(y_test, y_pred)) #14612.35920524073
print(r2_score(y_test, y_pred)) #0.48189495456660214

#######################Grid Search CV#############################
depth_range = [3,4,5,6,7,8,9]
minsplit_range = [5,10,20,25,30]
minleaf_range = [5,10,15]

parameters = dict(max_depth=depth_range,
                  min_samples_split=minsplit_range, 
                  min_samples_leaf=minleaf_range)

from sklearn.model_selection import KFold
kfold = KFold(n_splits=5, random_state=2024,shuffle=True)
from sklearn.model_selection import GridSearchCV
clf = DecisionTreeRegressor(random_state=2024)
cv = GridSearchCV(clf, param_grid=parameters,
                  cv=kfold,scoring='r2',verbose=2)

cv.fit(X,y)
# Best Parameters
print(cv.best_params_) #{'max_depth': 7, 'min_samples_leaf': 10, 'min_samples_split': 5}

print(cv.best_score_)#0.4863002978230278

######################################################################
best_model = cv.best_estimator_
import matplotlib.pyplot as plt

best_model.feature_importances_

ind = np.arange(X.shape[1])
plt.figure(figsize=(15,10))
plt.bar(ind,best_model.feature_importances_)
plt.xticks(ind,(X.columns),rotation=45)
plt.title('Feature Importance')
plt.xlabel("Variables")
plt.show()
########################################################################