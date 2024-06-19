# Perform the necessary imports
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

concrete = pd.read_csv(r"G:/Ddrive/PG DBDA/12 Practical Machine Learning_/Cases/Concrete Strength/Concrete_Data.csv")
X = concrete.iloc[:,:-1]
y = concrete.iloc[:,-1]

scaler = StandardScaler()
XScaled=scaler.fit_transform(X)
####################  DecisionTreeRegressor ########################

depth_range = [None,7,3]
minsplit_range = [2,5,10]
minleaf_range = [1,5]

parameters = dict(max_depth=depth_range,
                  min_samples_split=minsplit_range, 
                  min_samples_leaf=minleaf_range)

from sklearn.model_selection import KFold
kfold = KFold(n_splits=5, random_state=2024,shuffle=True)
from sklearn.model_selection import GridSearchCV
clf = DecisionTreeRegressor(random_state=2024)
cv = GridSearchCV(clf, param_grid=parameters,
                  cv=kfold,scoring='r2',verbose=2)

cv.fit(XScaled,y)
# Best Parameters
print(cv.best_params_) #{'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2}

print(cv.best_score_) #0.8433331946156594


#############  DecisionTreeRegressor with KMeans  ####################

clustNos = [2,3,4,5,6,7,8,9,10]
Inertia = []

for i in clustNos :
    model = KMeans(n_clusters=i,random_state=2024)
    model.fit(XScaled)
    Inertia.append(model.inertia_)

plt.plot(clustNos, Inertia, '-o')
plt.title("Scree Plot")
plt.xlabel('Number of clusters, k')
plt.ylabel('Inertia')
plt.xticks(clustNos)
plt.show()

# Create a KMeans instance with clusters: Best k model
model = KMeans(n_clusters=6,random_state=2024)

# Fit model to points
model.fit(XScaled)

#model.n_init
# Determine the cluster labels of new_points: labels
labels = model.predict(XScaled)

clusterID = pd.DataFrame({'ClustID':labels})

XScaled_pd = pd.DataFrame(XScaled,columns=X.columns,index=X.index)
clusteredData = pd.concat([XScaled_pd,clusterID],
                          axis='columns')

clusteredData['ClustID'] = clusteredData['ClustID'].astype('category')

X_new = pd.get_dummies(clusteredData,drop_first=True)

depth_range = [None,7,3]
minsplit_range = [2,5,10]
minleaf_range = [1,5]

parameters = dict(max_depth=depth_range,
                  min_samples_split=minsplit_range, 
                  min_samples_leaf=minleaf_range)

from sklearn.model_selection import KFold
kfold = KFold(n_splits=5, random_state=2024,shuffle=True)
from sklearn.model_selection import GridSearchCV
clf = DecisionTreeRegressor(random_state=2024)
cv = GridSearchCV(clf, param_grid=parameters,
                  cv=kfold,scoring='r2',verbose=2)

cv.fit(clusteredData,y)
print(cv.best_params_) #{'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2}

print(cv.best_score_) #0.8482917260252989

#############  LinearRegression  ####################
from sklearn.model_selection import cross_val_score,KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


lin_reg=LinearRegression()
kfold=KFold(n_splits=5,shuffle=True,random_state=(2024))
res=cross_val_score(lin_reg,XScaled,y,cv=kfold,scoring='r2',verbose=2)
print(f"R2 score: {res.mean()}")
#R2 score: 0.604193074066034

#############  LinearRegression with KMeans  ####################
clustNos = [2,3,4,5,6,7,8,9,10]
Inertia = []

for i in clustNos :
    model = KMeans(n_clusters=i,random_state=2024)
    model.fit(XScaled)
    Inertia.append(model.inertia_)

plt.plot(clustNos, Inertia, '-o')
plt.title("Scree Plot")
plt.xlabel('Number of clusters, k')
plt.ylabel('Inertia')
plt.xticks(clustNos)
plt.show()

# Create a KMeans instance with clusters: Best k model
model = KMeans(n_clusters=6,random_state=2024)

# Fit model to points
model.fit(XScaled)

#model.n_init
# Determine the cluster labels of new_points: labels
labels = model.predict(XScaled)

clusterID = pd.DataFrame({'ClustID':labels})

XScaled_pd = pd.DataFrame(XScaled,columns=X.columns,index=X.index)
clusteredData = pd.concat([XScaled_pd,clusterID],
                          axis='columns')

clusteredData['ClustID'] = clusteredData['ClustID'].astype('category')

X_new = pd.get_dummies(clusteredData,drop_first=True)

lin_reg=LinearRegression()
kfold=KFold(n_splits=5,shuffle=True,random_state=(2024))
res=cross_val_score(lin_reg,X_new,y,cv=kfold,scoring='r2',verbose=2)
print(f"R2 score: {res.mean()}")
# R2 score: 0.6783883271645138

