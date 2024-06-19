# Import the necessary modules
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier,plot_tree

df = pd.read_csv(r"G:\Ddrive\PG DBDA\12 Practical Machine Learning_/Cases/Wisconsin/BreastCancer.csv")
df.Class.value_counts()
dum_df = pd.get_dummies(df, drop_first=True)

X = dum_df.iloc[:,1:-1]
y = dum_df.iloc[:,-1]

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, 
                                                    random_state=2024,
                                                    stratify=y)

clf = DecisionTreeClassifier(max_depth=3,random_state=2024)
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test,y_pred))

plot_tree(clf,feature_names=X_train.columns,
               class_names=['Benign','Malignant'],
               filled=True,fontsize=5) 

################# Only can run on 0.22 and above #################
from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(clf,X_test,y_test,display_labels=['Benign','Malignant'])
################ROC##############################

# Import necessary modules
from sklearn.metrics import roc_curve, roc_auc_score

# Compute predicted probabilities: y_pred_prob
y_pred_prob = clf.predict_proba(X_test)[:,1]

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Plot ROC curve 
import matplotlib.pyplot as plt
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
roc_auc_score(y_test, y_pred_prob) #0.9528985507246377

################################################################
import graphviz 
from sklearn import tree

dot_data = tree.export_graphviz(clf, out_file=None, 
                         feature_names=X_train.columns,  
                         class_names=['Benign','Malignant'],  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = graphviz.Source(dot_data)  
graph 

# or
plt.figure(figsize=(15,10))
plot_tree(clf,feature_names=X_train.columns,
               class_names=['Benign','Malignant'],
               filled=True,fontsize=5)
plt.show()

####################### Grid Search CV ###########################
depth_range = [3,4,5,6,7,8,9]
minsplit_range = [5,10,20,25,30]
minleaf_range = [5,10,15]

parameters = dict(max_depth=depth_range,
                  min_samples_split=minsplit_range, 
                  min_samples_leaf=minleaf_range)

from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=5, random_state=2024,
                        shuffle=True)

from sklearn.model_selection import GridSearchCV
clf = DecisionTreeClassifier(random_state=2024)
cv = GridSearchCV(clf, param_grid=parameters,
                  cv=kfold,scoring='roc_auc',verbose=2)

cv.fit(X,y)

# Viewing all parameter sets
df_cv = pd.DataFrame(cv.cv_results_)

# Best Parameters
print(cv.best_params_) # {'max_depth': 4, 'min_samples_leaf': 10, 'min_samples_split': 5}

print(cv.best_score_) #0.9859106805646292

best_model = cv.best_estimator_
from sklearn import tree
dot_data = tree.export_graphviz(best_model, out_file=None, 
                         feature_names=X.columns,  
                         class_names=['Benign','Malignant'],  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = graphviz.Source(dot_data)  
graph 

#OR

import matplotlib.pyplot as plt
plt.figure(figsize=(15,10))
tree.plot_tree(best_model,feature_names=X.columns,
               class_names=['Benign','Malignant'],
               filled=True,fontsize=10,rounded=True)
plt.show()

########################################################
import matplotlib.pyplot as plt
best_model = cv.best_estimator_
print(best_model.feature_importances_)

ind = np.arange(9)
plt.figure(figsize=(15,10))
plt.bar(ind,best_model.feature_importances_)
plt.xticks(ind,(X.columns),rotation=45)
plt.title('Feature Importance')
plt.xlabel("Variables")
plt.show()
#######################################################