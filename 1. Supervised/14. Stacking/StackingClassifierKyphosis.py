import pandas as pd

df = pd.read_csv("G:/Ddrive/PG DBDA/12 Practical Machine Learning_/Cases/Kyphosis/Kyphosis.csv")
dum_df = pd.get_dummies(df, drop_first=True)

X = dum_df.iloc[:,:-1].values
y = dum_df.iloc[:,-1].values

# Import the necessary modules
from sklearn.model_selection import train_test_split 

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, 
                                                    random_state=2024,
                                                    stratify=y)


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

from sklearn.svm import SVC
svc = SVC(probability = True,kernel='rbf',random_state=2024)

from sklearn.naive_bayes import GaussianNB
gaussian = GaussianNB()

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(random_state=2024,max_depth = 4)

from sklearn.ensemble import StackingClassifier
models_considered = [('Logistic Regression', logreg),
                     ('SVM', svc),('Naive Bayes',gaussian),
                     ('Decision Tree',dtc)]

from xgboost import XGBClassifier
clf = XGBClassifier(random_state=2024)

stack = StackingClassifier(estimators = models_considered,
                           final_estimator=clf,
                           stack_method="predict_proba")

stack.fit(X_train,y_train)

##################### Test set operations ###############################
#sklearn.model_selection import StratifiedKFold
#kfold = StratifiedKFold(n_splits=5, random_state=2020,shuffle=True)

y_pred_prob = stack.predict_proba(X_test)[:,1]

from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_test, y_pred_prob))

######## Include the original data along with predicted variables #####

stack = StackingClassifier(estimators = models_considered,
                           final_estimator=clf,
                           stack_method="predict_proba",
                           passthrough=True)

stack.fit(X_train,y_train)

######################## Test set Operations ##########################
y_pred_prob = stack.predict_proba(X_test)[:,1]

from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_test, y_pred_prob))


