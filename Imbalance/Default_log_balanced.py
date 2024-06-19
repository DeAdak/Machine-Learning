import pandas as pd

Default = pd.read_csv("G:\Ddrive\PG DBDA/12 Practical Machine Learning_/Datasets/Default.csv")
Default['default'].value_counts()
Default.student.value_counts()
dum_Default = pd.get_dummies(Default, drop_first=True)

## Considering only Student variable
#X=pd.DataFrame(dum_Default['student_Yes'])

X = dum_Default.iloc[:,[0,1,3]]
y = dum_Default.iloc[:,2]

# Import the necessary modules
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.2, 
                                                    random_state=2024)


# Create the classifier: logreg
logreg = LogisticRegression()
# Fit the classifier to the training data
logreg.fit(X_train,y_train)
# Predict the labels of the test set: y_pred
y_pred = logreg.predict(X_test)
# Compute and print the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
#               precision    recall  f1-score   support

#        False       0.97      1.00      0.98      1939
#         True       0.50      0.15      0.23        61

#     accuracy                           0.97      2000
#    macro avg       0.74      0.57      0.61      2000
# weighted avg       0.96      0.97      0.96      2000
print(accuracy_score(y_test,y_pred)) #0.9695

################ROC#############################

# Import necessary modules
from sklearn.metrics import roc_curve, roc_auc_score

# Compute predicted probabilities: y_pred_prob
y_pred_prob = logreg.predict_proba(X_test)[:,1]

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
roc_auc_score(y_test, y_pred_prob) # 0.8960170444457596


################### Over-Sampling(Naive) ###############

from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=2024)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

y_train.value_counts()
# default_Yes
# False    7728
# True      272

y_resampled.value_counts()
# default_Yes
# False    7728
# True     7728

##########################################################
# Create the classifier: logreg
logreg = LogisticRegression()
# Fit the classifier to the training data
logreg.fit(X_resampled, y_resampled)
# Predict the labels of the test set: y_pred
y_pred = logreg.predict(X_test)
# Compute and print the confusion matrix and classification report
#print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test,y_pred)) # 0.8425

###########################ROC#############################

# Import necessary modules
from sklearn.metrics import roc_curve, roc_auc_score

# Compute predicted probabilities: y_pred_prob
y_pred_prob = logreg.predict_proba(X_test)[:,1]

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
roc_auc_score(y_test, y_pred_prob) #0.9316362160654047

################# Over-Sampling(SMOTE) #################
# Synthetic Minority Oversampling Technique

from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=2024)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

y_train.value_counts()
# default_Yes
# False    7728
# True      272

y_resampled.value_counts()
# default_Yes
# False    7728
# True     7728

##########################################################
# Create the classifier: logreg
logreg = LogisticRegression()
# Fit the classifier to the training data
logreg.fit(X_resampled, y_resampled)
# Predict the labels of the test set: y_pred
y_pred = logreg.predict(X_test)
# Compute and print the confusion matrix and classification report
#print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test,y_pred)) #0.671

###########################ROC#############################

# Import necessary modules
from sklearn.metrics import roc_curve, roc_auc_score

# Compute predicted probabilities: y_pred_prob
y_pred_prob = logreg.predict_proba(X_test)[:,1]

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
roc_auc_score(y_test, y_pred_prob) # 0.8238656058979194

################# Over-Sampling(ADASYN) #################
# Adaptive Synthetic Minority Oversampling Technique
from imblearn.over_sampling import ADASYN
adasyn = ADASYN(random_state=2024)
X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)
y_train.value_counts()
# default_Yes
# False    7728
# True      272

y_resampled.value_counts()
# default_Yes
# False    7728
# True     7688

##########################################################
# Create the classifier: logreg
logreg = LogisticRegression()
# Fit the classifier to the training data
logreg.fit(X_resampled, y_resampled)
# Predict the labels of the test set: y_pred
y_pred = logreg.predict(X_test)
# Compute and print the confusion matrix and classification report
#print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test,y_pred)) #0.662

###########################ROC#############################

# Import necessary modules
from sklearn.metrics import roc_curve, roc_auc_score

# Compute predicted probabilities: y_pred_prob
y_pred_prob = logreg.predict_proba(X_test)[:,1]

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
roc_auc_score(y_test, y_pred_prob) #0.8264696184445252


