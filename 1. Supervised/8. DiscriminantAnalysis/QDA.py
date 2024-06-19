import pandas as pd

df = pd.read_csv(r"G:\Ddrive\PG DBDA\12 Practical Machine Learning_/Cases/Satellite Imaging/Satellite.csv",sep=";")
#dum_df = pd.get_dummies(df.iloc[:,0:36], drop_first=True)

X = df.iloc[:,0:36]
y = df.iloc[:,36]

# Label Encoding for multi-class
## Unique classes
print(df.classes.unique())
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_le = le.fit_transform(y)

# Import the necessary modules
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_le, test_size = 0.3, 
                                                    random_state=2024, stratify=y_le)

qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train,y_train)
y_pred = qda.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test,y_pred)) #0.8513723459347489

# import matplotlib.pyplot as plt
# from sklearn.metrics import plot_confusion_matrix
# plot_confusion_matrix(da,X_test,y_test,labels=df.classes.unique())
# plt.show()

###################### ROC ###########################
y_pred_prob = qda.predict_proba(X_test)

from sklearn.metrics import roc_auc_score,log_loss
print(roc_auc_score(y_test,y_pred_prob,multi_class='ovr')) #0.9682297938798401
print(log_loss(y_test,y_pred_prob)) #0.9621284713539938


